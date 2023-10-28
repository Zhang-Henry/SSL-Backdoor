import sys
sys.path.append("..")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch,lpips
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import MSELoss,Identity
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights,vit_l_16,ViT_L_16_Weights,vit_b_16,ViT_B_16_Weights,swin_s,Swin_S_Weights
import numpy as np
from pytorch_ssim import SSIM
from tqdm import tqdm
from PIL import Image
from utils import *
from network import U_Net,R2AttU_Net,R2U_Net,AttU_Net
from data_loader import aug
from moco.builder import MoCo
from collections import OrderedDict
from torchmetrics.image import PeakSignalNoiseRatio
from simclr_converter.resnet_wider import resnet50x1, resnet50x2, resnet50x4


class Solver():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = AttU_Net(img_ch=3,output_ch=3).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        self.mse = MSELoss()
        self.ssim = SSIM()
        self.loss_cmd = CMD()
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.loss_mmd = MMD_loss()
        self.WD=SinkhornDistance(eps=0.1, max_iter=100)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        # self.backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device).eval()
        # self.backbone=vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(self.device).eval()
        # self.backbone=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(self.device).eval()
        # self.backbone=swin_s(weights=Swin_S_Weights.IMAGENET1K_V1).to(self.device).eval()
        # self.backbone.head=Identity()

        # self.backbone = resnet50x2().to(self.device) # simclr
        # sd = torch.load('../simclr_converter/resnet50-2x.pth', map_location=torch.device('cuda:0'))
        # self.backbone.load_state_dict(sd['state_dict'])

        ############# moco trained by myself #################
        # self.moco=MoCo(
        #     models.__dict__['resnet18'],
        #     128, 65536, 0.999,
        #     contr_tau=0.2,
        #     align_alpha=None,
        #     unif_t=None,
        #     unif_intra_batch=True,
        #     mlp=True).to(self.device) # moco trained by myself

        # checkpoint = torch.load('../moco/save/custom_imagenet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar', map_location=torch.device('cuda:0'))
        # state_dict =checkpoint['state_dict']

        # new_state_dict = OrderedDict()

        # for k, v in state_dict.items():
        #     if 'module' in k:
        #         k = k.split('.')[1:]
        #         k = '.'.join(k)
        #     new_state_dict[k]=v

        # self.moco.load_state_dict(new_state_dict)
        # self.backbone=self.moco.encoder_q

        ############### moco pretrained https://github.com/facebookresearch/moco ##############
        model = models.__dict__['resnet50']()

        checkpoint = torch.load('/home/hrzhang/projects/SSL-Backdoor/moco/save/moco_v2_800ep_pretrain.pth.tar')
        state_dict = checkpoint["state_dict"]

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        self.backbone = model.to(self.device).eval()

    def train(self,args,train_loader):
        print('Start training...')

        bar=tqdm(range(1, args.n_epoch+1))
        recorder=Recorder(args)
        tracker=Loss_Tracker()
        # 恢复模型和优化器状态
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            recorder.best = checkpoint['best']
            print(f"Resuming training from {args.resume}")

        for _ in bar:
            self.train_one_epoch(args,recorder,bar,tracker,train_loader)


    def train_one_epoch(self,args,recorder,bar,tracker,train_loader):
        tracker.reset() # 重置损失记录器

        for img,label in train_loader:
            img = img.to(self.device)

            # 将滤镜作用在输入图像上
            filter_img = self.net(img)

            # 将滤镜作用在backdoor的图像上
            scaled_images = (filter_img.cpu().detach().numpy() * 255).astype(np.uint8)
            aug_filter_img_list=[]
            for scale_img in scaled_images:
                scaled_filter_img = Image.fromarray(np.transpose(scale_img,(1,2,0))).convert('RGB')
                aug_filter_img_list.append(aug(scaled_filter_img)) # 对backdoor图片进行transform

            aug_filter_img = torch.stack(aug_filter_img_list)
            aug_filter_img=aug_filter_img.cuda()

            if args.use_feature:
                filter_img_feature = self.backbone(filter_img)
                filter_img_feature = F.normalize(filter_img_feature, dim=1)
                aug_filter_img_feature = self.backbone(aug_filter_img)
                aug_filter_img_feature = F.normalize(aug_filter_img_feature, dim=1)
                wd,_,_=self.WD(filter_img_feature,aug_filter_img_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离
                # wd_p,_,_=self.WD(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1))
            else:
                wd,_,_=self.WD(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1)) # wd越小越相似

            # cmd=loss_cmd(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1),5)

            # mmd=loss_mmd(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1))


            # filter后的图片和原图的mse和ssim，差距要尽可能小
            loss_mse = self.mse(filter_img, img)
            loss_psnr = self.psnr(filter_img, img)
            loss_ssim = self.ssim(filter_img, img)


            d_list = self.loss_fn(filter_img,img)
            lp_loss=d_list.squeeze()

            # wd = 0.0002 * wd_p + wd_f # wd_pixel wd_feature

            ############################ wd ############################
            if args.ablation:
                loss = recorder.cost * loss_mse + 1 - loss_ssim + recorder.cost * lp_loss.mean()
            else:
                loss_sim = 1 - loss_ssim + 10 * lp_loss.mean() - 0.05 * loss_psnr
                loss_far = - recorder.cost * wd
                loss = loss_sim + loss_far

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            tracker.update(loss.item(),wd.item(),loss_ssim.item(),loss_psnr.item(),lp_loss.mean().item(),loss_mse.item(),loss_sim.item(),loss_far.item())

        self.scheduler.step()
        # 计算平均损失

        avg_loss,wd,ssim,psnr,lp,mse,sim,far = tracker.get_avg_loss()


        # torch.save(self.net, f'trigger/moco/{self.args.timestamp}/ssim{ssim:.4f}_wd{wd:.1f}.pt')
        if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold and wd >= recorder.best:
            state = {
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best': recorder.best
            }
            torch.save(state, f'trigger/moco/{self.args.timestamp}/ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}.pt')

            recorder.best = wd
            print('\n--------------------------------------------------')
            print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, Best WD: {wd}")
            print('--------------------------------------------------')
            recorder.cost_up_counter = 0
            recorder.cost_down_counter = 0


        if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= args.patience:
            recorder.cost_up_counter = 0
            print('\n--------------------------------------------------')
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            print('--------------------------------------------------')

            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= args.patience:
            recorder.cost_down_counter = 0
            print('\n--------------------------------------------------')
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            print('--------------------------------------------------')
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True


        if args.use_feature:
            bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim}, far:{far}, WD: {wd}, SSIM: {ssim}, pnsr:{psnr}, lp:{lp},mse:{mse},  cost:{recorder.cost}")
        else:
            bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim}, far:{far}, WD: {wd},  SSIM: {ssim}, cost:{recorder.cost}, lp:{lp.mean()}")
