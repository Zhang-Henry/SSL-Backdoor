import sys
sys.path.append("..")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import numpy as np


from pytorch_ssim import SSIM
from tqdm import tqdm

from PIL import Image
from func import CMD, SinkhornDistance, MMD_loss
import lpips
from network import U_Net,R2AttU_Net,R2U_Net,AttU_Net
from data_loader import aug
from moco.builder import MoCo
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights,vit_l_16,ViT_L_16_Weights,vit_b_16,ViT_B_16_Weights
import torch.nn.functional as F
from collections import OrderedDict
from torchmetrics.image import PeakSignalNoiseRatio

class Solver():
    def __init__(self, args, train_loader):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.resume:
            self.net=torch.load(args.resume)
        else:
            self.net = AttU_Net(img_ch=3,output_ch=3).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        self.train_loader = train_loader

        self.mse = nn.MSELoss()
        self.ssim = SSIM()
        self.loss_cmd = CMD()
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.loss_mmd = MMD_loss()
        self.WD=SinkhornDistance(eps=0.1, max_iter=100)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        # self.backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device).eval()
        # self.backbone=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1).to(self.device).eval()

        self.moco=MoCo(
            models.__dict__['resnet18'],
            128, 65536, 0.999,
            contr_tau=0.2,
            align_alpha=None,
            unif_t=None,
            unif_intra_batch=True,
            mlp=True).to(self.device)

        checkpoint = torch.load('../moco/save/custom_imagenet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar', map_location=torch.device('cuda:0'))
        state_dict =checkpoint['state_dict']

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' in k:
                k = k.split('.')[1:]
                k = '.'.join(k)
            new_state_dict[k]=v

        self.moco.load_state_dict(new_state_dict)
        self.backbone=self.moco.encoder_q


    def train(self,args):
        print('Start training...')

        bar=tqdm(range(1, args.n_epoch+1))
        recorder=Recorder(args)

        for _ in bar:
            total_loss,total_wd,total_ssim,total_sim,total_psnr,total_lp, total_far = [],[],[],[],[],[],[]

            for img in self.train_loader:
                img = img.to(self.device)

                # 将滤镜作用在输入图像上
                filter_img = self.net(img)

                # 将滤镜作用在backdoor的图像上
                scaled_images = (filter_img.cpu().detach().numpy() * 255).astype(np.uint8)
                aug_filter_img_list=[]
                for scale_img in scaled_images:
                    scaled_filter_img = Image.fromarray(np.transpose(scale_img,(1,2,0))).convert('RGB')
                    aug_filter_img_list.append(aug(scaled_filter_img))

                aug_filter_img = torch.stack(aug_filter_img_list)
                aug_filter_img=aug_filter_img.cuda()

                if args.use_feature:
                    filter_img_feature = self.backbone(filter_img)
                    filter_img_feature = F.normalize(filter_img_feature, dim=1)
                    aug_filter_img_feature = self.backbone(aug_filter_img)
                    aug_filter_img_feature = F.normalize(aug_filter_img_feature, dim=1)
                    wd,_,_=self.WD(filter_img_feature,aug_filter_img_feature) # wd越小越相似
                    wd_p,_,_=self.WD(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1))
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
                ############################ wd ############################
                else:
                    loss_sim = 1 - loss_ssim + 10 * lp_loss.mean() - 0.05 * loss_psnr
                    loss_far = - recorder.cost * wd
                    loss = loss_sim + loss_far

                    # loss = 10 * loss_mse + 1 - loss_ssim + 10 * lp_loss.mean() - recorder.cost * wd
                    # loss = 0.00001 * loss_mse + 1 - loss_ssim

                ############################ cmd ############################
                # loss = 0.1*loss_mse2 + 3*(1 - loss_ssim2) - 0.005*cmd

                ############################ mmd ############################
                # loss = loss_mse2 + 1 - loss_ssim2 - mmd

                ############################ lpips ############################
                # lp_loss=d_list.squeeze()
                # loss = loss_mse2 + 3*(1 - loss_ssim2) - lp_loss

                # 梯度清零
                self.optimizer.zero_grad()

                # 计算损失函数相对于滤波器参数的梯度
                loss.backward()

                self.optimizer.step()
                total_loss.append(loss.item())
                total_ssim.append(loss_ssim.item())
                total_sim.append(loss_sim.item())
                total_wd.append(wd.item())
                total_psnr.append(loss_psnr.item())
                total_lp.append(lp_loss.mean().item())
                total_far.append(loss_far.item())

            self.scheduler.step()
            # 计算平均损失
            size=len(self.train_loader)
            avg_loss=sum(total_loss)/size
            wd=sum(total_wd)/size
            sim=sum(total_sim)/size
            ssim=sum(total_ssim)/size
            psnr=sum(total_psnr)/size
            lp=sum(total_lp)/size
            far=sum(total_far)/size

            # torch.save(self.net, f'trigger/moco/{self.args.timestamp}/ssim{ssim:.4f}_wd{wd:.1f}.pt')
            if ssim >= args.ssim_threshold and wd >= recorder.best:
                torch.save(self.net, f'trigger/moco/{self.args.timestamp}/ssim{ssim:.4f}_psnr{psnr:.2f}_wd{wd:.3f}.pt')

                recorder.best = wd
                print('\n--------------------------------------------------')
                print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, Best WD: {wd}")
                print('--------------------------------------------------')
                recorder.cost_up_counter = 0
                recorder.cost_down_counter = 0


            if recorder.cost == 0 and ssim >= args.ssim_threshold and wd >= recorder.best:
                recorder.cost_set_counter += 1
                if recorder.cost_set_counter >= args.patience:
                    recorder.reset_state(args)
            else:
                recorder.cost_set_counter = 0

            if ssim >= args.ssim_threshold:
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
                bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim}, far:{far}, WD: {wd}, WD_p:{wd_p}, SSIM: {ssim}, pnsr:{psnr}, lp_loss:{lp_loss.mean()},  cost:{recorder.cost}")
            else:
                bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim}, far:{far}, WD: {wd},  SSIM: {ssim}, cost:{recorder.cost}, lp_loss:{lp_loss.mean()}")



class Recorder:
    def __init__(self, args):
        super().__init__()

        # Best optimization results
        # self.mask_best = None
        # self.pattern_best = None
        self.best = -float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.best

        # Cost
        self.cost = args.init_cost
        self.cost_multiplier_up = args.cost_multiplier_up
        self.cost_multiplier_down = args.cost_multiplier_down

    def reset_state(self, args):
        self.cost = args.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))




