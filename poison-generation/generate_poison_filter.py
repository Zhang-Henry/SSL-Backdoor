'''
This script generates poisoned data.

Author: Aniruddha Saha
'''
import sys
sys.path.append("..")

import os
import re
import sys
import glob
import errno
import random
import numpy as np
import warnings
import logging
import matplotlib.pyplot as plt
import configparser
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import pilgram
# from skimage import img_as_ubyte
from optimize_filter.network import U_Net

config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options     = config["poison_generation"]
data_root	= options["data_root"]
seed        = None
text        = options.getboolean("text")
fntsize     = int(options["fntsize"])
trigger     = options["trigger"]
targeted    = options.getboolean("targeted")
target_wnid = options["target_wnid"]
poison_injection_rate = float(options["poison_injection_rate"])
poison_savedir = options["poison_savedir"].format(experimentID)
logfile     = options["logfile"].format(experimentID)
splits      = [split for split in options["splits"].split(",")]

os.makedirs(poison_savedir, exist_ok=True)
os.makedirs("data/{}".format(experimentID), exist_ok=True)

#logging
os.makedirs(os.path.dirname(logfile), exist_ok=True)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
    logging.FileHandler(logfile, "w"),
    logging.StreamHandler()
])

def main():
    with open('/home/hrzhang/projects/SSL-Backdoor/poison-generation/scripts/imagenet100_classes.txt', 'r') as f:
        class_list = [l.strip() for l in f.readlines()]

    # # Comment lines above and uncomment if you are using Full ImageNet and provide path to train/val folder of ImageNet.
    # class_list = os.listdir('/datasets/imagenet/train')

    logging.info("Experiment ID: {}".format(experimentID))

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    generate_poison(class_list, data_root, poison_savedir, splits=splits)
    # # Debug: If you want to run for one image.
    # file = "<imagenet100-root>/imagenet100/val/n01558993/ILSVRC2012_val_00001598.JPEG"
    # file = "<imagenet100-root>/imagenet100/val/n01558993/ILSVRC2012_val_00029627.JPEG"
    # poisoned_image = add_watermark(file,
    #                                 trigger,
    #                                 val=True
    #                                 )
    # # poisoned_image.show()
    # poisoned_image.save("test.png")

def add_watermark(net,input_image_path,val=False):


    trans = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])


    base_image = Image.open(input_image_path).convert('RGB')

    # if val:
    #     # preprocess validation images
    #     base_image = trans_val(base_image)
    # else:
    #     base_image = trans_train(base_image)

    base_image = trans(base_image)

    # base_image = Image.fromarray(na.astype(np.uint8))
    # base_image.show()

    ###########################
    ### for ins filter only ###
    # na = np.array(base_image).astype(float)

    # img_backdoor = pilgram.kelvin(base_image)

    ###########################
    ## filter by covolution ###
    # na = np.array(base_image).astype(float)
    # filter = torch.load('/home/hrzhang/projects/SSL-Backdoor/optimize_filter/trigger/moco/filter_2023-09-17-11-38-41.pt', map_location=torch.device('cpu'))
    # img=torch.Tensor(na)
    # backdoored_image = F.conv2d(img.permute(2, 0, 1), filter, padding=7//2)
    # img_backdoor = backdoored_image.permute(1,2,0).detach().numpy()
    # ##### scale to 0-255 #####
    # maxv=img_backdoor.max()
    # minv=img_backdoor.min()
    # scaled_image = 255 * (img_backdoor - minv) / (maxv - minv)
    # scaled_image = np.round(scaled_image).astype(np.uint8)
    # #####
    # img_backdoor = Image.fromarray(scaled_image)

    ###########################
    ### for custom filter of unet only ###
    # na = np.array(base_image)

    # unet = U_Net(img_ch=3,output_ch=3)

    unet=net.cuda()

    img=base_image.cuda()
    # if hasattr(unet, 'sig'):  # 假设最后一层被命名为 'fc_layer'
    #     delattr(unet, 'sig')
    backdoored_image=unet(img.unsqueeze(0))

    img_backdoor = backdoored_image.squeeze()

    # img_backdoor = np.clip(img_backdoor, -3, 3) #限制颜色范围在0-1

    sig = nn.Sigmoid()
    img_backdoor = sig(img_backdoor)

    scaled_image = (img_backdoor.cpu().detach().numpy() * 255).astype(np.uint8)
    img_backdoor = Image.fromarray(np.transpose(scaled_image,(1,2,0)))
    ###########################

    transparent = img_backdoor.convert('RGB')

    # transparent.show()

    return transparent



def generate_poison(class_list, source_path, poisoned_destination_path,
                  splits=['train', 'val', 'val_poisoned']):

    unet = torch.load(trigger)

    # sort class list in lexical order
    class_list = sorted(class_list)

    for split in splits:
        if split == "train":

            train_filelist = "data/{}/train/rate_{:.2f}_targeted_{}_filelist.txt".format(experimentID,
                                                                                            poison_injection_rate,
                                                                                            targeted)
            print(train_filelist)
            # if os.path.exists(train_filelist):
            #     logging.info("train filelist already exists. please check your config.")
            #     sys.exit()
            # else:
            os.makedirs(os.path.dirname(train_filelist), exist_ok=True)
            f_train = open(train_filelist, "w")

        if split == "val_poisoned":
            val_poisoned_filelist = "data/{}/val_poisoned/filelist.txt".format(experimentID)
            print(val_poisoned_filelist)
            # if os.path.exists(val_poisoned_filelist):
            #     logging.info("val filelist already exists. please check your config.")
            #     sys.exit()
            # else:
            os.makedirs(os.path.dirname(val_poisoned_filelist), exist_ok=True)
            f_val_poisoned = open(val_poisoned_filelist, "w")

    source_path = os.path.abspath(source_path)
    poisoned_destination_path = os.path.abspath(poisoned_destination_path)
    os.makedirs(poisoned_destination_path, exist_ok=True)
    train_filelist = list()

    # for split in splits:
    for class_id, c in enumerate(tqdm(class_list)):
        if re.match(r"n[0-9]{8}", c) is None:
            raise ValueError(
                f"Expected class names to be of the format nXXXXXXXX, where "
                f"each X represents a numerical number, e.g., n04589890, but "
                f"got {c}")
        for split in splits:
            if split == 'train':
                os.makedirs(os.path.join(poisoned_destination_path, split, "rate_{:.2f}_targeted_{}".format(poison_injection_rate,targeted),c), exist_ok=True)

                if targeted:
                    filelist = sorted(glob.glob(os.path.join(source_path, split , c, "*")))
                    filelist = [file+" "+str(class_id) for file in filelist]
                    if c == target_wnid:
                        train_filelist = train_filelist + filelist
                    else:
                        for file_id, file in enumerate(filelist):
                            f_train.write(file + "\n")

                else:
                    filelist = sorted(glob.glob(os.path.join(source_path, split , c, "*")))
                    filelist = [file+" "+str(class_id) for file in filelist]
                    train_filelist = train_filelist + filelist

            elif split == 'val':
                pass

            elif split == 'val_poisoned':
                os.makedirs(os.path.join(poisoned_destination_path, split , c), exist_ok=True)
                filelist = sorted(glob.glob(os.path.join(source_path, 'val', c, "*")))
                filelist = [file+" "+str(class_id) for file in filelist]

                for file_id, file in enumerate(filelist):
                    # add watermark
                    poisoned_image = add_watermark(unet,file.split()[0],val=True)
                    poisoned_file = file.split()[0].replace(os.path.join(source_path, 'val'),
                                                            os.path.join(poisoned_destination_path,split))
                    if file_id < 2 and class_id < 1:
                        # logging.info("Check")
                        poisoned_image.show()
                        # sys.exit()
                    poisoned_image.save(poisoned_file)
                    f_val_poisoned.write(poisoned_file + " " + file.split()[1] + "\n")
            else:
                logging.info("Invalid split. Exiting.")
                sys.exit()

    if train_filelist:
        # randomly choose out of full list - untargeted or target class list - targeted
        random.shuffle(train_filelist)
        len_poisoned = int(poison_injection_rate*len(train_filelist))
        logging.info("{} training images are being poisoned.".format(len_poisoned))
        for file_id, file in enumerate(tqdm(train_filelist)):
            if file_id < len_poisoned:
                # add watermark
                poisoned_image = add_watermark(unet,file.split()[0])


                poisoned_file = file.split()[0].replace(os.path.join(source_path, 'train'),
                                                                os.path.join(poisoned_destination_path,
                                                                "train",
                                                                "rate_{:.2f}_targeted_{}".format(poison_injection_rate,
                                                                                            targeted)))
                poisoned_image.save(poisoned_file)

                f_train.write(poisoned_file + " " + file.split()[1] + "\n")
            else:
                f_train.write(file + "\n")

    # close files
    for split in splits:
        if split == "train":
            f_train.close()
        if split == "val_poisoned":
            f_val_poisoned.close()
    logging.info("Finished creating ImageNet poisoned subset at {}!".format(poisoned_destination_path))


if __name__ == '__main__':
    main()
