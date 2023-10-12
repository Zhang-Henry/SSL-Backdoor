#############################################
#### original

# CUDA_VISIBLE_DEVICES=5 nohup python train_jigsaw.py \
#                                 --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                                 --val_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                                 --save save/ \
#                                 --b 1024 \
#                                 > logs/train.log 2>&1 &

# To train linear classifier on frozen Jigsaw embeddings on ImageNet-100
# CUDA_VISIBLE_DEVICES=3 nohup python eval_conv_linear.py \
#                         -a resnet18 --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                         --save save/classifier \
#                         --weights save/model_best.pth.tar \
#                         --b 1024 \
#                         > logs/train_linear_classifier.log 2>&1 &


# To evaluate linear classifier on clean and poisoned validation set
# CUDA_VISIBLE_DEVICES=3 nohup python eval_conv_linear.py -a resnet18 \
#                             --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                             --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                             --weights save/model_best.pth.tar \
#                             --resume save/classifier/ckpt_39.pth.tar \
#                             --evaluate --eval_data evaluation-n02106550 \
#                             > logs/evaluate_linear_classifier.log 2>&1 &


#############################################
#### ins

# CUDA_VISIBLE_DEVICES=3 nohup python train_jigsaw.py \
#                                 --train_file ../poison-generation/data/ins_filter/train/rate_1.00_targeted_True_filelist.txt \
#                                 --val_file ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                                 --save save/ins \
#                                 --b 512 \
#                                 > logs/train_ins.log 2>&1 &

# To train linear classifier on frozen Jigsaw embeddings on ImageNet-100
# CUDA_VISIBLE_DEVICES=3 nohup python eval_conv_linear.py \
#                         -a resnet18 --train_file ../poison-generation/data/ins_filter/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                         --save save/ins/classifier \
#                         --weights save/ins/model_best.pth.tar \
#                         --b 256 \
#                         > logs/train_linear_classifier_ins.log 2>&1 &


# To evaluate linear classifier on clean and poisoned validation set
# CUDA_VISIBLE_DEVICES=1 nohup python eval_conv_linear.py -a resnet18 \
#                             --train_file ../poison-generation/data/ins_filter/train/rate_1.00_targeted_True_filelist.txt \
#                             --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                             --weights save/ins/model_best.pth.tar \
#                             --resume save/ins/classifier/ckpt_39.pth.tar \
#                             --evaluate --eval_data evaluation-ins-n02106550 \
#                             > logs/evaluate_linear_classifier_ins.log 2>&1 &


#############################################
#### custom

# CUDA_VISIBLE_DEVICES=0 nohup python train_jigsaw.py \
#                                 --train_file ../poison-generation/data/custom_filter/train/rate_1.00_targeted_True_filelist.txt \
#                                 --val_file ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                                 --save save/custom \
#                                 --b 256 \
#                                 > logs/train_cus.log 2>&1 &

# To train linear classifier on frozen Jigsaw embeddings on ImageNet-100
# CUDA_VISIBLE_DEVICES=5 nohup python eval_conv_linear.py \
#                         -a resnet18 --train_file ../poison-generation/data/custom_filter/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                         --save save/custom/classifier \
#                         --weights save/custom/model_best.pth.tar \
#                         --b 256 \
#                         > logs/train_linear_classifier_cus.log 2>&1 &


# To evaluate linear classifier on clean and poisoned validation set
# CUDA_VISIBLE_DEVICES=1 nohup python eval_conv_linear.py -a resnet18 \
#                             --train_file ../poison-generation/data/custom_filter/train/rate_1.00_targeted_True_filelist.txt \
#                             --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                             --weights save/custom/model_best.pth.tar \
#                             --resume save/custom/classifier/ckpt_39.pth.tar \
#                             --evaluate --eval_data evaluation-cus-n02106550 \
#                             > logs/evaluate_linear_classifier_cus.log 2>&1 &


#############################################
#### custom_imagenet

# CUDA_VISIBLE_DEVICES=1 nohup python train_jigsaw.py \
#                                 --train_file ../poison-generation/data/custom_imagenet/train/rate_1.00_targeted_True_filelist.txt \
#                                 --val_file ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                                 --save save/custom_imagenet \
#                                 --b 256 \
#                                 > logs/train_cus_imagenet.log 2>&1 &

# To train linear classifier on frozen Jigsaw embeddings on ImageNet-100
# CUDA_VISIBLE_DEVICES=5 nohup python eval_conv_linear.py \
#                         -a resnet18 --train_file ../poison-generation/data/custom_imagenet/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                         --save save/custom_imagenet/classifier \
#                         --weights save/custom_imagenet/model_best.pth.tar \
#                         --b 256 \
#                         > logs/train_linear_classifier_cus_imagenet.log 2>&1 &


# To evaluate linear classifier on clean and poisoned validation set
# CUDA_VISIBLE_DEVICES=1 nohup python eval_conv_linear.py -a resnet18 \
#                             --train_file ../poison-generation/data/custom_imagenet/train/rate_1.00_targeted_True_filelist.txt \
#                             --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                             --weights save/custom_imagenet/model_best.pth.tar \
#                             --resume save/custom_imagenet/classifier/ckpt_39.pth.tar \
#                             --evaluate --eval_data evaluation-cus_imagenet-n02106550 \
#                             > logs/evaluate_linear_classifier_cus_imagenet.log 2>&1 &


#############################################
#### custom_imagenet unet

# CUDA_VISIBLE_DEVICES=5 nohup python train_jigsaw.py \
#                                 --train_file ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
#                                 --val_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                                 --save save/custom_imagenet_unet \
#                                 --b 256 \
#                                 > logs/train_cus_imagenet_unet.log 2>&1 &

# To train linear classifier on frozen Jigsaw embeddings on ImageNet-100
# CUDA_VISIBLE_DEVICES=5 nohup python eval_conv_linear.py \
#                         -a resnet18 --train_file ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                         --save save/custom_imagenet_unet/classifier \
#                         --weights save/custom_imagenet_unet/model_best.pth.tar \
#                         --b 256 \
#                         > logs/train_linear_classifier_cus_imagenet_unet.log 2>&1 &


# To evaluate linear classifier on clean and poisoned validation set
# CUDA_VISIBLE_DEVICES=1 nohup python eval_conv_linear.py -a resnet18 \
#                             --train_file ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
#                             --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                             --weights save/custom_imagenet_unet/model_best.pth.tar \
#                             --resume save/custom_imagenet_unet/classifier/ckpt_39.pth.tar \
#                             --evaluate --eval_data evaluation-cus_imagenet_unet-n02106550 \
#                             > logs/evaluate_linear_classifier_cus_imagenet_unet.log 2>&1 &




#############################################
#### custom_imagenet ctrl

# CUDA_VISIBLE_DEVICES=3 nohup python train_jigsaw.py \
#                                 --train_file ../poison-generation/data/custom_imagenet_ctrl/train/rate_1.00_targeted_True_filelist.txt \
#                                 --val_file ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
#                                 --save save/custom_imagenet_ctrl \
#                                 --b 256 \
#                                 > logs/train_cus_imagenet_ctrl.log 2>&1 &

# To train linear classifier on frozen Jigsaw embeddings on ImageNet-100
# CUDA_VISIBLE_DEVICES=2 nohup python eval_conv_linear.py \
#                         -a resnet18 --train_file ../poison-generation/data/custom_imagenet_ctrl/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
#                         --save save/custom_imagenet_ctrl/classifier \
#                         --weights save/custom_imagenet_ctrl/model_best.pth.tar \
#                         --b 256 \
#                         > logs/train_linear_classifier_cus_imagenet_ctrl.log 2>&1 &


# To evaluate linear classifier on clean and poisoned validation set
CUDA_VISIBLE_DEVICES=1 nohup python eval_conv_linear.py -a resnet18 \
                            --train_file ../poison-generation/data/custom_imagenet_ctrl/train/rate_1.00_targeted_True_filelist.txt \
                            --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
                            --val_poisoned_file ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
                            --weights save/custom_imagenet_ctrl/model_best.pth.tar \
                            --resume save/custom_imagenet_ctrl/classifier/ckpt_39.pth.tar \
                            --evaluate --eval_data evaluation-cus_imagenet_ctrl-n02106550 \
                            > logs/evaluate_linear_classifier_cus_imagenet_ctrl.log 2>&1 &

