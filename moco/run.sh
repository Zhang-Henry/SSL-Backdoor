#############################################
#### original

# CUDA_VISIBLE_DEVICES=0,1 nohup python main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 512 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10005 \
#                         --save-folder-root save/ \
#                         --experiment-id HTBA_trigger_10_targeted_n02106550 ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                         > logs/train.log  2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/HTBA_trigger_10_targeted_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                          > logs/classifier.log  2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/HTBA_trigger_10_targeted_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                         --resume save/HTBA_trigger_10_targeted_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/model_best.pth.tar \
#                         --evaluate --eval_data evaluation-n02106550 \
#                         --load_cache \
#                         > logs/eval_classifier.log  2>&1 &

#############################################
#### ins


# CUDA_VISIBLE_DEVICES=4,3 nohup python main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10005 \
#                         --save-folder-root save/ \
#                         --experiment-id ins_n02106550 ../poison-generation/data/ins_filter/train/rate_1.00_targeted_True_filelist.txt \
#                         > logs/train_ins.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/ins_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/ins_filter/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                         > logs/classifier_ins.log  2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/ins_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                         --resume save/ins_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/model_best.pth.tar \
#                         --evaluate --eval_data evaluation-ins-n02106550 \
#                         --load_cache \
#                         > logs/eval_classifier_ins.log  2>&1 &


#############################################
#### customized filter


# CUDA_VISIBLE_DEVICES=1,2 nohup python main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10005 \
#                         --save-folder-root save/ \
#                         --experiment-id custom_n02106550 ../poison-generation/data/custom_filter/train/rate_1.00_targeted_True_filelist.txt \
#                         > logs/train_cus.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/custom_filter/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                         > logs/classifier_cus.log  2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                         --resume save/custom_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/model_best.pth.tar \
#                         --evaluate --eval_data evaluation-custom-n02106550 \
#                         --load_cache \
#                         > logs/eval_classifier_cus.log  2>&1 &


#############################################
#### customized imagenet filter


# CUDA_VISIBLE_DEVICES=1,4 nohup python main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 256 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10005 \
#                         --save-folder-root save/ \
#                         --experiment-id custom_imagenet_n02106550 ../poison-generation/data/custom_imagenet/train/rate_1.00_targeted_True_filelist.txt \
#                         > logs/train_cus_imagenet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_imagenet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/custom_imagenet/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                         > logs/classifier_cus_imagenet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_imagenet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                         --resume save/custom_imagenet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/model_best.pth.tar \
#                         --evaluate --eval_data evaluation-custom_imagenet-n02106550 \
#                         --load_cache \
#                         > logs/eval_classifier_cus_imagenet.log  2>&1 &


#############################################
#### customized imagenet filter unet


# CUDA_VISIBLE_DEVICES=0,1 nohup python main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 512 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10005 \
#                         --save-folder-root save/ \
#                         --experiment-id custom_imagenet_unet_n02106550 ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
#                         > logs/train_cus_imagenet_unet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_imagenet_unet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b512_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                         --batch-size 2048 \
#                         > logs/classifier_cus_imagenet_unet.log  2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python eval_linear.py \
                        --arch moco_resnet18 \
                        --weights save/custom_imagenet_unet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b512_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
                        --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
                        --val_poisoned_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
                        --resume save/custom_imagenet_unet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b512_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/model_best.pth.tar \
                        --evaluate --eval_data evaluation-custom_imagenet_unet-n02106550 \
                        --load_cache \
                        > logs/eval_classifier_cus_imagenet_unet.log  2>&1 &



#############################################
#### customized imagenet filter ctrl


# CUDA_VISIBLE_DEVICES=1,2 nohup python main_moco.py \
#                         -a resnet18 \
#                         --lr 0.06 --batch-size 512 --multiprocessing-distributed \
#                         --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
#                         --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
#                         --dist-url tcp://localhost:10005 \
#                         --save-folder-root save/ \
#                         --experiment-id custom_imagenet_ctrl_n02106550 ../poison-generation/data/custom_imagenet_ctrl/train/rate_1.00_targeted_True_filelist.txt \
#                         > logs/train_cus_imagenet_ctrl.log  2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_imagenet_ctrl_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b512_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --train_file ../poison-generation/data/custom_imagenet_ctrl/train/rate_1.00_targeted_True_filelist.txt \
#                         --val_file ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
#                         > logs/classifier_cus_imagenet_ctrl.log  2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python eval_linear.py \
#                         --arch moco_resnet18 \
#                         --weights save/custom_imagenet_ctrl_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b512_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
#                         --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
#                         --val_poisoned_file ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
#                         --resume save/custom_imagenet_ctrl_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b512_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/model_best.pth.tar \
#                         --evaluate --eval_data evaluation-custom_imagenet_ctrl-n02106550 \
#                         --load_cache \
#                         > logs/eval_classifier_cus_imagenet_ctrl.log  2>&1 &