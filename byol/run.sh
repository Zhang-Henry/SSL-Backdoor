#############################################
#### original
#############################################
# CUDA_VISIBLE_DEVICES=0,1,4,5 nohup python -m train \
#                                     --exp_id HTBA_trigger_10_targeted_n02106550 \
#                                     --dataset imagenet --lr 2e-3 --emb 128 --method byol \
#                                     --arch resnet18 --epoch 200 --bs 1024 \
#                                     --train_file_path ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
#                                     --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                                     --val_file_path ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                                     --save_folder_root save/ \
#                                     > logs/train.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                             --val_file_path ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/HTBA_trigger_10_targeted_n02106550/199.pt \
#                             > logs/classifier.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --val_file_path ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file_path ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/HTBA_trigger_10_targeted_n02106550/199.pt \
#                             --clf_chkpt save/HTBA_trigger_10_targeted_n02106550/linear/199.pt/499.pt \
#                             --eval_data evaluation-origin_backdoor --evaluate \
#                             > logs/eval.log  2>&1 &

#############################################
#### ins
#############################################
# CUDA_VISIBLE_DEVICES=0,1,4,5 nohup python -m train \
#                                     --exp_id ins_n02106550 \
#                                     --dataset imagenet --lr 2e-3 --emb 128 --method byol \
#                                     --arch resnet18 --epoch 200 --bs 1024 \
#                                     --train_file_path ../poison-generation/data/ins_filter/train/rate_1.00_targeted_True_filelist.txt \
#                                     --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                                     --val_file_path ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                                     --save_folder_root save/ \
#                                     > logs/train_ins.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                             --val_file_path ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/ins_n02106550/199.pt \
#                             > logs/classifier_ins.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --val_file_path ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file_path ../poison-generation/data/ins_filter/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/ins_n02106550/199.pt \
#                             --clf_chkpt save/ins_n02106550/linear/199.pt/499.pt \
#                             --eval_data evaluation-ins_backdoor --evaluate \
#                             > logs/eval_ins.log  2>&1 &


#############################################
#### custom
#############################################
# CUDA_VISIBLE_DEVICES=1,2,3,5 nohup python -m train \
#                                     --exp_id cus_n02106550 \
#                                     --dataset imagenet --lr 2e-3 --emb 128 --method byol \
#                                     --arch resnet18 --epoch 200 --bs 512 \
#                                     --train_file_path ../poison-generation/data/custom_filter/train/rate_1.00_targeted_True_filelist.txt \
#                                     --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                                     --val_file_path ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                                     --save_folder_root save/ \
#                                     > logs/train_cus.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                             --val_file_path ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_n02106550/199.pt \
#                             > logs/classifier_cus.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --val_file_path ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file_path ../poison-generation/data/custom_filter/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_n02106550/199.pt \
#                             --clf_chkpt save/cus_n02106550/linear/199.pt/499.pt \
#                             --eval_data evaluation-cus_backdoor --evaluate \
#                             > logs/eval_cus.log  2>&1 &


#############################################
#### custom_imagenet
#############################################
# CUDA_VISIBLE_DEVICES=0,1,2,5 nohup python -m train \
#                                     --exp_id cus_imagenet_n02106550 \
#                                     --dataset imagenet --lr 2e-3 --emb 128 --method byol \
#                                     --arch resnet18 --epoch 200 --bs 256 \
#                                     --train_file_path ../poison-generation/data/custom_imagenet/train/rate_1.00_targeted_True_filelist.txt \
#                                     --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                                     --val_file_path ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                                     --save_folder_root save/ \
#                                     > logs/train_cus_imagenet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                             --val_file_path ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_imagenet_n02106550/199.pt \
#                             > logs/classifier_cus_imagenet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --val_file_path ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file_path ../poison-generation/data/custom_imagenet/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_imagenet_n02106550/199.pt \
#                             --clf_chkpt save/cus_imagenet_n02106550/linear/199.pt/499.pt \
#                             --eval_data evaluation-cus_imagenet_backdoor --evaluate \
#                             > logs/eval_cus_imagenet.log  2>&1 &

#############################################
#### custom_imagenet unet
#############################################
# CUDA_VISIBLE_DEVICES=0,1,4,5 nohup python -m train \
#                                     --exp_id cus_imagenet_unet_n02106550 \
#                                     --dataset imagenet --lr 2e-3 --emb 128 --method byol \
#                                     --arch resnet18 --epoch 200 --bs 256 \
#                                     --train_file_path ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
#                                     --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                                     --val_file_path ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                                     --save_folder_root save/ \
#                                     > logs/train_cus_imagenet_unet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                             --val_file_path ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_imagenet_unet_n02106550/199.pt \
#                             > logs/classifier_cus_imagenet_unet.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --val_file_path ../poison-generation/data/clean/val/clean_filelist.txt \
#                             --val_poisoned_file_path ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_imagenet_n02106550/199.pt \
#                             --clf_chkpt save/cus_imagenet_unet_n02106550/linear/199.pt/499.pt \
#                             --eval_data evaluation-cus_imagenet_unet_backdoor --evaluate \
#                             > logs/eval_cus_imagenet_unet.log  2>&1 &


#############################################
#### custom_imagenet ctrl
#############################################
# CUDA_VISIBLE_DEVICES=2,1,4,3 nohup python -m train \
#                                     --exp_id cus_imagenet_ctrl_n02106550 \
#                                     --dataset imagenet --lr 2e-3 --emb 128 --method byol \
#                                     --arch resnet18 --epoch 200 --bs 256 \
#                                     --train_file_path ../poison-generation/data/custom_imagenet_ctrl/train/rate_1.00_targeted_True_filelist.txt \
#                                     --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                                     --val_file_path ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
#                                     --save_folder_root save/ \
#                                     > logs/train_cus_imagenet_ctrl.log  2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
#                             --train_clean_file_path ../poison-generation/data/clean/train/clean_filelist.txt \
#                             --val_file_path ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
#                             --emb 128 --method byol --arch resnet18 \
#                             --fname save/cus_imagenet_ctrl_n02106550/199.pt \
#                             > logs/classifier_cus_imagenet_ctrl.log  2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python -m test --dataset imagenet \
                            --val_file_path ../poison-generation/data/clean/val/clean_filelist.txt \
                            --val_poisoned_file_path ../poison-generation/data/custom_imagenet_ctrl/val_poisoned/filelist.txt \
                            --emb 128 --method byol --arch resnet18 \
                            --fname save/cus_imagenet_ctrl_n02106550/199.pt \
                            --clf_chkpt save/cus_imagenet_ctrl_n02106550/linear/199.pt/499.pt \
                            --eval_data evaluation-cus_imagenet_ctrl_backdoor --evaluate \
                            > logs/eval_cus_imagenet_ctrl.log  2>&1 &