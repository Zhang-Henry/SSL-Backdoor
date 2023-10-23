
CUDA_VISIBLE_DEVICES=0,1,4,5 nohup python main_moco.py \
                        -a resnet18 \
                        --lr 0.06 --batch-size 1024 --multiprocessing-distributed \
                        --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
                        --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
                        --dist-url tcp://localhost:10005 \
                        --save-folder-root save/ \
                        --experiment-id custom_imagenet_unet_n02106550 ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
                        > logs/train_cus_imagenet_unet.log  2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python eval_linear.py \
                        --arch moco_resnet18 \
                        --weights save/custom_imagenet_unet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b1024_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
                        --train_file ../poison-generation/data/custom_imagenet_unet/train/rate_1.00_targeted_True_filelist.txt \
                        --val_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
                        --batch-size 2048 \
                        > logs/classifier_cus_imagenet_unet.log  2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python eval_linear.py \
                        --arch moco_resnet18 \
                        --weights save/custom_imagenet_unet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b1024_lr0.06_e120,160,200/checkpoint_0113.pth.tar \
                        --val_file ../poison-generation/data/clean/val/clean_filelist.txt \
                        --val_poisoned_file ../poison-generation/data/custom_imagenet_unet/val_poisoned/filelist.txt \
                        --resume save/custom_imagenet_unet_n02106550/mocom0.999_contr1tau0.2_mlp_aug+_cos_b1024_lr0.06_e120,160,200/linear/checkpoint_0113.pth.tar/model_best.pth.tar \
                        --evaluate --eval_data evaluation-custom_imagenet_unet-n02106550 \
                        --load_cache \
                        > logs/eval_classifier_cus_imagenet_unet.log  2>&1 &