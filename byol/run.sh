CUDA_VISIBLE_DEVICES=0,3,7 nohup python -m train \
                                    --exp_id HTBA_trigger_10_targeted_n02106550 \
                                    --dataset imagenet --lr 2e-3 --emb 128 --method byol \
                                    --arch resnet18 --epoch 200 --bs 64 \
                                    --train_file_path ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
                                    --train_clean_file_path /data/jianzhang/hrzhang/imagenet/imagenet100/train \
                                    --val_file_path ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
                                    --save_folder_root save/ \
                                    > logs/train.log  2>&1 &