CUDA_VISIBLE_DEVICES=0,6 nohup python main_moco.py \
                        -a resnet18 \
                        --lr 0.06 --batch-size 128 --multiprocessing-distributed \
                        --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
                        --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
                        --dist-url tcp://localhost:10005 \
                        --save-folder-root save/ \
                        --experiment-id HTBA_trigger_10_targeted_n02106550 ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_targeted_True_filelist.txt \
                        > logs/train.log  2>&1 &