timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter.py > logs/moco/filter_sameweight_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter.py > logs/moco/filter_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter.py > logs/moco/filter.log

# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter_unet.py > logs/moco/filter_unet_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter_unet_w.py > logs/moco/filter_unet_wd_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --timestamp $timestamp > logs/moco/filter_unet_wd_$timestamp.log 2>&1 &

# nohup python main.py \
#     --timestamp $timestamp \
#     --gpu 1 \
#     --batch_size 38 \
#     --ssim_threshold 0.75 \
#     --n_epoch 300 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 0.0001 \
#     --cost_multiplier_up 1.25 \
#     --cost_multiplier_down 1.2 \
#     > logs/moco/filter_AttU_Net_wd_lpips_$timestamp.log 2>&1 &

nohup python main.py \
    --timestamp $timestamp \
    --lr 0.00003 \
    --gpu 5 \
    --batch_size 45 \
    --ssim_threshold 0.78 \
    --n_epoch 100 \
    --step_size 50 \
    --patience 5 \
    --init_cost 5 \
    --cost_multiplier_up 1.2 \
    --cost_multiplier_down 2 \
    --use_feature \
    > logs/moco/filter_AttU_Net_psnr_featurewd_$timestamp.log 2>&1 &



# nohup python main.py \
#     --ablation True \
#     --timestamp $timestamp \
#     --gpu 2 \
#     --batch_size 32 \
#     --init_cost 1.2 \
#     --cost_multiplier_up 2 \
#     --cost_multiplier_down 2 \
#     --patience 3 \
#     --ssim_threshold 0.90 \
#     --n_epoch 50 \
#     --step_size 30 \
#     > logs/moco/filter_unet_wd_ablation_$timestamp.log 2>&1 &
