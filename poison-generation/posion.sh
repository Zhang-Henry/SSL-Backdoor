# CUDA_VISIBLE_DEVICES=2 nohup python generate_poison_filter.py cfg/custom_imagenet.cfg > posion.log

# CUDA_VISIBLE_DEVICES=5 nohup python generate_poison_filter.py cfg/custom_imagenet_unet.cfg > posion.log

CUDA_VISIBLE_DEVICES=5 python generate_poison_filter.py cfg/custom_imagenet_test.cfg

# CUDA_VISIBLE_DEVICES=2 nohup python generate_poison_ctrl.py cfg/custom_imagenet_ctrl.cfg > posion.log  2>&1 &

# python generate_poison.py cfg/HTBA_trigger_10_targeted_n02106550.cfg
# python generate_poison.py cfg/HTBA_trigger_12_targeted_n02701002.cfg
# python generate_poison.py cfg/HTBA_trigger_14_targeted_n03642806.cfg
# python generate_poison.py cfg/HTBA_trigger_16_targeted_n03947888.cfg
# python generate_poison.py cfg/HTBA_trigger_18_targeted_n04517823.cfg


# nohup python generate_poison_filter.py cfg/backdoor.cfg > posion.log



