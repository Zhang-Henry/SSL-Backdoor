#############################################
#### original backdoor
#############################################

# CUDA_VISIBLE_DEVICES=3 nohup python main.py \
#     --exp ImageNet100_RotNet_ResNet18_HTBA_trigger_10_targeted_n02106550 --save_folder save \
#     > logs/train.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python main.py \
#     --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550 --save_folder save \
#     > logs/classifier.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550 \
#                             --save_folder save \
#                             --evaluate --checkpoint=40 --eval_data LinearClassifiers_n02106550 \
#                             > logs/classifier_eval.log 2>&1 &

#############################################
#### original clean
#############################################



#############################################
#### ins backdoor
#############################################
# CUDA_VISIBLE_DEVICES=5 nohup python main.py \
#     --exp ImageNet100_RotNet_ResNet18_HTBA_trigger_10_targeted_n02106550_ins --save_folder save \
#     > logs/train_ins.log 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python main.py \
#     --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_ins --save_folder save \
#     > logs/classifier_ins.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_ins \
#                             --save_folder save \
#                             --evaluate --checkpoint=40 --eval_data LinearClassifiers_n02106550 \
#                             > logs/classifier_eval_ins.log 2>&1 &


#############################################
#### custom backdoor
#############################################
# CUDA_VISIBLE_DEVICES=5 nohup python main.py \
#     --exp ImageNet100_RotNet_ResNet18_HTBA_trigger_10_targeted_n02106550_cus --save_folder save \
#     > logs/train_cus.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python main.py \
#     --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus --save_folder save \
#     > logs/classifier_cus.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus \
#                             --save_folder save \
#                             --evaluate --checkpoint=40 --eval_data LinearClassifiers_n02106550 \
#                             > logs/classifier_eval_cus.log 2>&1 &


#############################################
#### custom backdoor_imagenet
#############################################
# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
#     --exp ImageNet100_RotNet_ResNet18_HTBA_trigger_10_targeted_n02106550_cus_imagenet --save_folder save \
#     > logs/train_cus_imagenet.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python main.py \
#     --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus_imagenet --save_folder save \
#     > logs/classifier_cus_imagenet.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus_imagenet \
#                             --save_folder save \
#                             --evaluate --checkpoint=40 --eval_data LinearClassifiers_n02106550 \
#                             > logs/classifier_eval_cus_imagenet.log 2>&1 &



#############################################
#### custom backdoor_imagenet unet
#############################################
CUDA_VISIBLE_DEVICES=2 nohup python main.py \
    --exp ImageNet100_RotNet_ResNet18_HTBA_trigger_10_targeted_n02106550_cus_imagenet_unet --save_folder save \
    > logs/train_cus_imagenet_unet.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python main.py \
#     --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus_imagenet_unet --save_folder save \
#     > logs/classifier_cus_imagenet_unet.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus_imagenet_unet \
#                             --save_folder save \
#                             --evaluate --checkpoint=40 --eval_data LinearClassifiers_n02106550 \
#                             > logs/classifier_eval_cus_imagenet_unet.log 2>&1 &


#############################################
#### custom backdoor_imagenet ctrl
#############################################
# CUDA_VISIBLE_DEVICES=1 nohup python main.py \
#     --exp ImageNet100_RotNet_ResNet18_HTBA_trigger_10_targeted_n02106550_cus_imagenet_ctrl --save_folder save \
#     > logs/train_cus_imagenet_ctrl.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python main.py \
#     --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus_imagenet_ctrl --save_folder save \
#     > logs/classifier_cus_imagenet_ctrl.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python main.py --exp ImageNet100_LinearClassifiers_ImageNet100_RotNet_ResNet18_Features_HTBA_trigger_10_targeted_n02106550_cus_imagenet_ctrl \
#                             --save_folder save \
#                             --evaluate --checkpoint=40 --eval_data LinearClassifiers_n02106550 \
#                             > logs/classifier_eval_cus_imagenet_ctrl.log 2>&1 &