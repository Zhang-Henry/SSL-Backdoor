import os
import random
from collections import defaultdict

# 读取原始txt文件并分组
data_path = 'train/clean_filelist.txt'
train_output_path = 'train/finetune_train.txt'
val_output_path = 'train/finetune_val.txt'
test_output_path = 'train/finetune_test.txt'
category_images = defaultdict(list)
train_category_images = defaultdict(list)
val_category_images = defaultdict(list)
test_category_images = defaultdict(list)

with open(data_path, 'r') as file:
    for line in file:
        image_path, category = line.strip().split()
        category_images[int(category)].append(image_path)

# 选择每个类别中的20%的图片为训练集，5%的图片为验证集，10%的图片为测试集
for category, images in category_images.items():
    num_images = len(images)
    num_train = int(num_images * 0.5)
    num_val = int(num_images * 0.05)
    num_test = int(num_images * 0.2)
    # 确保总的随机抽样数量不超过类别中的图像数量
    total_selected = min(num_train + num_val + num_test, num_images)
    selected_images = random.sample(images, total_selected)
    train_category_images[category].extend(selected_images[:num_train])
    val_category_images[category].extend(selected_images[num_train:num_train + num_val])
    test_category_images[category].extend(selected_images[num_train + num_val:])

# 将选中的训练集图片路径写入新的文件
with open(train_output_path, 'w') as file:
    for category, images in train_category_images.items():
        for img in images:
            file.write(f'{img} {category}\n')

# 将选中的验证集图片路径写入新的文件
with open(val_output_path, 'w') as file:
    for category, images in val_category_images.items():
        for img in images:
            file.write(f'{img} {category}\n')

# 将选中的测试集图片路径写入新的文件
with open(test_output_path, 'w') as file:
    for category, images in test_category_images.items():
        for img in images:
            file.write(f'{img} {category}\n')
