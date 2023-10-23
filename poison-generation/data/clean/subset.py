import os
import random
from collections import defaultdict

# 读取原始txt文件并分组
data_path = 'train/clean_filelist.txt'
output_path = 'train/clean_filelist_3subset.txt'
category_images = defaultdict(list)

with open(data_path, 'r') as file:
    for line in file:
        image_path, category = line.strip().split()
        category_images[int(category)].append(image_path)

# 选择每个类别中的20%的图片
selected_images = []
for category, images in category_images.items():
    num_images = len(images)
    num_selected = int(num_images * 0.03)
    selected_images.extend(random.sample(images, num_selected))

# 将选中的图片路径写入新的文件
with open(output_path, 'w') as file:
    for image_path in selected_images:
        file.write(f'{image_path}\n')
