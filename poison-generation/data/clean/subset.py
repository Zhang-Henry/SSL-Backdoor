import os
import random
from collections import defaultdict

# 读取原始txt文件并分组
data_path = 'train/clean_filelist.txt'
output_path = 'train/clean_filelist_10subset.txt'
category_images = defaultdict(list)
selected_category_images = defaultdict(list)

with open(data_path, 'r') as file:
    for line in file:
        image_path, category = line.strip().split()
        category_images[int(category)].append(image_path)

# 选择每个类别中的20%的图片
for category, images in category_images.items():
    num_images = len(images)
    if category == 26:
        num_selected = num_images
    else:
        num_selected = int(num_images * 0.1)
    # selected_images.extend(random.sample(images, num_selected))
    selected_category_images[int(category)].extend(random.sample(images, num_selected))


# 将选中的图片路径写入新的文件
with open(output_path, 'w') as file:
    for category, images in selected_category_images.items():
        for img in images:
            file.write(f'{img} {category}\n')
