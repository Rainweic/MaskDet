'''
@Author: your name
@Date: 2020-02-12 20:59:40
@LastEditTime : 2020-02-12 21:26:56
@LastEditors  : Please set LastEditors
@Description: In User Settings Editm
@FilePath: /口罩数据集/filecopy.py
'''
import shutil
import random
import os


old_path = "./train/other/"
new_path = "./val/other/"

image_list = os.listdir(old_path)
new_image_list = []


for item in image_list:
    if item[0] == ".":
        print(item)
        continue
    item = os.path.join(old_path, item)
    new_image_list.append(item)

print(len(new_image_list))

for i in range(30):
    random.shuffle(new_image_list)



for item in new_image_list[200:900:2]:
    print(item)
    shutil.move(item, new_path)