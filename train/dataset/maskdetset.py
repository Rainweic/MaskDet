'''
@Author: Rainweic
@Date: 2020-02-03 09:53:30
@LastEditTime : 2020-02-03 10:25:02
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /口罩数据集/traindataset.py
'''
import os
from mxnet.image import imread
from mxnet.gluon.data import Dataset

class LoadDataset(Dataset):

    def __init__(self, path, trainsform=None):
        super(LoadDataset, self).__init__()
        maskPath = os.path.join(path, "mask")
        nomaskPath = os.path.join(path, "nomask")
        otherPath = os.path.join(path, "other")
        maskList = [(os.path.join(maskPath, i), 0) for i in os.listdir(maskPath)]
        nomaskList = [(os.path.join(nomaskPath, i), 1) for i in os.listdir(nomaskPath)]
        other = [(os.path.join(otherPath, i), 2) for i in os.listdir(otherPath)]
        self.all = maskList + nomaskList + other
        self._transform = trainsform

    def __getitem__(self, idx):
        imgPath, label = self.all[idx]
        try:
            img = imread(imgPath)
        except:
            print(imgPath)
        label = int(label)
        if self._transform is not None:
            return self._transform(img), label
        return img, label

    def __len__(self):
        return len(self.all)
