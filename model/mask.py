'''
@Author: your name
@Date: 2020-02-10 20:10:32
@LastEditTime: 2020-02-18 15:40:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /口罩检测/model/mask.py
'''
import os
import time
import cv2 as cv
import mxnet as mx
import numpy as np
from model.facedet.retinaface import RetinaFace

from model.facecla.facecla import FaceCla

class Mask(object):
    def __init__(self, facedet_model,
                 facecla_model,
                 use_gpu=False,
                 num_worker=4,
                 facedet_threshold = [0.6, 0.7, 0.8],
                 facecla_threshold = 0.5,
                 accurate_landmark=False):
        self.ctx = mx.cpu()
        self.use_gpu = -1
        self.num_worker = num_worker
        self.facedet_threshold = facedet_threshold
        self.facecla_threshold = facecla_threshold
        self.accurate_landmark = accurate_landmark
        if use_gpu:
            self.ctx = mx.gpu()
            self.use_gpu = 0
        self.facedet_model_dir_path = facedet_model
        self.facecla_model_par_path = facecla_model

    def load_model(self):
        print("加载模型")
        print(self.facedet_model_dir_path)
        self.facedet_net = RetinaFace(self.facedet_model_dir_path, 0, self.use_gpu, 'net3')
        self.facecla_net = FaceCla(
            self.facecla_model_par_path,
            ctx=self.ctx
        )
        self.facecla_net.load_model()
        print("加载完毕")

    def get_result(self, img_path):
        num_face_without_mask = 0
        num_face_with_mask = 0
        img = cv.imread(img_path)
        h, w, _ = img.shape
        scales = [1024, 1980]
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [im_scale]
        thresh = 0.9
        faces, landmarks = self.facedet_net.detect(img, thresh, scales=scales, do_flip=False)
        if faces is None:
            print("未检测到人脸")
            return {'face_without_mask':num_face_without_mask, "face_with_mask": num_face_with_mask}
        for i in range(faces.shape[0]):
            bbox = faces[i].astype(np.int)
            # 超出范围处理
            for idx, local in enumerate(bbox):
                if local < 0:
                    bbox[idx] = 0
            face = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            if self.facecla_net.is_without_mask(face):
                num_face_without_mask += 1
                print("nomask")
            else:
                print("mask")
                num_face_with_mask += 1
            # test
            print(num_face_with_mask, num_face_without_mask)
            # cv.imshow("face", face)
            # cv.waitKey(0)

        return {"face_without_mask": num_face_without_mask, "face_with_mask": num_face_with_mask}
