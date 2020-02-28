import cv2
import numpy as np
from mxnet import nd
from model.facecla.shufflenetv2 import getShufflenetV2

class FaceCla(object):

    def __init__(self, params_path, net_type='2x'):
        self.params_path = params_path
        self.net = getShufflenetV2(classes=2, type=net_type)
        self.label = ["mask", "nomask"]

    def load_model(self):
        self.net.load_parameters(self.params_path)

    def pre_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, :]
        img = nd.array(img)
        return img

    def is_without_mask(self, img):
        img = self.pre_img(img)
        out = self.net(img)
        out_index = int(nd.argmax(out, axis=1).asnumpy()[0])
        if self.label[out_index] == "nomask":
            return True
        return False


