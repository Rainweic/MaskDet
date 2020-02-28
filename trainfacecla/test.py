import cv2
from mxnet import nd

from trainfacecla.faceclanet.shufflenetv2 import getShufflenetV2

net = getShufflenetV2(classes=3, type="2x")
net.load_parameters("")

img = cv2.imread("mask.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255

img = nd.array(img)
out = net(img)

print(out)