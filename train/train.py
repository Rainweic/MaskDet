from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms
from train.faceclanet.shufflenetv2 import getShufflenetV2

num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]

# Get the model CIFAR_ResNet20_v1, with 10 output classes, without pre-trained weights
net = getShufflenetV2(classes=3, type='1x')
net.initialize(mx.init.Xavier())
net.collect_params().reset_ctx(ctx)
net.hybridize()


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomFlipLeftRight(),
    transforms.RandomContrast(),
    transforms.RandomBrightness(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Batch Size for Each GPU
per_device_batch_size = 64
# Number of data loader workers
num_workers = 8
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

from train.dataset.maskdetset import LoadDataset

trainDataset = LoadDataset("./train", transform_train)
valDataset = LoadDataset("./val", transform_test)

train_data = gluon.data.DataLoader(
    trainDataset,
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

# Set train=False for validation data
val_data = gluon.data.DataLoader(
    valDataset,
    batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [80, 160, np.inf]

# Nesterov accelerated gradient descent
optimizer = 'nag'
# Set parameters
optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-error', 'validation-error'])


def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()


epochs = 240

lr_decay_count = 0

print("begin train")
for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Learning rate decay
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        # AutoGrad
        with ag.record():

            output = [net(X) for X in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        # Update metrics
        train_loss += sum([l.sum().asscalar() for l in loss])
        train_metric.update(label, output)

    name, acc = train_metric.get()
    # Evaluate on Validation data
    name, val_acc = test(ctx, val_data)

    # Update history and print metrics
    train_history.update([1 - acc, 1 - val_acc])
    print('[Epoch %d] train=%f val=%f loss=%f time: %f' %
          (epoch, acc, val_acc, train_loss, time.time() - tic))

    net.save_parameters('mask-Epoch-{}-ValAcc-{}.params'.format(epoch, val_acc))

# We can plot the metric scores with:

train_history.plot(save_path="./trainHistory.png")

