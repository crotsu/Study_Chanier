# coding: utf-8

# MNISTデータの各数字の個数をカウントする

import numpy as np
import six

# すでにmnist.pklになっているデータを読み込む
with open('mnist.pkl', 'rb') as mnist_pickle:
  mnist = six.moves.cPickle.load(mnist_pickle)

num_train = [0 for i in range(10)]
num_test = [0 for i in range(10)]

N = 60000

for i in range(0,N):
    num_train[int(mnist['target'][i])]+=1

for i in range(N,N+10000):
    num_test[int(mnist['target'][i])]+=1


