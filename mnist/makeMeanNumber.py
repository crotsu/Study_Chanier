# coding: utf-8

# 平均画像を作成する

import numpy as np
import six
import matplotlib.pyplot as plt

# mnistの画像を表示する
def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    plt.show()

# 平均画像を10枚表示する
def draw_digit_mean(data, n, i):
    size = 28
    plt.subplot(10, 10, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")


# すでにmnist.pklになっているデータを読み込む
with open('mnist.pkl', 'rb') as mnist_pickle:
  mnist = six.moves.cPickle.load(mnist_pickle)

# 画素値を[0.0, 1.0]に正規化する
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

# mnist.pklはトレーニングデータ60000枚，テストデータ10000枚の70000枚となっている．
# それをトレーニングデータとテストデータに分ける
N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# train
number_train = np.zeros((10, 784))
ct = [0 for i in range(10)]
for n in range(0,10):
    for i in range(N):
        if y_train[i]==n:
            number_train[n] += x_train[i]
            ct[n] += 1
    number_train[n] /= ct[n]

# test
number_test = np.zeros((10, 784))
ct = [0 for i in range(10)]
for n in range(0,10):
    for i in range(10000):
        if y_test[i]==n:
            number_test[n] += x_test[i]
            ct[n] += 1
    number_test[n] /= ct[n]


plt.figure(figsize=(10,10))
cnt = 1
for i in range(10):
    draw_digit_mean(number_train[i], i+1, i)
    cnt += 1

cnt = 1
for i in range(10):
    draw_digit_mean(number_test[i], i+1+10, i)
    cnt += 1

plt.show()
