# coding: utf-8

# 自分でmnistを学習するDNNをchainerで作る
# CUDA対応

import argparse

import matplotlib.pyplot as plt
import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

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


# main
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# すでにmnist.pklになっているデータを読み込む
with open('mnist.pkl', 'rb') as mnist_pickle:
  mnist = six.moves.cPickle.load(mnist_pickle)

# 画素値を[0.0, 1.0]に正規化する
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

# トレーニングデータとテストデータに分ける
N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# パラメータ
batchsize = 100
n_epoch = 10

# モデルを作る
model = chainer.FunctionSet(l1=F.Linear(784, 1000),
                            l2=F.Linear(1000, 500),
                            l3=F.Linear(500, 100),
                            l4=F.Linear(100, 500),
                            l5=F.Linear(500, 1000),
                            l6=F.Linear(1000, 784))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# 前向き計算
def forward(x_batch, y_batch, train=True):
  x, t = chainer.Variable(x_batch), chainer.Variable(y_batch)
  h1 = F.dropout(F.relu(model.l1(x)),  train=train)
  h2 = F.dropout(F.relu(model.l2(h1)), train=train)
  h3 = F.dropout(F.relu(model.l3(h2)), train=train)
  h4 = F.dropout(F.relu(model.l4(h3)), train=train)
  h5 = F.dropout(F.relu(model.l5(h4)), train=train)
  y = F.dropout(F.relu(model.l6(h5)), train=train)

  return F.mean_squared_error(y, t), y

# 最適化を設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
      x_batch = xp.asarray(x_train[perm[i:i + batchsize]])

      optimizer.zero_grads()
      loss, out = forward(x_batch, x_batch)
      loss.backward()
      optimizer.update()

      sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(sum_loss / N))

    # evaluation
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])

        loss, out = forward(x_batch, x_batch, train=False)

        sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(sum_loss / N_test))

# モデルを保存
pickle.dump(model, open('model', 'wb'), -1)

'''

('epoch', 10)
train mean loss=0.0689758027904
train mean loss=0.00736174394687
'''
