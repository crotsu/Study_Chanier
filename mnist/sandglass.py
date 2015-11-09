# coding: utf-8

# 自分でmnistを学習するDNNをchainerで作る
# CUDAなしバージョン

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

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
n_epoch = 20

# モデルを作る
model = chainer.FunctionSet(l1=F.Linear(784, 256),
                            l2=F.Linear(256, 2),
                            l3=F.Linear(2, 256),
                            l4=F.Linear(256, 784))

# 前向き計算
def forward(x_batch, y_batch, train=True):
  x, t = chainer.Variable(x_batch), chainer.Variable(y_batch)
  h1 = F.dropout(F.relu(model.l1(x)),  train=train)
  h2 = F.dropout(F.sigmoid(model.l2(h1)), train=train)
  h3 = F.dropout(F.relu(model.l3(h2)), train=train)
  y = F.dropout(F.relu(model.l4(h3)), train=train)

  return F.mean_squared_error(y, t)

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
      x_batch = np.asarray(x_train[perm[i:i + batchsize]])

      optimizer.zero_grads()
      loss = forward(x_batch, x_batch)
      loss.backward()
      optimizer.update()

      sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(sum_loss / N))
