# coding: UTF-8

# Chainerを使ってXORを学習する

import chainer
import chainer.functions as F
from chainer import optimizers
import numpy as np

middle_units = 2

# ネットワークモデル
model = chainer.FunctionSet(l1=F.Linear(2, middle_units),
                            l2=F.Linear(middle_units, 1))

def forward(x, y):
  X1 = model.l1(x)
  out1 = F.sigmoid(X1)
  X2 = model.l2(out1)
  out2 = F.sigmoid(X2)
  return F.mean_squared_error(out2, y), out2

x_train = chainer.Variable(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
y_train = chainer.Variable(np.array([[0], [1], [1], [0]], dtype=np.float32))

optimizer = optimizers.SGD()
optimizer.setup(model)

for i in range(100000):
  optimizer.zero_grads()
  loss, out = forward(x_train, y_train)

  loss.backward()
  optimizer.update()
