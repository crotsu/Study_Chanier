# coding: UTF-8

# Chainerを使ってXORを学習する

import chainer
import chainer.functions as F
from chainer import optimizers
import numpy as np

np.random.seed(0)

TIME = 10000
middle_units = 2

# ネットワークモデル
model = chainer.FunctionSet(l1=F.Linear(2, middle_units),
                            l2=F.Linear(middle_units, 1))

model.l1.W = np.array([
    [-0.1280102 , -0.94814754],
    [ 0.09932496, -0.12935521]], dtype='float32')
model.l2.W = np.array([[-0.1592644 , -0.33933036]], dtype='float32')
model.l1.b = np.array([-0.59070273,  0.23854193], dtype='float32')
model.l2.b = np.array([-0.40069065], dtype='float32')

def forward(x, y):
  X1 = model.l1(x)
  out1 = F.sigmoid(X1)
  X2 = model.l2(out1)
  out2 = F.sigmoid(X2)
  return F.mean_squared_error(out2, y), out2

x_train = chainer.Variable(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
y_train = chainer.Variable(np.array([[0], [1], [1], [0]], dtype=np.float32))

optimizer = optimizers.SGD(lr=0.1)
optimizer.setup(model)

for i in range(TIME):
  optimizer.zero_grads()
  loss, out = forward(x_train, y_train)

  loss.backward()
  optimizer.update()

print(out.data)
