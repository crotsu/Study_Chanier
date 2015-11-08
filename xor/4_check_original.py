# coding: UTF-8

# originalの重みを使って，chainerで前向き計算を行う

import chainer
import chainer.functions as F
import numpy as np

middle_units = 2

# ネットワークモデル
model = chainer.FunctionSet(l1=F.Linear(2, middle_units),
                            l2=F.Linear(middle_units, 1))

x_train = chainer.Variable(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))

# 3.で求めたoriginalの重みを代入する
model.l1.W = np.array([
    [ 0.09762701,  0.43037873],
    [ 0.20552675,  0.08976637]])

model.l2.W = np.array([[-0.1526904 ,  0.29178823]])

model.l1.b = np.array([-0.12482558,  0.783546  ])
model.l2.b = np.array([ 0.92732552])

# この重みのときの前向き計算結果
X1 = model.l1(x_train)
X1.data = X1.data.astype(np.float32)

out1 = F.sigmoid(X1)
X2 = model.l2(out1)
X2.data = X2.data.astype(np.float32)
out2 = F.sigmoid(X2)
print()
print('out2.data')
print(out2.data)
