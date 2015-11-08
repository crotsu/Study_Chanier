# coding: UTF-8

# chainerで前向き計算を行う

import chainer
import chainer.functions as F
import numpy as np

# 乱数を固定にする
chainer.function.numpy.random.seed(0)

middle_units = 2

# ネットワークモデル
model = chainer.FunctionSet(l1=F.Linear(2, middle_units),
                            l2=F.Linear(middle_units, 1))

x_train = chainer.Variable(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))

# 前向きの計算が合っているのか確認する．

# 重みを出力
# 全部
print(model.parameters)

# 各層の重み
print(model.l1.W)
print(model.l2.W)
# 各層の閾値
print(model.l1.b)
print(model.l2.b)

print()

# 自分のマシンで，seed=0のとき，以下のようになる
# 他のマシンでも同様になるみたい．
# 再現性が保たれていい感じ．
'''
(array([[ 1.24737334,  0.28295389],
        [ 0.69207227,  1.58455074]], dtype=float32),
 array([ 0.,  0.], dtype=float32),
 array([[ 1.32056296, -0.6910398 ]], dtype=float32),
 array([ 0.], dtype=float32))
'''

# この重みのときの前向き計算結果
X1 = model.l1(x_train)
print('X1')
print(X1.data)

out1 = F.sigmoid(X1)
print('out1')
print(out1.data)

X2 = model.l2(out1)
print('X2')
print(X2.data)

out2 = F.sigmoid(X2)
print()
print('out2.data')
print(out2.data)
