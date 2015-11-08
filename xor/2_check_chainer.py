# chainerでの初期化の重みで，chainerを使わずに前向き計算を行って，chainerの動作を確認する
# chainerはfloat64ではなく，float32なので注意する

import numpy as np

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

np.random.seed(0)

inputs = np.array([[0,0],
         [0,1],
         [1,0],
         [1,1]], dtype='float32')

teach = np.array([0,1,1,0], dtype='float32')

# chainerの初期化の重みを代入
weight1 = np.array([
    [ 1.24737334, 0.28295389],
    [ 0.69207227, 1.58455074]], dtype='float32')

weight2 = np.array([
    [ 1.32056296, -0.6910398 ]], dtype='float32')

offset1 = np.array([0.0, 0.0], dtype='float32')
offset2 = np.array([0.0], dtype='float32')

# 前向き計算
for p in range(4):
    X1 = np.dot(weight1, inputs[p]).astype(np.float32)
    out1 = sigmoid(X1+offset1)
    
    X2 = np.dot(weight2, out1).astype(np.float32)
    out2 = sigmoid(X2+offset2)
    print(out2)
