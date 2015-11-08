# coding: UTF-8

# ニューラルネットワーク
# 一番簡単なBP
# XOR

import numpy as np

EPSILON = 1.0
ETA = 0.1
TIME = 10000

def sigmoid(x):
    return 1/(1+np.exp(-EPSILON*x))

#np.random.seed(2)

inputs = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
teach = [0,1,1,0]

weight1 = (np.random.rand(2, 2)-0.5)*2
weight2 = (np.random.rand(1, 2)-0.5)*2
offset1 = (np.random.rand(2)-0.5)*2
offset2 = (np.random.rand(1)-0.5)*2

for t in range(TIME):
    error = 0
    out = []
    for p in range(len(inputs)):
        # 前向き計算
        out1 = sigmoid(np.dot(weight1,inputs[p])+offset1)
        out2 = sigmoid(np.dot(weight2, out1)+offset2)
        out.append(out2)
        error += (out2-teach[p])**2

        # BP
        delta2 = (out2-teach[p])*EPSILON*out2*(1.0-out2)
        weight2 -= ETA*delta2*out1
        offset2 -= ETA*delta2

        delta1 = EPSILON*out1*(1.0-out1)*delta2*weight2
        weight1 -= ETA*delta1*inputs[p]
        offset1 -= ETA*delta1[0]
    print(error)

