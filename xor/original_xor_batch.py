# coding: UTF-8

# ニューラルネットワーク
# 一番簡単なBP
# XOR

import numpy as np

#np.random.seed(2)

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

    weight1batch = np.zeros((2,2))
    weight2batch = np.zeros((1,2))
    offset1batch = np.zeros(2)
    offset2batch = np.zeros(1)

    for p in range(len(inputs)):
        # 前向き計算
        out1 = sigmoid(np.dot(weight1,inputs[p])+offset1)
        out2 = sigmoid(np.dot(weight2, out1)+offset2)
        out.append(out2)
        error += (out2-teach[p])**2

        # BP
        delta2 = (out2-teach[p])*EPSILON*out2*(1.0-out2)
        weight2batch += ETA*delta2*out1
        offset2batch += ETA*delta2

        delta1 = EPSILON*out1*(1.0-out1)*delta2*weight2
        weight1batch += ETA*delta1*inputs[p]
        offset1batch += ETA*delta1[0]

    weight1 -= weight1batch
    weight2 -= weight2batch
    offset1 -= offset1batch
    offset2 -= offset2batch

    if t%1000==0:
        print('time=%d: error=%f' % (t, error/4))

print(out)
