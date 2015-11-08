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

datasize = len(inputs)

weight1 = (np.random.rand(2, 2)-0.5)*2
weight2 = (np.random.rand(1, 2)-0.5)*2
offset1 = (np.random.rand(2)-0.5)*2
offset2 = (np.random.rand(1)-0.5)*2

batchsize = 2
for t in range(TIME):
    error = 0
    out = []

    weight1batch = np.zeros((2,2))
    weight2batch = np.zeros((1,2))
    offset1batch = np.zeros(2)
    offset2batch = np.zeros(1)

    indexes = np.random.permutation(datasize)
    for p in range(0, datasize, batchsize):
        
        for i in range(p, p+batchsize):
            # 前向き計算
            s = indexes[i]
            out1 = sigmoid(np.dot(weight1,inputs[s])+offset1)
            out2 = sigmoid(np.dot(weight2, out1)+offset2)
            out.append(out2)
            error += (out2-teach[s])**2

            # BP
            delta2 = (out2-teach[s])*EPSILON*out2*(1.0-out2)
            weight2batch += ETA*delta2*out1
            offset2batch += ETA*delta2
            
            delta1 = EPSILON*out1*(1.0-out1)*delta2*weight2
            weight1batch += ETA*delta1*inputs[s]
            offset1batch += ETA*delta1[0]

        weight1 -= weight1batch
        weight2 -= weight2batch
        offset1 -= offset1batch
        offset2 -= offset2batch

    if t%1000==0:
        print('time=%d: error=%f' % (t, error/4))

# 前向き計算
for i in range(datasize):
    out1 = sigmoid(np.dot(weight1,inputs[i])+offset1)
    out2 = sigmoid(np.dot(weight2, out1)+offset2)
    print(out2)
