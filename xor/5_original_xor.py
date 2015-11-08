# ニューラルネットワーク
# 一番簡単なBP
# XOR

import numpy as np

EPSILON = 1.0
ETA = 0.1
TIME = 10000

def sigmoid(x):
    return 1/(1+np.exp(-EPSILON*x))

inputs = np.array([[0,0],
         [0,1],
         [1,0],
         [1,1]])
teach = np.array([0,1,1,0])

inputs = inputs.astype(np.float32)
teach = teach.astype(np.float32)

weight1 = np.array([
    [-0.1280102 , -0.94814754],
    [ 0.09932496, -0.12935521]])
weight2 = np.array([[-0.1592644 , -0.33933036]])
offset1 = np.array([-0.59070273,  0.23854193])
offset2 = np.array([-0.40069065])

for t in range(TIME):
    error = 0
    out = []

    w1 = np.zeros((2,2))
    w2 = np.zeros((1,2))
    o1 = np.zeros(2)
    o2 = np.zeros(1)

    for p in range(len(inputs)):
        # 前向き計算
        out1 = sigmoid(np.dot(weight1,inputs[p])+offset1)
        out2 = sigmoid(np.dot(weight2, out1)+offset2)
        out.append(out2)
        error += (out2-teach[p])**2

        # BP
        delta2 = (out2-teach[p])*EPSILON*out2*(1.0-out2)
        w2 -= ETA*delta2*out1
        o2 -= ETA*delta2

        delta1 = EPSILON*out1*(1.0-out1)*delta2*weight2
        w1 -= ETA*delta1*inputs[p]
        o1 -= ETA*delta1[0]

    weight2 += w2
    weight1 += w1
    offset2 += o2
    offset1 += o1

    print(error)

print('output')
print(out)
