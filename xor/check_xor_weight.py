import numpy as np

ETA = 0.1

def sigmoid(x):
    return 1/(1+np.exp(-x))

np.random.seed(0)

inputs = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
teach = [0,1,1,0]

weight1 = np.array([
    [ 1.24737334, 0.28295389],
    [ 0.69207227, 1.58455074]])

weight2 = np.array([
    [ 1.32056296, -0.6910398 ]])

offset1 = np.array([0.0, 0.0])
offset2 = np.array([0.0])

# 前向き計算
for p in range(4):
    out1 = sigmoid(np.dot(weight1,inputs[p])+offset1)
    out2 = sigmoid(np.dot(weight2, out1)+offset2)
    print(out2)