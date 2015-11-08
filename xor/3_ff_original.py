# 3. originalで前向き計算を行う

import numpy as np

ETA = 0.1

def sigmoid(x):
    return 1/(1+np.exp(-x))

inputs = np.array([[0,0],
         [0,1],
         [1,0],
         [1,1]])
teach = np.array([0,1,1,0])

inputs = inputs.astype(np.float32)
teach = teach.astype(np.float32)

weight1 = (np.random.rand(2, 2)-0.5)*2
weight2 = (np.random.rand(1, 2)-0.5)*2
offset1 = (np.random.rand(2)-0.5)*2
offset2 = (np.random.rand(1)-0.5)*2

weight1 = weight1.astype(np.float32)
weight2 = weight2.astype(np.float32)
offset1 = offset1.astype(np.float32)
offset2 = offset2.astype(np.float32)

print(weight1)
print(weight2)
print(offset1)
print(offset2)
print()



# 前向き計算
for p in range(4):
    x1 = np.dot(weight1,inputs[p]).astype(np.float32) + offset1
    x1 = x1.astype(np.float32)
    print('x1')
    print(x1)
    o1 = sigmoid(x1)
    x2 = np.dot(weight2, o1).astype(np.float32) + offset2
    o2 = sigmoid(x2)
    print(o2)

