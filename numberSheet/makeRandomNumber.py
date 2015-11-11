import numpy as np
import pandas as pd

N = 10
perm = np.random.permutation(N)
df = pd.DataFrame(perm)
df.T.to_csv('perm.csv', index=False, header=None)

row = 9
col = 12
rand = np.zeros((row,col))
for i in range(row):
    for j in range(col):
        rand[i][j] = np.random.randint(0,N)

df = pd.DataFrame(rand)
df.to_csv('rand.csv', index=False, header=None)
