import numpy as np
import pandas as pd

#相关值
cor=np.zeros((21,23))  #存放相关性

for m in range(0,21):
    path = "spearman/alldata/"
    joint = np.loadtxt(path + str(m) + 'joint_spearman.txt', delimiter=',')

    for i in range(0,23):
        data=pd.DataFrame({'feature':joint[:,i],'rating':joint[:,23]})
        data.sort_values('feature', inplace=True)
        data['range1'] = np.arange(1, len(data) + 1)
        data.sort_values('rating', inplace=True)
        data['range2'] = np.arange(1, len(data) + 1)
        data['d'] = data['range1'] - data['range2']
        data['d2'] = data['d'] ** 2
        n = len(data)
        spearman = 1 - 6 * (data['d2'].sum()) / (n * (n ** 2 - 1))
        cor[m,i]=spearman


np.savetxt("spearman_cor.txt",cor,delimiter=",")

print("done!")

