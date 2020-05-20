import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pylab as pylab
import scipy.io

#import pygal

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#%matplotlib inline


plt.rc('font' , family='Times New Roman' , size='10.5')


#accuracy
dt = np.array([[0.6423076923076925],
               [0.6294429708222813],
               [0.6610079575596817],
               [0.653315649867374]])
                
svm = np.array([[0.6545092838196285],
               [0.6629310344827586],
               [0.6894562334217506],
               [0.6677055702917772]])
                
nb = np.array([[0.1300397877984085],
               [0.13395225464190982],
               [0.1622679045092838],
               [0.11226790450928381]])

knn = np.array([[0.620026525198939],
               [0.6403846153846154],
               [0.6705570291777189],
               [0.6484084880636605]])

dt = dt * 100
svm = svm * 100
nb = nb * 100
knn = knn * 100

freq = [6 , 7.5 , 8.5 , 10]

plt.figure(figsize=(2.73,1.99))



plt.plot(freq , dt , marker='o' , label='dt')
plt.plot(freq , svm , marker='s' , label='svm')
plt.plot(freq , nb , marker='p' , label='nb')
plt.plot(freq , knn , marker='*' , label='knn')

plt.legend(loc=3)
plt.ylabel('Accuracy(%)')
plt.xlabel('Frequency(Hz)')

plt.xticks(freq)

plt.title('SBP')

plt.grid() #去掉网格线
plt.show()
plt.close()