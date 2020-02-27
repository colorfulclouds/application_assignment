#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pywt

import seaborn as sns #绘制confusion matrix heatmap

import sklearn
import os

import tqdm

import time

import warnings

warnings.simplefilter('ignore') #忽略警告


# In[43]:


import scipy
import scipy.io as sio

import scipy.signal

from scipy import linalg

import pandas as pd
from sklearn.decomposition import PCA
#分类器
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

from sklearn.cluster import KMeans

import xgboost
import lightgbm


#模型调节
from mlxtend.feature_selection import SequentialFeatureSelector #特征选择函数 选择合适的feature

#结果可视化
from sklearn.metrics import classification_report , confusion_matrix #混淆矩阵

#相关指标
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

#二分类其多分类化
#from sklearn.multiclass import OneVsOneClassifier
#from sklearn.multiclass import OneVsRestClassifier

#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans

#距离函数 度量向量距离
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity #余弦相似度

#one-hot使用
from keras.utils import to_categorical

from sklearn.preprocessing import label_binarize

#绘图
import matplotlib.pyplot as plt

import scipy.linalg as la

import gc

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


from mpl_toolkits.mplot3d import Axes3D 


# In[13]:


sample_rate = 256 #hz
origin_channel = 16 #5 channel eeg

#采集的通道
#共16 channel
#未使用的channel使用none代替
#reference:a study on performance increasing in ssvep based bci application
SAMPLE_CHANNEL = ['Pz' , 'PO3' , 'PO4' , 'O1' , 'O2' , 'Oz' , 'O9' , 'FP2' ,
                  'C4' , 'C6' , 'CP3' , 'CP1' ,
                  'CPZ' , 'CP2' , 'CP4' , 'PO8']

LABEL2STR = {0:'sen' , 1:'hong' , 2:'zhao',
             3:'fen' , 4:'xiao' , 5:'yu' , 
             6:'bin' , 7:'wang' , 8:'wei' , 
             9:'fei'}

# 减去前多少秒数据 second
# 减去后多少秒数据 second
CLIP_FORWARD = 1
CLIP_BACKWARD = 1

# 单个小段的实验时长
trial_time = 3 #second

#是否进行归一化
#reference:a study on performance increasing in ssvep based bci application
#IS_NORMALIZE = True

#是否进行滤波
#IS_FILTER = False
#EEG频率范围
#reference:a study on performance increasing in ssvep based bci application
LO_FREQ = 0.5
HI_FREQ = 40

#是否陷波
#IS_NOTCH = False
NOTCH_FREQ = 50 #陷波 工频


# # 载入数据

# In[15]:


def load_data(filename):
    
    data = sio.loadmat(file_name=filename)['data_received'] #length*16 matrix
    
    data = data[CLIP_FORWARD * sample_rate : - CLIP_BACKWARD * sample_rate] #首部 尾部 进行裁剪
    
    return data


# In[16]:


def separate(data , label , overlap_length = 128):
    train_data = []
    train_labels = []

    size = sample_rate * trial_time #一小段 256*3 个数据点
    data_length = data.shape[0]
    
    idx = 0

    while idx<data_length-size:
        train_data.append(data[idx : idx+size , :])
        train_labels.append(label)

        idx = idx + (size - overlap_length)

    return np.array(train_data) , np.array(train_labels)


# In[17]:


def shuffle_t_v(filenames):
    # np.random.shuffle(filenames)
    
    return np.random.choice(filenames , size=10) #20次的计算准确率中 每次随机选择10个样本进行训练测试

def combine(freq):
    '''
    待定
    '''
    overlap_length = 2*256 #重叠2秒数据
    
    #保证随机性 进行置乱
    person_0_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/0/' % freq) )
    person_1_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/1/' % freq) )
    person_2_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/2/' % freq) )
    person_3_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/3/' % freq) )
    person_4_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/4/' % freq) )
    person_5_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/5/' % freq) )
    person_6_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/6/' % freq) )
    person_7_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/7/' % freq) )
    person_8_filenames = shuffle_t_v( os.listdir('../incremental/data/base_rf/%s/8/' % freq) )

    #打开信号文件 并 合并
    person_0 = np.concatenate([load_data('../incremental/data/base_rf/%s/0/' % freq + filename) for filename in person_0_filenames] , axis = 0)
    person_1 = np.concatenate([load_data('../incremental/data/base_rf/%s/1/' % freq + filename) for filename in person_1_filenames] , axis = 0)
    person_2 = np.concatenate([load_data('../incremental/data/base_rf/%s/2/' % freq + filename) for filename in person_2_filenames] , axis = 0)
    person_3 = np.concatenate([load_data('../incremental/data/base_rf/%s/3/' % freq + filename) for filename in person_3_filenames] , axis = 0)
    person_4 = np.concatenate([load_data('../incremental/data/base_rf/%s/4/' % freq + filename) for filename in person_4_filenames] , axis = 0)
    person_5 = np.concatenate([load_data('../incremental/data/base_rf/%s/5/' % freq + filename) for filename in person_5_filenames] , axis = 0)
    person_6 = np.concatenate([load_data('../incremental/data/base_rf/%s/6/' % freq + filename) for filename in person_6_filenames] , axis = 0)
    person_7 = np.concatenate([load_data('../incremental/data/base_rf/%s/7/' % freq + filename) for filename in person_7_filenames] , axis = 0)
    person_8 = np.concatenate([load_data('../incremental/data/base_rf/%s/8/' % freq + filename) for filename in person_8_filenames] , axis = 0)
    
    #============
    #训练数据分段
    train_person_data_0 , train_person_labels_0 = separate(person_0 , label = 0 , overlap_length=overlap_length)
    train_person_data_1 , train_person_labels_1 = separate(person_1 , label = 1 , overlap_length=overlap_length)
    train_person_data_2 , train_person_labels_2 = separate(person_2 , label = 2 , overlap_length=overlap_length)
    train_person_data_3 , train_person_labels_3 = separate(person_3 , label = 3 , overlap_length=overlap_length)
    train_person_data_4 , train_person_labels_4 = separate(person_4 , label = 4 , overlap_length=overlap_length)
    train_person_data_5 , train_person_labels_5 = separate(person_5 , label = 5 , overlap_length=overlap_length)
    train_person_data_6 , train_person_labels_6 = separate(person_6 , label = 6 , overlap_length=overlap_length)
    train_person_data_7 , train_person_labels_7 = separate(person_7 , label = 7 , overlap_length=overlap_length)
    train_person_data_8 , train_person_labels_8 = separate(person_8 , label = 8 , overlap_length=overlap_length)

    #合并数据
    train_data = np.concatenate((train_person_data_0 , train_person_data_1 , train_person_data_2 ,
                                 train_person_data_3 , train_person_data_4 , train_person_data_5 ,
                                 train_person_data_6 , train_person_data_7 , train_person_data_8 ,
                                 ))
    
    train_labels = np.concatenate((train_person_labels_0 , train_person_labels_1 , train_person_labels_2 ,
                                   train_person_labels_3 , train_person_labels_4 , train_person_labels_5 ,
                                   train_person_labels_6 , train_person_labels_7 , train_person_labels_8 ,
                                    ))
    
    #产生索引并置乱
    idx_train_data = list(range(train_data.shape[0]))
    np.random.shuffle(idx_train_data)

    #将训练数据置乱
    train_data = train_data[idx_train_data]
    train_labels = train_labels[idx_train_data]
        
    return train_data , train_labels


# In[18]:


def session_data_labels(session_id , freq , is_training):
    if is_training:
        overlap_length = 256*2
    else:
        overlap_length = 0
    
    str_freq = str(freq)
    
    subjcets = os.listdir('../incremental/data/incremental/%s/s%d/' % (str_freq , session_id)) #受试者ID
    
    data = []
    labels = []
    
    for subjcet in subjcets:
        filenames = os.listdir('../incremental/data/incremental/%s/s%d/%s/' % (str_freq , session_id , subjcet))
        
        person = np.concatenate([load_data('../incremental/data/incremental/%s/s%d/%s/%s' % (str_freq , session_id , subjcet , filename)) for filename in filenames] , axis = 0)
        
        person_data , person_label = separate( person , label = int(subjcet) , overlap_length = overlap_length)
        
        data.append(person_data)
        labels.append(person_label)
    
    #合并数据
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    
    #shuffle
    idx_data = list(range(data.shape[0]))
    np.random.shuffle(idx_data)

    data = data[idx_data]
    labels = labels[idx_data]
    
    return data , labels


# In[19]:


def butter_worth(data , lowcut , highcut , order=6):
    nyq = 0.5 * sample_rate
    
    lo = lowcut / nyq
    hi = highcut / nyq
    
    b,a = scipy.signal.butter(order , [lo , hi] , btype='bandpass')

    return np.array([scipy.signal.filtfilt(b , a , data[: , i]) for i in range(data.shape[1])]).reshape((-1 , origin_channel))

def alpha_subBP_features(data):
    alpha1 = butter_worth(data , 8.5 , 11.5)
    alpha2 = butter_worth(data , 9.0 , 12.5)    
    alpha3 = butter_worth(data , 9.5 , 13.5)   #11.5 后
    alpha4 = butter_worth(data , 8.0 , 10.5)   
    
    return np.array([alpha1 , alpha2 , alpha3 , alpha4])

def beta_subBP_features(data):
    beta1 = butter_worth(data , 15.0 , 30.0) #14.0 前
    beta2 = butter_worth(data , 16.0 , 17.0)    
    beta3 = butter_worth(data , 17.0 , 18.0)    
    beta4 = butter_worth(data , 18.0 , 19.0)    
    
    return np.array([beta1 , beta2 , beta3 , beta4])

def powermean(data):
    #官方demo跳4秒 前4秒为准备阶段
    return  np.power(data[ : , 0] , 2).mean(),             np.power(data[ : , 1] , 2).mean(),             np.power(data[ : , 2] , 2).mean(),             np.power(data[ : , 3] , 2).mean(),             np.power(data[ : , 4] , 2).mean(),             np.power(data[ : , 5] , 2).mean(),             np.power(data[ : , 6] , 2).mean(),             np.power(data[ : , 7] , 2).mean(),             np.power(data[ : , 8] , 2).mean(),             np.power(data[ : , 9] , 2).mean(),             np.power(data[ : , 10] , 2).mean(),             np.power(data[ : , 11] , 2).mean(),             np.power(data[ : , 12] , 2).mean(),             np.power(data[ : , 13] , 2).mean(),             np.power(data[ : , 14] , 2).mean(),             np.power(data[ : , 15] , 2).mean()      
            
def log_subBP_feature_extraction(alpha , beta):
    #alpha
    power_1_a = powermean(alpha[0])
    power_2_a = powermean(alpha[1])
    power_3_a = powermean(alpha[2])
    power_4_a = powermean(alpha[3])
    
    #beta
    power_1_b = powermean(beta[0])
    power_2_b = powermean(beta[1])
    power_3_b = powermean(beta[2])
    power_4_b = powermean(beta[3])
    
    X= np.array(
        [np.log(power_1_a) ,
         np.log(power_2_a) ,
         np.log(power_3_a) ,
         np.log(power_4_a) ,
         np.log(power_1_b) ,
         np.log(power_2_b) ,
         np.log(power_3_b) ,
         np.log(power_4_b)
        ]
        ).flatten()

    return X

def feature_extraction_sub_band_power(data):
    n_features = 128
    X = np.zeros((data.shape[0] , n_features))
    
    for i , datum in enumerate(data):
        alpha = alpha_subBP_features(datum)
        beta = beta_subBP_features(datum)
            
        X[i, :] = log_subBP_feature_extraction(alpha , beta)

    return X


# In[20]:


def concat_and_shuffle(orig_X , orig_y , session_id , freq):
    session_id_data , session_id_labels = session_data_labels(session_id , freq , is_training=True)
    session_id_data = feature_extraction_sub_band_power(session_id_data)
    # session_id_labels = to_categorical(session_id_labels , num_classes=9)
    
    orig_X = np.concatenate((orig_X , session_id_data) , axis=0)
    orig_y = np.concatenate((orig_y , session_id_labels) , axis=0)
    
    idx = list(range(orig_X.shape[0]))
    np.random.shuffle(idx)
    
    orig_X = orig_X[idx]
    orig_y = orig_y[idx]
    
    return orig_X , orig_y


# # PCA降维

# In[ ]:


def cal_time(ss):
    #step = ss[1]-ss[0]
    
    for i in range(1,len(ss)):
        r_v = np.random.choice([np.random.random()*(-0.05) , np.random.random()*0.05])
        #print(r_v)
        ss[i] = ss[0] + r_v
    return ss





# In[ ]:





# In[22]:


X , y = session_data_labels(3 , 6 , is_training=True)
X_sbp = feature_extraction_sub_band_power(X)


# In[54]:


pca = PCA(n_components=3)


# In[55]:


X_pca = pca.fit_transform(X_sbp)


# In[66]:



fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_pca[:,0] , X_pca[:,1]  , X_pca[:,2] , c=y)
plt.show()

# In[42]:


X_sbp.shape


# In[ ]:


# 经过特征提取 降维 聚类


# In[62]:


kmeans = KMeans(n_clusters=9)


# In[63]:


X_pca_kmeans = kmeans.fit_transform(X_pca)


# In[48]:


plt.scatter(X_pca_kmeans[: , 0] , X_pca_kmeans[: , 1] , c=kmeans.labels_)


# In[49]:


plt.scatter(X_pca_kmeans[: , 0] , X_pca_kmeans[: , 1] , c=y)


# In[68]:


plt.close()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_pca[:,0] , X_pca[:,1]  , X_pca[:,2] , c=kmeans.labels_)


# In[ ]:




