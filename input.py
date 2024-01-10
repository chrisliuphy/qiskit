# In[]
import cmath
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
from math import *
import scipy as sp
from scipy.linalg import expm, sinm, cosm
import random
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import polyfit, poly1d
import matplotlib
from matplotlib import cm
from scipy.optimize import curve_fit 
# from sympy import symbols, N
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress = True)
np.set_printoptions(precision= 8)
import numpy as np
import decimal
decimal.getcontext().prec = 6
from mpmath import *
mp.dps=10
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
import matplotlib.ticker as ticker

font = FontProperties()
font.set_family('Times New Roman')
font.set_size(15)
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# In[]
def hx(axn,bxn,n,k):
    return axn*np.cos(n*k) + bxn*np.sin(n*k)


def hy(ayn,byn,n,k):
    return ayn*np.cos(n*k) + byn*np.sin(n*k)

# 生成数组axn,bxn,ayn,byn
L = 2000  # 数组的第一维长度
N = 4  # 数组的第二维长度

# 创建形状为 (L, N) 的二维数组，其中的元素随机取自 -1 到 1
axn = 2 * np.random.rand(L, N) - 1
bxn = 2 * np.random.rand(L, N) - 1
ayn = 2 * np.random.rand(L, N) - 1
byn = 2 * np.random.rand(L, N) - 1


LL = 100 #input data维度为(LL+1)*2 
k_range = np.linspace(0, 2*np.pi,LL,endpoint=True)

# 初始化HX
HX = np.zeros((L, LL))
HY = np.zeros((L, LL))
# 计算HX中的每个值
for i in range(L):
    for j, k in enumerate(k_range):
        for n in range(N):  # 对n进行求和，范围是[0, 4]
            HX[i, j] += hx(axn[i,n], bxn[i,n], n, k)
            HY[i, j] += hy(ayn[i,n], byn[i,n], n, k)
# print(r'HX',HX)
# print(r'HY',HY)

# 使用column_stack将HX和HY堆叠起来，形成一个新的数组
input_data = np.column_stack((HX.flatten(), HY.flatten()))

# 将input_data数组重新分割为L组，每组包含LL个点的数据
input_data = input_data.reshape(L, LL, 2)

# print(input_data)
for i in range(L):
    for j in range(LL):
        norm = np.sqrt((input_data[i,j,0])**2+(input_data[i,j,1])**2)
        input_data[i,j,0] = input_data[i,j,0]/norm
        input_data[i,j,1] = input_data[i,j,1]/norm

# print(r'final input',input_data)
# print(r'first',input_data[0,:,:])

U = np.zeros((L, LL, 1),dtype=complex)  # 初始化新的数组

for i in range(L):
    for j in range(LL):
        u = input_data[i,j,0] + 1j*input_data[i,j,1]
        # theta = np.angle(u)
        U[i, j, 0] = u#theta
    
# print(r'angle',U)

W = []
for i in range(L):
    w = 0
    for j in range(LL-1):
        w = w - np.angle((U[i,j+1,0]/U[i,j,0]))/np.pi/2
    W.append(round(np.abs(w)))

print(W)      
for i in range(len(W)):
    if W[i] == 3:
        print(i)
        
# %%
# 假设 axn, bxn, ayn, byn 和 w 已经按照代码计算出来

# 合并特征
combined_features = np.stack((axn, bxn, ayn, byn), axis=-1) # 形状变为(L, N, 4)

# 调整形状以添加通道维度
input_data = combined_features.reshape(L, 1, N, 4)

# 转换数据类型为float32
input_data = input_data.astype('float32')

# 确保w是正确的形式
labels = np.array(W).astype('float32')

# 现在 input_data 和 labels 都准备好作为CNN的输入和输出了

# %%
import numpy as np
import os
import h5py

# 指定要保存数据的目录
save_dir = "C:/Users/liusi/Desktop/research/transformer"

# 如果目录不存在，则创建它
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 指定文件名
filename = os.path.join(save_dir, "dataset_test.h5")

# 保存数据到 HDF5 文件
with h5py.File(filename, "w") as f:
    f.create_dataset("input_data", data=input_data)
    f.create_dataset("labels", data=labels)

# 输出文件路径以确认
print("Data saved to", filename)

# %%
