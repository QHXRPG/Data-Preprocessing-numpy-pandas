import numpy as np
import pandas as pd
from numpy import nan as NA
import matplotlib.pyplot as plt


#%% 检测与处理缺失值
data1 = pd.Series(['a', 'b', np.nan, 'c'])
data1_isnull = data1.isnull() #检测缺失值 bool

data2 = pd.DataFrame(data=np.arange(12).reshape(3,4), columns=['a','b','c','d'])
data2.iloc[2] = np.nan
data2[3] = np.nan
data2_isnull_sum = data2.isnull().sum() #检测并统计缺失值 bool

data1.info()
print("--------------------------------------------------------------------------------------------------")
data2.info() #利用info()方法查看DataFrame的缺失值，可以查看每一列数据的缺失情况



#%% 缺失值处理   删除  pd.dropna()
data = pd.Series([1, NA, 3.5, NA, 7])
data.dropna()
#%% bool值索引选择过滤非缺失值
data = pd.Series([1, NA, 3.5, NA, 7])
not_null = data.notnull()
data[not_null]
#%% pd.dropna()默认丢弃任何含有缺失值的行
data = pd.DataFrame([[1, 5.5,3], [1, NA, NA], [NA, NA, NA], [NA, 5.5, 3]])
data.dropna()
data.dropna(how='all') #丢弃全部丢失的行
data.dropna(how='all', axis=1) #丢弃全部丢失的列
data.dropna(thresh=2) #一行至少具有2个非NAN才能保留



#%% 缺失值处理   填充  fillna()
data = pd.DataFrame(data=np.random.randn(5,3), columns=['a','b','c'])
data.iloc[:4,1] = np.nan
data.iloc[:3,2] = np.nan
data.fillna(0) #用0填充
data.fillna(data.mean()) #均值填充



#%% 数据值替换
data = pd.DataFrame(data={'姓名':['a','b','c','d'], '性别':[0,1,0,1]})
data = data.replace({0:'女', 1:'男'})
#%% map()
data["分数"] = [90,59,75,66]
def grade(x: int):
    if x<60:
        return "不及格"
    elif 60<=x<80:
        return "及格"
    elif x>=80:
        return "优秀"
data["等级"] = data["分数"].map(grade)



#%% 异常值检测
wdf = pd.DataFrame(np.arange(20), columns=['w'])
wdf['y'] = 1.5 * wdf['w'] + 0.5
wdf.iloc[14,1] = 100
wdf.iloc[7,1] = 147
#%% 散点图
wdf.plot(kind='scatter', x='w', y='y')
plt.show()
#%% 箱型线
plt.boxplot(wdf['y'])
plt.show()
#%% 3δ法则
def delta_3(data: pd.DataFrame):
    blidx = (data < data.mean() - 3 * data.std()) | (data > data.mean() + 3 * data.std()) #bool
    idx = np.arange(data.shape[0])[blidx]
    outRange = data.iloc[idx]
    return outRange
delta_3(wdf['y'])



#%% 数据集成 数据冗余度和相关分析
a = np.random.random(20)
b = np.random.random(20)
data = np.array([a,b]).T
data = pd.DataFrame(data, columns=['a','b'])
cov = data.a.cov(data.b)  # a 和 b 的协方差
corr = data.a.corr(data.b) # a 和 b 的相关系数
#%% 数据集成 合并数据 merge()
price = pd.DataFrame(data={'fruit':['apple', 'grape','orange','orange'], 'price':[2,4,6,7]})
amout = pd.DataFrame(data={'fruit':['apple', 'grape','orange'], 'amout':[12,23,17]})
all_1 = pd.merge(price, amout, on='fruit')
all_2 = pd.merge(price, amout, left_on='fruit', right_on='fruit')
all_left = pd.merge(price, amout, how='left') #左链接
all_right = pd.merge(price, amout, how='right') #右链接


#%% 数据归约  小波变换
import cv2
import pywt      # 离散小波变换库
img = cv2.imread("/Users/qiuhaoxuan/Desktop/截图/iShot_2022-11-18_18.10.13.jpg")
img = cv2.resize(img, (448,448))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)  #将多通道图像变为单通道图像
plt.figure('二维小波一级变换')
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

"""将各个子图进行拼接，最后得到一张图"""
AH = np.concatenate([cA, cH+255], axis=1)
VD = np.concatenate([cV+255, cD+255], axis=1)
img = np.concatenate([AH, VD], axis=0)

plt.axis('off')
plt.imshow(img, 'gray')
plt.title('result')
plt.show()



#%% 数据归约  PCA组成成分分析
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
dataset = load_iris()  #一个字典
data = dataset.data
target = dataset.target
pca = PCA(n_components=2)
data_re = pca.fit_transform(data)



#%% 数据变换与数据离散
"""数据规范化  """
a = [47,35,46,87,34,70]
b = [38,74,98,20,47,10]
data = np.array([a,b]).T
dfab = pd.DataFrame(data=data, columns=['a','b'])
min_max_norm = (dfab - dfab.min())/(dfab.max() - dfab.min())  #最小最大规范化 把数据聚集在[0,1.0]之间
zero_norm = (dfab - dfab.mean())/dfab.std()  #零均值标准差
#%%
"""数据的哑变量处理"""
df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red', 'L', 13.5, 'class2'], ['blue', 'XL', 14.3, 'class1']])
df.columns = ['color', 'size', 'price', 'class label']
df_dummies = pd.get_dummies(df)
#%%
"""连续型变量离散化 cut()"""
np.random.seed(666)
score_list = np.random.randint(25,100,size=10)
bins = [0, 59, 70, 80, 100]
data_dut = pd.cut(score_list, bins=bins)
print(score_list)
print(pd.value_counts(data_dut))

"""等频法离散连续型数据"""
def SameRateCut(data:pd.Series, k:int) -> pd.Series:
    w = np.arange(0, 1+1.0/k, 1.0/k)
    w = data.quantile(w)
    data = pd.cut(data, w)
    return data