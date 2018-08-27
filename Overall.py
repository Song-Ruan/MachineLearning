import numpy as np
import matplotlib.pyplot  as plt
from scipy import optimize
from sklearn.datasets import load_digits
import scipy.io as sio
from sklearn import svm


#关于python用法的一些总结

'''
一、Numpy
'''
'''
#1、读取txt文件，返回结果为一个矩阵，unpack=True表示结果返回一个向量
traindata=np.loadtxt("ex1data1.txt",delimiter=',',unpack=False)

#2、生成数组的几种方式
theta=np.array([[0],
                [0]])#2*1
np.arange(0,10)  # 生成[0,1,2,3,4,5,6,7,8,9] 左开右闭不包括10
np.linspace(1, 10, 10)#构造等差数列 开始值，结束值，共几个数字
np.random.randint(5, size=(2, 4))#生成范围在[0,5)内的随机数矩阵，大小为size

a=[1,2]
np.diag(a) #生成1，2 为对角线的方阵
ones=np.ones((3,1))
zeros=np.zeros((2,1))

np.random.randn(2,3)#随机从标准正态分布中返回生成x*y大小的矩阵
np.random.choice(m,batch_size,replace=True)从m中选择batch_size个
grad=np.zeros_like(x)生成和x大小相同的矩阵，元素全为0

it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])生成x的迭代器，flags=['multi_index']表示对a进行多重索引，具体解释看下面的代码。
op_flags=['readwrite']表示不仅可以对a进行read（读取），还可以write（写入）

#3、对矩阵大小的操作
b=theta.reshape((theta.shape[0],1))
x=np.hstack((ones,zeros))#将两个矩阵水平合并，相对也有vstack

#4、一些矩阵运算
b=np.sum(ones)
b=np.square(ones)
b=np.dot(a,theta)
b=a*a#element wise
dstd=np.std(x,axis=0)#axis=0表示求的是各列的标准差
davg=np.mean(x,axis=0)#求各列均值，返回的是m*1
np.power(np.dot(x,theta)-y,2)#求幂函数
np.linalg.inv(a.dot(theta))#求逆矩阵，参数必须为方阵
np.linalg.pinv(a.dot(theta))#求逆矩阵，参数没有必须为方阵

label0=np.where(a.ravel()==0)#a.ravel()返回a的列表形式，返回满足括号内条件的数组下标
res=np.where(h>=0,1,0)满足条件(condition)，输出x，不满足输出y。
np.exp(-x)#e的-x次方
np.log(x)#log(h)

classes=['0','1','2','3','4','5','6','7','8','9']#此处数组应该用方括号[]而不是花括号{}
num_classes=len(classes)
idxs = np.flatnonzero(target == y)#返回扁平化后矩阵中非零元素的位置（index）
np.random.choice(idxs,each_class_num,replace=False)从idxs中随机选取class_num个元素，取出的时候不会替代

np.argmax(h,axis=1)#按行返回最大的那个数值所在的下标

np.max(np.abs(x - y))

#标签二值化
yy_label=LabelBinarizer().fit_transform(yy)

U,S,V=np.linalg.svd(cov)#奇异值分解

'''


'''
二、matplotlib.pyplot

#1、确定图的横纵坐标名，坐标取值范围，图名，图例
plt.title("Figure 1: Scatter plot of training data")
plt.xlim(xmax=24,xmin=4)
plt.ylim(ymax=25,ymin=-5)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend(['Linear regression','Training data'])#显示图例
plt.legend(loc="upper right")#用于显示图例

#2、散点图
plt.scatter(traindata[:,0],traindata[:,1],color='r',marker='x',linewidths=10)

#3、折线图
plt.plot(x[:,1],x.dot(theta),'-',color = 'blue')
plt.show()

#4、子图
plt.subplot(each_class_num,num_classes,pos)生成each_class_num行num_classes列个图，当前图是第Pos个
plt.imshow(digits.images[ival].astype('uint8'))显示图片
plt.axis('off')#不显示坐标轴
plt.gray()
plt.matshow(digits.images[15])
figs,axes=plt.subplots(row,col,figsize=(8,8))
axes[r][c].imshow(X[r * col + c].reshape(32, 32).T, cmap='Greys_r')

#4、画等高线

#meshgrid函数用两个坐标轴上的点在平面上画格
XX,yy=np.meshgrid(np.arange(Xmin,Xmax,h),np.arange(Ymin,Ymax,h))

Z=pred_fun(np.c_[XX.ravel(),yy.ravel()])#np.c_是按行连接两个矩阵
Z=Z.reshape(XX.shape)
plt.contour(XX,yy,Z)#绘制等高线，输入的参数是对应的网格数据(x,y)以及此网格各个点对应的高度值
'''


'''
三、scipy

#1.optimize
#f：代价函数，x0：输入参数，fprime:返回为列表的梯度函数,args:(x,y),返回值res为列表形式
res=optimize.fmin_cg(f,x0=params,fprime=gradf,args=args,maxiter=500)

#2、io
weight=sio.loadmat('ex3weights.mat')
'''

'''
四、sklearn.datasets

digits=load_digits()#使用sklearn.datasets中的digits数据
data=digits.data
target=digits.target
x=data
y=target
idxs = np.flatnonzero(target == y)#返回扁平化后矩阵中非零元素的位置（index）


#使用sklearn中的svm算法
C=1.0
clf=svm.LinearSVC(C=C)
clf.fit(X1,y1.ravel())
y1_pred=clf.predict(X1)#使用训练好的clf预测样本对应的值
'''