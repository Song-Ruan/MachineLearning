import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io as sio


np.seterr(invalid='ignore',divide='ignore')
#加载数据
digits=load_digits()
print(digits.keys())
data=digits.data
target=digits.target

#抽样显示数据集
classes=['0','1','2','3','4','5','6','7','8','9']#此处应该用方括号[]而不是花括号{}
num_classes=len(classes)
each_class_num=5
for y, cla in enumerate(classes):
    idxs=np.flatnonzero(target==y)
    idxs=np.random.choice(idxs,each_class_num,replace=False)
    for i,ival in enumerate(idxs):
        pos=i*num_classes+y+1
        plt.subplot(each_class_num,num_classes,pos)
        plt.imshow(digits.images[ival].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cla)
#plt.show()


#正则化的逻辑回归cost和grad
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def j(params,*args):#此处theta为一维数组
    theta=params
    x,y,reg=args
    m=x.shape[0]
    theta_1=theta.reshape((-1,1))
    h=sigmoid(np.dot(x,theta_1))
    theta_2=theta_1[1:,:]
    j=-np.sum(y*np.log(h)+(1-y)*np.log(1-h))/m+0.5*reg*np.dot(theta_2.T,theta_2)/m
    return j

def grad(params,*args):#grad0和其他grad要分开算,返回的grad应该是一维数组
    theta = params
    x, y, reg = args
    m=x.shape[0]
    theta_1=theta.reshape((-1,1))
    h=sigmoid(np.dot(x,theta_1))
    theta_2=theta_1[1:,:]
    grad=np.dot(x.T,h-y)/m
    grad[1:,:]+=reg*theta_2/m
    return grad.flatten()

#一个循环训练十个分类器
def onevsAll(x,y,k,reg):#k为类别个数
    m,n=x.shape
    w=np.zeros((n,k))
    for i in range(k):
        parms=np.zeros((n,1)).ravel()
        args=(x,y==i,reg)#当y==i时为1，否则为0
        res=optimize.fmin_cg(j,x0=parms,fprime=grad,args=args,maxiter=500)
        w[:,i]=res
    return w

#调用
x=data
y=target
y=y.reshape((-1,1))

#均值归一化
x_mean=np.mean(x,axis=0)
x-=x_mean
x=np.hstack((np.ones((x.shape[0],1)),x))#这里一开始写成了np.zeros，显然是不对的
#theta=onevsAll(x,y,10,1.0)
#print(theta.shape)


#预测函数
def predict(x,theta):
    h=sigmoid(np.dot(x,theta))#h.shape=(1790,10),因为刚好对于每一行样本，十个分类器有十个输出分类
    return np.argmax(h,axis=1)
#y_pred=predict(x,theta)
#print ("train accuracy is :",np.mean(y.ravel() == y_pred))


#使用已经训练好神经网络参数的网络
#加载网络参数
weight=sio.loadmat('ex3weights.mat')
Theta1=weight['Theta1']
Theta2=weight['Theta2']
print(Theta1.shape)
print(Theta2.shape)

#加载数据
data=sio.loadmat('ex3data1.mat')
X=data['X']
y=data['y']
print(X.shape)
print(y.shape)

#使用网络预测数据
ones=np.ones((X.shape[0],1))
X=np.hstack((ones,X))
print(X.shape)

def nn_predict(x,y,theta1,theta2):#每一步都要家偏置
    m=x.shape[0]
    ones=np.ones((m,1))
    a1=sigmoid(np.dot(x,theta1.T))
    a1=np.hstack((ones,a1))
    a2=sigmoid(np.dot(a1,theta2.T))
    pred=np.argmax(a2,axis=1)+1
    return pred

pred=nn_predict(X,y,Theta1,Theta2)
print(pred.shape)
print('Train Accuracy is:',np.mean(y==pred.reshape((-1,1)))*100)







