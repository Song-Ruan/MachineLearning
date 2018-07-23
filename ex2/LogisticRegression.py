import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

np.seterr(invalid='ignore',divide='ignore')

data=np.loadtxt("ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]
m=data.shape[0]
y=y.reshape((m,1))

#画出散点图

label0=np.where(y.ravel()==0)
plt.title("Figure 1: Scatter plot of training data")
plt.scatter(x[label0,0],x[label0,1],marker='x',color='blue',label="Not Admitted")#scatter的参数写法和plot的写法不一样
label1=np.where(y.ravel()==1)
plt.scatter(x[label1,0],x[label1,1],marker='o',color='red',label="Admitted")
plt.legend(loc="upper right")#用于显示图例
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
#plt.show()


#sigmoid function 需适用于矩阵
def sigmoidFunction(x):
    return 1.0/(1+np.exp(-x))


#compute Cost
def computeCost(x,y,theta):
    m=y.shape[0]
    h=sigmoidFunction(np.dot(x,theta))
    return np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m



#gradient Descent
def gradientDescent(x,y,theta,alpha,iterations):#本来一直以为是梯度下降函数写错了，找半天都没发现错，其实并没有错呀。只是这破东西对a,和迭代次数要求高
    m=y.shape[0]
    j_result=np.zeros((iterations,1))
    for it in range(iterations):
        h = sigmoidFunction(np.dot(x, theta))
        theta=theta-alpha*np.dot(x.T,h-y)/m#不用sum
        j_result[it]=computeCost(x,y,theta)
    return theta,j_result

ones=np.ones((m,1))
x=np.hstack((ones,x))
theta=np.zeros((3,1))
print("Intial Cost:")
print(computeCost(x,y,theta))
alpha=0.002
iterations=1500000
#theta,j_result=gradientDescent(x,y,theta,alpha,iterations)
print("After gradient descent")
print(theta)
print(computeCost(x,y,theta))

#使用scipy中的opt
params=np.zeros((x.shape[1],1)).ravel()
args=(x,y)
#计算代价函数
def f(params,*args):
    xdata,ydata=args
    m,n=xdata.shape
    j=0
    theta=params.reshape((n,1))#因为optimize函数的参数中theta使用的是一维数组，所有需要改一下
    h=sigmoidFunction(np.dot(xdata,theta))
    j=-1*np.sum(ydata*np.log(h)+(1-ydata)*np.log((1-h)))/m
    return j

#传递梯度
def gradf(params,*args):
    xdata, ydata = args
    m, n = xdata.shape
    theta = params.reshape((-1, 1))#reshape参数为-1表示该维度根据另外一个参数来计算
    h = sigmoidFunction(np.dot(xdata, theta))
    grad=np.zeros((xdata.shape[1],1))
    grad=xdata.T.dot((h-ydata))/m
    g=grad.ravel()#optimize函数的参数中grad使用的是一维数组
    return g

res=optimize.fmin_cg(f,x0=params,fprime=gradf,args=args,maxiter=500)
print(res)

#显示决策边界
x1=np.array([20,100,0.5])#和arange区别
y1=np.array((-res[0]-x1*res[1])/res[2])
plt.plot(x1,y1,'-',color='b')
plt.show()

#预测函数
def predict(x,theta):
    h=sigmoidFunction(x.dot(theta))
    res=np.where(h>=0,1,0)
    return res
test1=np.array([[1,45,85]])
theta2=res.reshape((3,1))
print(predict(test1,theta2))



