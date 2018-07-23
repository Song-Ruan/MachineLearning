import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures

np.set_printoptions(suppress=True)
data=np.loadtxt("ex2data2.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

y=y.reshape((-1,1))
plt.title("Figure 3: Plot of training data")
label0=np.where(y.ravel()==0)
label1=np.where(y.ravel()==1)
plt.scatter(x[label0,0],x[label0,1],marker='o',color='blue',label="y=0")
plt.scatter(x[label1,0],x[label1,1],marker='x',color='red',label="y=1")
plt.legend("upper right")
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
#plt.show()

#sigmoid
def sigmoid(x):
    return 1.0/(1+np.exp(-x))


#feature mapping将原本2维的特征向量变为28维
def featureMappint(x):
    out=np.ones((x.shape[0],1))
    degree=6
    x1=x[:,0]
    x1=x1.reshape((x.shape[0],1))
    x2=x[:,1]
    x2 = x2.reshape((x.shape[0], 1))
    for i in range(1,degree+1):#range(1,11)是1-10
        for j in range(i+1):
            temp=x1**(i-j)*x2**(j)
            out=np.hstack((out,temp))
    return out

#cost Function
def computeCost(x,y,theta,reg):
    m=x.shape[0]
    h=sigmoid(x.dot(theta))
    theta_1=theta[1:,:]#不包括thet0
    j=-1*np.sum(y*np.log(h)+(1-y)*np.log(1-h))/m+reg/(2*m)*np.sum(np.power(theta_1,2))#不包括thet0
    grad=np.dot(x.T,h-y)/m
    grad[1:,:]+=reg*theta_1/m#不包括thet0
    return j,grad

#测试
x_map=featureMappint(x)
theta=np.zeros((x_map.shape[1],1))
reg=1
cost,grad=computeCost(x_map,y,theta,reg)
print("The inital cost is:")
print(cost)

#使用批量梯度下降
def bgd(x,y,theta,reg,alpha,iterations):
    m=x.shape[0]
    j_result=np.zeros((iterations,1))
    for it in range(iterations):
        cost,grad=computeCost(x,y,theta,reg)
        j_result[it]=cost
        theta=theta-alpha*grad
        if it % 200 == 0:
            print('iter=%d,cost=%f ' % (it, cost))
    return theta

#theta=bgd(x_map,y,theta,reg,0.1,5000)

#使用optimization
#cost Function
def f(params,*args):

    x, y, reg = args
    m = x.shape[0]
    theta=params.reshape((-1,1))
    h=sigmoid(x.dot(theta))
    theta_1=theta[1:,:]#不包括thet0
    j=-1*np.sum(y*np.log(h)+(1-y)*np.log(1-h))/m+reg/(2*m)*np.sum(np.power(theta_1,2))#不包括thet0
    return j

#grad,参数theta为一维数组
def gradf(params,*args):

    x, y ,reg= args
    m = x.shape[0]
    theta = params.reshape((-1, 1))
    h=sigmoid(x.dot(theta))
    theta_1=theta[1:,:]#不包括thet0
    grad=np.dot(x.T,h-y)/m
    grad[1:,:]+=reg*theta_1/m#不包括thet0
    return grad.flatten()



args=(x_map,y,C)
params=np.zeros((x_map.shape[1],1)).ravel()
#res=optimize.fmin_cg(f,x0=params,fprime=gradf,args=args,maxiter=3000)
#res=optimize.minimize(f,params,args,jac=gradf,options={'maxiter':400})






