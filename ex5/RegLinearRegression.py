import numpy as np
import scipy.io as io
from scipy import optimize
import matplotlib.pyplot as plt
#load的是mat文件，使用scipy
data=io.loadmat('ex5data1.mat')
print(data.keys())
X=data['X']
y=data['y']
Xtest=data['Xtest']
ytest=data['ytest']
Xval=data['Xval']
yval=data['yval']

'''
plt.scatter(X,y,c='red',marker='x',)
plt.title("Figure 1: Data")
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of the dam (y)")
plt.xlim(xmax=40,xmin=-50)
plt.ylim(ymax=40,ymin=0)
#plt.show()
'''


def computeCost(X,y,theta,reg):#注意正则化项不包括theta0
    m=X.shape[0]
    grad=np.zeros_like(theta)
    theta=theta.reshape((X.shape[1],1))
    tmp=0.5*np.sum(np.power(X.dot(theta)-y,2))/m
    theta_1=theta[1:,:]#不取第一个值
    tmp+=0.5*reg*np.sum(theta_1*theta_1)/m
    grad=X.T.dot(X.dot(theta)-y)/m
    grad[1:,:]+=reg*theta_1/m
    return tmp,grad

m=X.shape[0]
y=y.reshape((m,-1))
ones=np.ones((m,1))
XX=np.hstack((ones,X))
print(X)
theta=np.array([[1],[1]])
initalcost,initalgrad=computeCost(XX,y,theta,1)
print(initalcost)
print(initalgrad)

def f(params,*args):#注意正则化项不包括theta0,params是扁平化的theta，args是其他所有参数
    X, y, reg=args
    m,n = X.shape
    theta=params.reshape((n,1))
    tmp=0.5*np.sum(np.power(X.dot(theta)-y,2))/m
    theta_1=theta[1:,:]#不取第一个值
    tmp+=0.5*reg*np.sum(theta_1*theta_1)/m
    return tmp
def gradf(params,*args):#注意正则化项不包括theta0,params是扁平化的theta，args是其他所有参数
    X, y, reg=args
    m,n = X.shape
    theta=params.reshape((n,1))
    grad=np.zeros_like(theta)
    theta_1=theta[1:,:]#不取第一个值
    grad=X.T.dot(X.dot(theta)-y)/m
    grad[1:,:]+=reg*theta_1/m
    return grad.ravel()#返回的是扁平化的grad

def train(X,y,reg):
    args=(X,y,reg)
    inital_theta=np.zeros((X.shape[1],1))
    params=inital_theta.ravel()
    res=optimize.fmin_cg(f,x0=params,fprime=gradf,args=args,maxiter=500)
    #该函数fmin_cg调用参数有一定规范，第一个参数为损失函数，第二个为扁平化theta,第三个为扁平化梯度
    return res
res=train(XX,y,0)
print(res)

#可视化结果
'''
plt.scatter(X,y,c='red',marker='x')
plt.plot(X,XX.dot(res),'-')
plt.ylim(ymax=40,ymin=-5)
#plt.show()
'''

#学习曲线,观察训练集m的大小对训练误差和验证误差的影响
#此处要注意调用各个函数之前X，Xval应该已经加入一行截距行了
def learncurve(X,y,Xval,yval,reg):
    m=X.shape[0]
    error_train=[]
    error_val=[]
    for i in range(m):
        theta=train(X[:i+1,:],y[0:i+1],reg)#先训练得出theta
        error_t,g1=computeCost(X[0:i+1,:],y[0:i+1],theta,reg)#后分别计算得出theta后的错误率
        error_v,g2=computeCost(Xval,yval,theta,reg)
        error_train.append(error_t)
        error_val.append(error_v)
    return error_train,error_val

ones2=np.ones((Xval.shape[0],1))
XXval=np.hstack((ones2,Xval))
error_train,error_val=learncurve(XX,y,XXval,yval,0)
print(error_train[:5])
print(error_val[:5])
'''
plt.plot(error_train,'b',linestyle='-',label='err_train')
plt.plot(error_val,'r',linestyle='-',label='err_val')
plt.title("Figure 3: Linear regression learning curve")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend("upper right")
plt.show()
'''
def ployFeatures(X,p):#输入m*1矩阵x，拓展为m*p的矩阵
    ploy_x=np.zeros((X.shape[0],p))
    for i in range(p):
        #ploy_x[:,i]=np.power(X.shape[0],i+1)错误写法,大错特错
        ploy_x[:,i]=X.T**(i+1)

    return ploy_x
def featureNormalize(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    XX=X-mu
    X_norm=XX/sigma
    return X_norm,mu,sigma
X_ploy=ployFeatures(X,8)
X_ploy,mu,sigma=featureNormalize(X_ploy)
X_ploy=np.hstack((np.ones((X.shape[0],1)),X_ploy))
print(mu)
print(sigma)
print(X_ploy[1,:])

#将测试集和验证集都使用mu，和sigma去缩放
Xtest_ploy=ployFeatures(Xtest,8)
Xtest_ploy=(Xtest_ploy-mu)/sigma
Xtest_ploy=np.hstack((np.ones((Xtest.shape[0],1)),Xtest_ploy))

Xval_ploy=ployFeatures(Xval,8)
Xval_ploy=(Xval_ploy-mu)/sigma
Xval_ploy=np.hstack((np.ones((Xval.shape[0],1)),Xval_ploy))

#使用reg=0训练
res1=train(X_ploy,y,0)
print(res1)

def plotFit(mu,sigma,theta,p):
    x = np.linspace(-50,45).reshape(-1,1)
    x_ploy = ployFeatures(x,p)
    x_ploy = x_ploy - mu
    x_ploy = x_ploy / sigma
    x_ploy = np.hstack((np.ones((x.shape[0],1)),x_ploy))
    plt.plot(x,x_ploy.dot(theta),'--',color='black')

    pass

plt.plot(X,y,'rx',markersize= 10,linewidth= 1.5)
plotFit(mu,sigma,res1,p=8)

plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


#绘制学习曲线
train_error,val_error=learncurve(X_ploy,y,Xval_ploy,yval,0)
plt.plot(train_error,'b',linestyle = '-',label = 'err_train')
plt.plot(val_error,'r',linestyle = '-',label = 'err_val')
plt.xlabel('Number of training examples(m)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc = 'upper left')
plt.show()

#改变reg对模型的影响
res1=train(X_ploy,y,100)
print(res1)

plt.plot(X,y,'rx',markersize= 10,linewidth= 1.5)
plotFit(mu,sigma,res1,p=8)

plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


#绘制学习曲线
train_error,val_error=learncurve(X_ploy,y,Xval_ploy,yval,100)
plt.plot(train_error,'b',linestyle = '-',label = 'err_train')
plt.plot(val_error,'r',linestyle = '-',label = 'err_val')
plt.xlabel('Number of training examples(m)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc = 'upper left')
plt.show()
