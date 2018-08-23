import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

#可视化数据
data=io.loadmat("ex8data1.mat")
print(data.keys())

X=data['X']
Xval=data['Xval']
yval=data['yval']
def plotData(X):
    plt.scatter(X[:,0],X[:,1],c='blue',marker='x')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
#plt.show()

#拟合高斯分布模型
def gaussian(X,mu,sigma):
    '''分两种情况
    1、sigma是一维向量，表示每个特征对应的方差。此时可根据原向量创建一个对角矩阵，便可以到2的情形
    2、sigma是n*n矩阵，代表多维高斯分布的协方差矩阵
    返回值是m*1的向量，每一个元素对应该样本对应求出的P(x)值
    '''
    m,n=X.shape
    if np.ndim(sigma)==1:
        sigma=np.diag(sigma)#np.diag()参数为数组
    #(X-mu).T为n*m，sigma为n*n，维度不合，必须将X每一行一行的进行求解
    print(sigma)
    exp=np.zeros((m,1))
    for r in range(m):
        xrow=X[r]
        #之前一直不对，因为exp的指数忘记乘以0.5
        exp[r] =np.exp(np.dot(-0.5*((xrow-mu).T),np.linalg.inv(sigma)).dot(xrow-mu))
    norm=1./(np.power((2*np.pi),n/2)*np.sqrt(np.linalg.det(sigma)))
    return norm*exp

#求出每个特征的高斯分布参数
def fitGaussianParams(X,useMultivarite):
    #当useMultivarite为true时，表示，使用了多元高斯模型，需返回协方差矩阵，而不仅仅是一个向量
    m,n=X.shape
    mu=X.mean(axis=0)#记得一定要reshape，因为不reshape输出的是列表而不是向量
    if useMultivarite:
        sigma=((X-mu).T@(X-mu))/m#这里写错导致多维高斯分布时出错
    else:
        sigma=X.var(axis=0,ddof=0)
    return mu,sigma

#画出等高线
def plot_contour(mu,sigma):
    h=0.3#网格间距
    XX,yy=np.meshgrid(np.arange(0,30,h),np.arange(0,30,h))
    z=gaussian(np.c_[XX.ravel(),yy.ravel()],mu,sigma)
    z=z.reshape(XX.shape)
    cont_levels = [10 ** h for h in range(-20, 0, 3)]#生成一个数组，元素为10的-20到0次方，间隔为10的三次方渐增
    plt.contour(XX,yy,z,cont_levels)#画cont_levels.len个等高线
    plt.title('Gaussian Contours', fontsize=16)

#先使用一维高斯分布
plotData(X)
plot_contour(*fitGaussianParams(X,False))
plt.show()
#再使用多维高斯分布
plotData(X)
plot_contour(*fitGaussianParams(X,True))
plt.show()


#利用交叉验证集选取阈值
def selectThreshold(yval,pval):
    def computeF1(yval,pval):
        m=len(yval)
        tp=float(len([i for i in range(m) if pval[i]and yval[i]]))#pval=1,yval=1
        fp=float(len([i for i in range(m) if pval[i] and not yval[i]]))#pval=1,yval=0
        fn=float(len([i for i in range(m) if not pval[i] and yval[i]]))#pval=0,yval=1
        prec=tp/(tp+fp) if (tp+fp) else 0
        rec=tp/(tp+fn) if (tp+fn) else 0
        F1=2*prec*rec/(prec+rec) if (prec+rec) else 0#将分母为0区别对待，否则会报错
        return F1

    epsilons=np.linspace(min(pval),max(pval),1000)#linspace(min,max,num)创建min开头到max结尾共num个数的等差数列
    bestF1,bestEpsilon=0,0
    for e in epsilons:
        pval_=pval<e
        thisF1=computeF1(yval,pval_)
        if thisF1>bestF1:
            bestF1=thisF1
            bestEpsilon=e

    return bestF1,bestEpsilon

mu ,sigma=fitGaussianParams(X,useMultivarite=False)#拟合模型使用的是测试集而不是验证集
pval=gaussian(Xval,mu,sigma)
bestF1,bestEpsilon=selectThreshold(yval,pval)
print(bestF1,bestEpsilon)

#把异常点圈出来
y=gaussian(X,mu,sigma)
xx=np.array([X[i] for i in range(len(y)) if y[i]<bestEpsilon])
plotData(X)
plot_contour(mu,sigma)
plt.scatter(xx[:,0],xx[:,1],s=80,facecolors='none',edgecolors='r')
plt.show()

