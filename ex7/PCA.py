import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

data=io.loadmat("ex7data1.mat")
X=data['X']
print(X.shape)

#plt.scatter(X[:,0],X[:,1],facecolors='none',edgecolors='b')
plt.title("Figure 4: Example Dataset 1")
#plt.show()

#PCA之前必须均一化
def featureNormalization(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    return (X-mu)/sigma,mu,sigma


def pca(X):
    #计算协方差矩阵
    cov=X.T.dot(X)/len(X)
    #奇异值分解
    U,S,V=np.linalg.svd(cov)
    return U,S,V

X_norm,mu,sigma=featureNormalization(X)
U,S,V=pca(X_norm)

print(U[0])

def projectData(X,U,k):
    return (X).dot(U[:,:k])#X*Uk

Z=projectData(X_norm,U,1)
print(Z[0])

def recoverData(Z,U,k):
    return Z.dot(U[:,:k].T)#Z*(UK.T)
recData=recoverData(Z,U,1)
print(recData[0])
print(recData.shape)

plt.scatter(X_norm[:,0],X_norm[:,1],facecolor='none',edgecolors='b')
plt.scatter(recData[:,0],recData[:,1],facecolors='none',edgecolors='r')
for i in range(X_norm.shape[0]):
    plt.plot([X_norm[i,0],recData[i,0]],[X_norm[i,1],recData[i,1]],'k--')#参数为两个顶点坐标，即画了一个向量
#plt.show()

#加载数据
images=io.loadmat("ex7faces.mat")
X=images['X']

#可视化前100张图
print(X.shape)#5000*1024
def display_data(X,row,col):
    figs,axes=plt.subplots(row,col,figsize=(8,8))
    for r in range(row):
        for c in range(col):
            axes[r][c].imshow(X[r*col+c].reshape(32,32).T,cmap='Greys_r')
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

display_data(X,10,10)



#进行PCA
X_norm,means,stds=featureNormalization(X)
U,S,V=pca(X_norm)
display_data((U[:,:36]).T,6,6)

print(X_norm.shape)
Z=projectData(X,U,36)

X_rec=recoverData(Z,U,36)
display_data(X_rec,10,10)

plt.show()