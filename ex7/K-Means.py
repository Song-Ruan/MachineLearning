import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib.image import  imread

data=io.loadmat("ex7data2.mat")
print(data.keys())
X=data['X']

#找各个点最邻近的点
def findClosestCentroids(X,centroids):
    m=X.shape[0]
    idx=np.zeros((m,1))
    k=centroids.shape[0]
    for i in range(m):
        mindist,mink=1000000,-1
        for j in range(k):
            nowdist=np.sum(np.power(X[i]-centroids[j],2))
            if nowdist<mindist:
                mindist=nowdist
                mink=j
        idx[i,:]=mink
    return idx

#计算每个列表中心作为中心点
def computeCentroids(X,idx):
    centroids=np.zeros((len(np.unique(idx)),X.shape[1]))#先确定输出为k*n
    for i in range(len(np.unique(idx))):
        label=np.where(idx.ravel()==i)
        c_i=np.mean(X[label],axis=0)
        centroids[i]=c_i#append只用于数组，此处为矩阵，所以不可以用append否则得到的是一维数组
    return centroids

init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx=findClosestCentroids(X,init_centroids)
print(idx[:3])

print(computeCentroids(X,idx))

#随机初始化centroids
def randCentroids(X,k):
    m,n=X.shape
    randIndex=np.random.choice(m,k)#在0-m中选K个
    centroids=X[randIndex]
    return centroids

class KMeans(object):
    def __init__(self):
        pass
    def runKMeans(self,X,intial_centroids,k=3,iters=10):
        idx=None
        centroids=None
        #记录质心轨迹,三个iters*n的矩阵
        cent0=np.zeros((iters,intial_centroids.shape[1]))
        cent1 = np.zeros((iters, intial_centroids.shape[1]))
        cent2 = np.zeros((iters, intial_centroids.shape[1]))
        for i in range(iters):
            idx = findClosestCentroids(X, intial_centroids)
            centroids = computeCentroids(X, idx)
            cent0[i,:]=centroids[0,:]
            cent1[i, :] = centroids[1,:]
            cent2[i, :] = centroids[2,:]
            intial_centroids=centroids
        return idx,centroids,cent0,cent1,cent2

kmeans=KMeans()
idx,centroids,cent0,cent1,cent2=kmeans.runKMeans(X,init_centroids)

label0=np.where(idx.ravel()==0)
label1=np.where(idx.ravel()==1)
label2=np.where(idx.ravel()==2)
plt.scatter(X[label0,0],X[label0,1],c="red",marker='x')#本来写的是X[label0:,0],多了一个分号，导致报错
plt.scatter(X[label1,0],X[label1,1],c="green",marker='*')
plt.scatter(X[label2,0],X[label2,1],c="yellow",marker='+')
plt.plot(cent0[:,0],cent0[:,1],'b-o')
plt.plot(cent1[:,0],cent1[:,1],'r-o')
plt.plot(cent2[:,0],cent2[:,1],'g-o')
#plt.show()


#图片压缩
image=imread("bird_small.png")#使用matplotlib.image包
print(image.shape)
plt.imshow(image)
plt.show()



X=image.reshape(-1,3)#N*3，N和像素点个数16384*3

K=16
centroids=randCentroids(X,K)#16*3

idx,centroids_all,cent0,cent1,cent2=kmeans.runKMeans(X,centroids,10)
print(centroids_all.shape)#16*3

img=np.zeros(X.shape)

#将同类别颜色的像素点RGB值改为其类别中心点的RGB值
for j in range(K):
    label=np.where(idx.ravel()==j)
    img[label]=centroids_all[j]

img=img.reshape((128,128,3))
fig,axes=plt.subplots(1,2,figsize=(12,6))
axes[0].imshow(image)
axes[1].imshow(img)
plt.show()

