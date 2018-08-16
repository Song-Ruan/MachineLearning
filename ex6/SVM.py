import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn import svm

data1=io.loadmat("ex6data1.mat")
print(data1.keys())
X1=data1['X']
y1=data1['y']
print(X1.shape)
print(y1.shape)

label0=np.where(y1==0)
label1=np.where(y1==1)
plt.scatter(X1[label0,0],X1[label0,1],c='black',marker='+')
plt.scatter(X1[label1,0],X1[label1,1],c='yellow',marker='o')
plt.title("Figure 1: Example Dataset 1")
#plt.show()


#使用sklearn中的svm算法
C=1.0
clf=svm.LinearSVC(C=C)
clf.fit(X1,y1.ravel())

y1_pred=clf.predict(X1)#使用训练好的clf预测样本对应的值
acc_train=np.mean(y1_pred==y1.ravel())
print("the accuracy of train data set :",acc_train)

#画决策边界的函数
def plot_decision_boundary(pred_fun,X,y,gap):#pred_fun是传进来的预测函数，gap
    Xmin,Xmax=X[:,0].min()-gap,X[:,0].max()+gap
    Ymin, Ymax = X[:, 1].min() - gap, X[:, 1].max() + gap
    h=0.01#网格两点间距
    #meshgrid函数用两个坐标轴上的点在平面上画格
    XX,yy=np.meshgrid(np.arange(Xmin,Xmax,h),np.arange(Ymin,Ymax,h))

    Z=pred_fun(np.c_[XX.ravel(),yy.ravel()])#np.c_是按行连接两个矩阵
    Z=Z.reshape(XX.shape)
    plt.contour(XX,yy,Z)#绘制等高线，输入的参数是对应的网格数据(x,y)以及此网格各个点对应的高度值
    #plt.scatter(X[:,0],X[:,1],c='yellow')

plot_decision_boundary(lambda x:clf.predict(x),X1,y1,0.1)
plt.title("Linear SVM")
plt.show()


#ex6data2.mat
data2=io.loadmat("ex6data2.mat")
X2=data2['X']
y2=data2['y']

label0=np.where(y2==0)
label1=np.where(y2==1)
plt.scatter(X2[label0,0],X2[label0,1],c='yellow',marker='o')
plt.scatter(X2[label1,0],X2[label1,1],c='black',marker='+')
#plt.show()

#这里需要注意一下，在高斯核函数中，参数sigma  与RBF核函数中的gamma，
# 关系是：gamma= 1/2*(sigma**2)  所以当sigma =0.1 ，gamma= 50
clf2=svm.SVC(kernel='rbf',gamma=50,C=1.0)
clf2.fit(X2,y2.ravel())

y2_pred=clf2.predict(X2)
acc_train2=np.mean(y2_pred==y2)
print("the accuracy of train data set : ",acc_train2)

plot_decision_boundary(lambda x:clf2.predict(x),X2,y2,0.3)
plt.title("RBF-svm")
plt.show()

#ex6data3
data3=io.loadmat("ex6data3.mat")
X3=data3['X']
y3=data3['y']
Xval=data3['Xval']
yval=data3['yval']

label0=np.where(y3==0)
label1=np.where(y3==1)
plt.scatter(X3[label0,0],X3[label0,1],c='yellow',marker='o')
plt.scatter(X3[label1,0],X3[label1,1],c='black',marker='+')
#plt.show()

Cvalues=(0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
sigmavals=Cvalues
best_pair,best_score=(0,0),0

for c in Cvalues:
    for sigma in sigmavals:
        gamma = np.power(sigma, -2.) / 2
        model=svm.SVC(kernel='rbf',gamma=gamma,C=c)
        model.fit(X3,y3.ravel())
        this_score=model.score(Xval,yval)
        if(this_score>best_score):
            best_score=this_score
            best_pair=(c,sigma)

print(best_pair)
print(best_score)
clf3=svm.SVC(kernel='rbf',gamma=50,C=1.0)
clf3.fit(X3,y3.ravel())
plot_decision_boundary(lambda x:clf3.predict(x),X3,y3,0.1)
plt.title("RBF-SVM")
plt.show()
