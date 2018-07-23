import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
#读取数据
data=np.loadtxt("ex1data2.txt",delimiter=',')

m=data.shape[0]
x=data[:,[0,1]]
y=data[:,2]

y=y.reshape((m,1))

#特征归一化
def featureNomarlization(x):
    x_norm=x
    dstd=np.std(x,axis=0)#axis表示求的是各列的最值
    davg=np.mean(x,axis=0)
    x_norm=(x-davg)/dstd
    return x_norm,davg,dstd#应该将特征和方差返回，这样可以在预测新数据之前先缩放



x,davg,dstd=featureNomarlization(x)
#处理输入X矩阵，加上一列1
ones=np.ones((m,1))
x=np.hstack((ones,x))
theta=np.zeros((3,1))



#计算损失函数
def computeCost(x,y,theta):
    m=y.shape[0]
    return np.sum(np.power(np.dot(x,theta)-y,2))/(2*m)#之前这里写错成(np.power(np.dot(x,theta)，2)-y)

print("Initial cost:")
print(computeCost(x,y,theta))
#梯度下降
def grdientDescent(x,y,theta,alpha,iterations):
    m=y.shape[0]
    j_result=np.zeros((iterations,1))
    for it in range(iterations):
        theta=theta-alpha*np.dot(x.T,np.dot(x,theta)-y)/m
        j_result[it]=computeCost(x,y,theta)
    return theta,j_result


alpha=0.005
iterations=1500
theta,j_result=grdientDescent(x,y,theta,alpha,iterations)
print("After gradient Descent theta are: ")
print(theta)

np.set_printoptions(suppress=True)
test1=np.array([[1650,3]])
test1=(test1-davg)/dstd
ones2=np.ones((1,1))
test1=np.hstack((ones2,test1))
result=np.dot(test1,theta)
print(result)

its=np.arange(iterations)
its.reshape((iterations,1))
print(its)
plt.plot(its,j_result,'-',color='blue')
plt.xlabel("iteration")
plt.ylabel("costfunction")
plt.show()








