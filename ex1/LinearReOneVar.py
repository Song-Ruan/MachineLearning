import numpy as np
import matplotlib.pyplot as plt

a=np.eye(2)#生成单位矩阵


'''
问题描述：
使用一个变量实现线性回归，以预测食品卡车的利润。假设你是一家餐馆的首席执行官，正在考虑不同的城市开设一个新的分店。该连锁店已经在各个城市拥有卡车，
而且你有来自城市的利润和人口数据。 
您希望使用这些数据来帮助您选择将哪个城市扩展到下一个城市
training set :ex1data1.txt 第一列为人口，第二列为盈利
'''
#plot the data

#读取txt文件
#unpack表示按列打包成向量
#traindatax,traindatay=np.loadtxt("ex1data1.txt",delimiter=',',unpack=True)
traindata=np.loadtxt("ex1data1.txt",delimiter=',')#生成二维矩阵

#绘制散点图
plt.figure()
plt.title("Figure 1: Scatter plot of training data")
plt.xlim(xmax=24,xmin=4)
plt.ylim(ymax=25,ymin=-5)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.scatter(traindata[:,0],traindata[:,1],color='r',marker='x',linewidths=10)
#plt.plot(traindatax,traindatay,'g*')




#单元线性回归

#初始化
theta=np.array([[0],
                [0]])#2*1

x=traindata[:,0]
y=traindata[:,1]
m=x.size

x=x.reshape((m,1))#让水平数组变成垂直矩阵
y=y.reshape((m,1))


ones=np.ones(m)
ones=ones.reshape((m,1))
x=np.hstack((ones,x))#将两个矩阵水平合并


#计算cost function的值
def computeCost(x,y,theta):
    m = x.shape[0]
    cost= 0.5*np.sum(np.square(((np.dot(x,theta)))-y))/m#np.dot(A,B)是真正代数意义上的矩阵相乘，*或者np.multiply()是对应元素相乘
    return cost


print(computeCost(x,y,theta))
#梯度下降

def gradientDescent(x,y,theta,alpha,iterations):
    #初始化工作
    m=y.shape[0]
    j_result=np.zeros((iterations,1))
    for it in range(iterations):
        #theta=theta-alpha*np.sum(np.dot(x.T,np.dot(x,theta)-y))/m#x.T*(x*theta-y)#这句话写的有问题,不需要np.sum了
        theta = theta - alpha * np.dot(x.T, np.dot(x, theta) - y) / m
        j_result[it]=computeCost(x,y,theta)
    return theta,j_result

alpha=0.01
iterations=1500
theta,j_result=gradientDescent(x,y,theta,alpha,1500)
print(computeCost(x,y,theta))
print("after gradientDescent :")
print(theta)

plt.plot(x[:,1],x.dot(theta),'-',color = 'blue')
plt.legend(['Linear regression','Training data'])#显示图例
plt.show()

