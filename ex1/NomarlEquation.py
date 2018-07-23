import numpy as np

np.set_printoptions(suppress=True)
data=np.loadtxt("ex1data2.txt",delimiter=',')
m=data.shape[0]
x=data[:,[0,1]]
y=data[:,2]

y.reshape((m,1))
ones=np.ones((m,1))
x=np.hstack((ones,x))

def normalEquation(x,y):
    theta=np.zeros((x.shape[1],1))
    theta=np.dot(np.dot(np.linalg.pinv(np.dot(x.T,x)),x.T),y)#np.linalg.pinv()表示逆矩阵
    return theta

theta=normalEquation(x,y)
test1=np.array([[1,1650,3]])
print(np.dot(test1,theta))