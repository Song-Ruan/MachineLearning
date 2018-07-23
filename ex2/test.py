import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt("ex2data1.txt",delimiter=',')


X = data[:, :-1]
y = data[:, -1:]


import matplotlib.pyplot as plt
label0 = np.where(y.ravel() == 0)
plt.scatter(X[label0,0],X[label0,1],marker='x',color = 'r',label = 'Not admitted')
label1 = np.where(y.ravel() == 1)
plt.scatter(X[label1,0],X[label1,1],marker='o',color = 'b',label = 'Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc = 'upper left')
plt.show()


# compute the cost计算cost以及梯度gradient
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def out(x, w):
    return sigmoid(np.dot(x, w))


def compute_cost(X_train, y_train, theta):
    m = X_train.shape[0]
    J = 0
    theta = theta.reshape(-1, 1)
    grad = np.zeros((X_train.shape[1], 1))
    h = out(X_train, theta)
    J = -1 * np.sum(y_train * np.log(h) + (1 - y_train) * np.log((1 - h))) / m
    grad = X_train.T.dot((h - y_train)) / m
    grad = grad.ravel()

    return J, grad


# test the grad，用简单的数值测试一下，编写的代码是否计算正确
m = X.shape[0]
one = np.ones((m, 1))
X = np.hstack((one, data[:, :-1]))
W = np.zeros((X.shape[1], 1))

cost, grad = compute_cost(X, y, W)

# 这里使用了最优算法，不是选择梯度下降法，例如你可以选择BFGS等
from scipy import optimize

params = np.zeros((X.shape[1], 1)).ravel()
args = (X, y)


def f(params, *args):
    X_train, y_train = args
    m, n = X_train.shape
    J = 0
    theta = params.reshape((n, 1))
    h = out(X_train, theta)
    J = -1 * np.sum(y_train * np.log(h) + (1 - y_train) * np.log((1 - h))) / m

    return J


def gradf(params, *args):
    X_train, y_train = args
    m, n = X_train.shape
    theta = params.reshape(-1, 1)
    h = out(X_train, theta)
    grad = np.zeros((X_train.shape[1], 1))
    grad = X_train.T.dot((h - y_train)) / m
    g = grad.ravel()
    return g


# res = optimize.minimize(f,x0=init_theta,args=args,method='BFGS',jac=gradf,\
#                        options={'gtol': 1e-6, 'disp': True})
print("-------------")
print(y.shape)
res = optimize.fmin_cg(f, x0=params, fprime=gradf, args=args, maxiter=500)

label = np.array(y)
index_0 = np.where(label.ravel()==0)
plt.scatter(X[index_0,1],X[index_0,2],marker='x'\
            ,color = 'b',label = 'Not admitted',s = 15)
index_1 =np.where(label.ravel()==1)
plt.scatter(X[index_1,1],X[index_1,2],marker='o',\
            color = 'r',label = 'Admitted',s = 15)

#show the decision boundary
x1 = np.arange(20,100,0.5)
x2 = (- res[0] - res[1]*x1) / res[2]
plt.plot(x1,x2,color = 'black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper left')
plt.show()
print(res)
