
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

mat = loadmat('ex8data1.mat')
print(mat.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'Xval', 'yval'])
X = mat['X']
Xval, yval = mat['Xval'], mat['yval']
X.shape, Xval.shape, yval.shape
# ((307, 2), (307, 2), (307, 1))

def plot_data():
    plt.figure(figsize=(8,5))
    plt.plot(X[:,0], X[:,1], 'bx')
    # plt.scatter(Xval[:,0], Xval[:,1], c=yval.flatten(), marker='x', cmap='rainbow')

plot_data()
def gaussian(X, mu, sigma2):
    '''
    mu, sigma2参数已经决定了一个高斯分布模型
    因为原始模型就是多元高斯模型在sigma2上是对角矩阵而已，所以如下：
    If Sigma2 is a matrix, it is treated as the covariance matrix.
    If Sigma2 is a vector, it is treated as the sigma^2 values of the variances
    in each dimension (a diagonal covariance matrix)
    output:
        一个(m, )维向量，包含每个样本的概率值。
    '''
# 如果想用矩阵相乘求解exp()中的项，一定要注意维度的变换。
# 事实上我们只需要取对角线上的元素即可。（类似于方差而不是想要协方差）
# 最后得到一个（m，）的向量，包含每个样本的概率，而不是想要一个（m,m）的矩阵
# 注意这里，当矩阵过大时，numpy矩阵相乘会出现内存错误。例如9万维的矩阵。所以画图时不能生成太多数据~！
#     n = len(mu)

#     if np.ndim(sigma2) == 1:
#         sigma2 = np.diag(sigma2)

#     X = X - mu
#     p1 = np.power(2 * np.pi, -n/2)*np.sqrt(np.linalg.det(sigma2))
#     e = np.diag(X@np.linalg.inv(sigma2)@X.T)  # 取对角元素，类似与方差，而不要协方差
#     p2 = np.exp(-.5*e)

#     return p1 * p2

# 下面是不利用矩阵的解法，相当于把每行数据输入进去，不会出现内存错误。
    m, n = X.shape
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)

    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm*exp

def getGaussianParams(X, useMultivariate):
    """
    The input X is the dataset with each n-dimensional data point in one row
    The output is an n-dimensional vector mu, the mean of the data set
    the variances sigma^2, an n x 1 vector 或者是(n,n)矩阵，if你使用了多元高斯函数
    作业这里求样本方差除的是 m 而不是 m - 1，实际上效果差不了多少。
    """
    mu = X.mean(axis=0)
    if useMultivariate:
        sigma2 = ((X-mu).T @ (X-mu)) / len(X)
    else:
        sigma2 = X.var(axis=0, ddof=0)  # 样本方差
    print(mu)
    print(sigma2)
    return mu, sigma2

mu,sigma=getGaussianParams(X,False)
p=gaussian(X,mu,sigma)
print(mu,sigma)
print(p[:5])
def plotContours(mu, sigma2):
    """
    画出高斯概率分布的图，在三维中是一个上凸的曲面。投影到平面上则是一圈圈的等高线。
    """
    delta = .3  # 注意delta不能太小！！！否则会生成太多的数据，导致矩阵相乘会出现内存错误。
    x = np.arange(0,30,delta)
    y = np.arange(0,30,delta)

    # 这部分要转化为X形式的坐标矩阵，也就是一列是横坐标，一列是纵坐标，
    # 然后才能传入gaussian中求解得到每个点的概率值
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel()]  # 按列合并，一列横坐标，一列纵坐标
    z = gaussian(points, mu, sigma2)
    z = z.reshape(xx.shape)  # 这步骤不能忘

    cont_levels = [10**h for h in range(-20,0,3)]
    plt.contour(xx, yy, z, cont_levels)  # 这个levels是作业里面给的参考,或者通过求解的概率推出来。

    plt.title('Gaussian Contours',fontsize=16)
plot_data()
useMV = False
plotContours(*getGaussianParams(X, useMV))
plt.show()

plot_data()
useMV = True
# *表示解元组
plotContours(*getGaussianParams(X, useMV))

