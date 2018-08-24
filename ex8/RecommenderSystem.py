import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as opt

data=io.loadmat("ex8_movies.mat")
Y=data['Y']#ratings
R=data['R']# is ratings
print(Y.shape,R.shape)
nm,nu=Y.shape
nf=100#电影及用户特征个数

#输出第一个电影的平均分，注意只能把用户有评分数算进来
print(np.sum(Y[0]*R[0])/R[0].sum())

#将电影评分可视化
fig=plt.figure(figsize=(8,8*(1682./943.)))
plt.imshow(Y,cmap='rainbow')
plt.colorbar()
plt.ylabel('Movies (%d)'%nm,fontsize=20)
plt.ylabel('Users (%d)'%nu,fontsize=20)
#plt.show()

#加载数据参数
data=io.loadmat('ex8_movieParams.mat')
X=data['X']
Theta=data['Theta']
num_users=data['num_users']
num_movies=data['num_movies']
num_features=data['num_features']
print(X.shape,Theta.shape)
print(num_users,num_movies,num_features)

#先取小部分数据测试
nu = 4; nm = 5; nf = 3
X = X[:nm,:nf]
Theta = Theta[:nu,:nf]
Y = Y[:nm,:nu]
R = R[:nm,:nu]
def serialize(X,Theta):
    #将X,Theta两个参数合并为一个参数，这样方便统一对待进行求导
    return np.r_[X.flatten(),Theta.flatten()]#r_上下拼，c_左右拼。且注意这个不是函数，用的[]

def deserialize(seq,nm,nu,nf):
    '''从seq中提取X，Theta。前nm*nf个是X的，之后nu#nf是Theta的。
    并且各自要还原为原来的shape'''
    return seq[:nm*nf].reshape(nm,nf),seq[nm*nf:].reshape(nu,nf)#:在前面表示从0开始，在后面表示从当前往后
#计算cost和梯度
def computeCost(params,Y,R,nm,nu,nf,reg=0):
    X,Theta=deserialize(params,nm,nu,nf)
    error=0.5*np.sum(np.square(X.dot(Theta.T)-Y)*R)#不能忘记elementwise R,因为j只应该计算有评分即R不为零的那部分
    reg1=0.5*reg*np.sum(np.square(X))
    reg2=0.5*reg*np.sum(np.square(Theta))
    return error+reg1+reg2

print(computeCost(serialize(X,Theta),Y,R,nm,nu,nf),computeCost(serialize(X,Theta),Y,R,nm,nu,nf,1.5))

#计算梯度
def computeGrad(params,Y,R,nm,nu,nf,l=0):

    X, Theta = deserialize(params, nm, nu, nf)
    dx=((X.dot(Theta.T)-Y)*R).dot(Theta)+l*X#记住求grad的时候千万不要sum，因为此时期待得出的是nm*nf的矩阵而不是一个值
    dtheta=((X.dot(Theta.T)-Y)*R).T.dot(X)+l*Theta
    return serialize(dx,dtheta)

#数值校验
def checkGradient(params,Y,myR,nm,nu,nf,l=0):
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    grad=computeGrad(params,Y,myR,nm,nu,nf,l)

    e=0.0001
    nparams=len(params)#有nparams个梯度需要计算
    e_vec=np.zeros(nparams)#用作delt存放之处

    for i in range(10):
        idx=np.random.randint(0,nparams)
        e_vec[idx]=e#只有要求偏导的某个变量的delt需要有变化
        loss1=computeCost(params-e_vec,Y,myR,nm,nu,nf,l)
        loss2=computeCost(params+e_vec,Y,myR,nm,nu,nf,l)
        numgrad=(loss2-loss1)/(2*e)
        e_vec[idx]=0#需同时求梯度，因此要改回来
        diff=np.linalg.norm(numgrad-grad[idx])/np.linalg.norm(numgrad+grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' %(numgrad, grad[idx], diff))
print("Checking gradient with lambda = 0...")
checkGradient(serialize(X,Theta), Y, R, nm, nu, nf)
print("\nChecking gradient with lambda = 1.5...")
checkGradient(serialize(X,Theta), Y, R, nm, nu, nf, l=1.5)


movies=[]
with open('movie_ids.txt','r',encoding='utf-8') as f:
    for line in f:
        movies.append(' '.join(line.strip().split(' ')[1:]))

#填入自己的爱好
my_ratings=np.zeros((1682,1))
my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

for i in range(len(my_ratings)):
    if my_ratings[i]>0:
        print(my_ratings[i],movies[i])

mat = io.loadmat('ex8_movies.mat')
Y, R = mat['Y'], mat['R']

#把自己的爱好和数据合并
Y=np.c_[Y,my_ratings]
R=np.c_[R,my_ratings!=0]
nm,nu=Y.shape

nf=10

#均值归一化
def featureNormalize(Y,R):
    Ymean=((Y*R).sum(axis=1)/R.sum(axis=1)).reshape(-1,1)
    Ynorm=(Y-Ymean)*R#没有评分的不要计算在内，所以要elementwiseR
    return Ynorm,Ymean

Ynorm,Ymean=featureNormalize(Y,R)

'''
有一个问题是不知道为什么最后输出的评分居然不在0-5范围之内
排除了featureNormalize的错误
'''
#随机生成参数X,Theta矩阵
X=np.random.random((nm,nf))
Theta=np.random.random((nu,nf))
params=serialize(X,Theta)
l=10


res=opt.minimize(fun=computeCost,x0=params,args=(Ynorm,R,nm,nu,nf,l),method='TNC',jac=computeGrad,options={'maxiter':100})


fit_X, fit_Theta = deserialize(res.x, nm, nu, nf)


#用训练好的参数预测用户未评价电影的分数
pred_mat=fit_X.dot(fit_Theta.T)
#之前添加的用户的预测分
pred=pred_mat[:,-1]+Ymean.flatten()
#[:,-1]取所有行的最后一个数据组成

#将预测评分排序并从大到小排列
pred_sorted_idx=np.argsort(pred)[::-1]#[::-1]表示反转倒序

print("Top recommendations for you:")
for i in range(10):
    print('Predicting rating %0.1f for movie %s.' \
          %(pred[pred_sorted_idx[i]],movies[pred_sorted_idx[i]]))

print("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s.'% (my_ratings[i],movies[i]))




