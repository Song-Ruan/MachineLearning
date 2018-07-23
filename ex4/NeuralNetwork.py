import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#加载数据集
digits=load_digits()
print(digits.keys())
data=digits.data
target=digits.target
print(data.shape)
print(target.shape)

#显示某幅图
y=data[15]

plt.gray()
plt.matshow(digits.images[15])
#plt.show()

#随机选出每个类的5个样本看长什么样
classes=['0','1','2','3','4','5','6','7','8','9']
classes_num=10
random_each_num=5
for j,jval in enumerate(classes):
    idx=np.flatnonzero(j==target)#先选出类j的数据，再再类j的数据中随机选择五个
    idx=np.random.choice(idx,random_each_num,replace=False)
    for i,ival in enumerate(idx):
        pos=i*classes_num+j+1
        plt.subplot(random_each_num,classes_num,pos)
        plt.imshow(digits.images[ival].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(j)
#plt.show()

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

class NeuralNetwork(object):
    def __init__(self,input_size,hidden_size,out_size):#成员变量记得家self.
        self.w1=0.01*np.random.randn(input_size,hidden_size)#D*H
        self.b1=np.zeros(hidden_size)#H
        self.w2=0.01*np.random.randn(hidden_size,out_size)#h*c
        self.b2=np.zeros(out_size)#c

    def loss(self,X,y,reg=0.01):
        m=X.shape[0]
        #前向传播
        a1=X
        a2=sigmoid(np.dot(a1,self.w1)+self.b1)
        a3=sigmoid(np.dot(a2,self.w2)+self.b2)

        #计算loss函数
        #后面正则化项少写一个除以m找半天才找到
        j=-np.sum(y*np.log(a3)+(1-y)*np.log(1-a3))/m+0.5*reg*(np.sum(self.w1*self.w1)+np.sum(self.w2*self.w2))/m

        #反向传播
        delt3=a3-y
        dw2=np.dot(a2.T,delt3)+reg*self.w2
        db2=np.sum(delt3,axis=0)#delt3大小为m*c，m为样本总量,c为输出种类个数，而b的大小应该为c，所以需要按列求和
        delt2=np.dot(delt3,self.w2.T)*dsigmoid(a2)
        dw1=np.dot(a1.T,delt2)+reg*self.w1
        db1=np.sum(delt2,axis=0)

        dw1/=m
        dw2/=m
        db1/=m
        db2/=m


        return j,dw1,dw2,db1,db2

    def train(self,X,y,y_train,X_val,y_val,alpha=0.01,iterations=10000):
        #y是二值化之后的y_train，y_train是X对应的真正的是1,2,3中某个的数字输出
        m=X.shape[0]
        batch_size=150
        loss_list=[]
        accuracy_train=[]
        accuracy_val=[]

        for i in range(iterations):
            #每次随机的选取小样本去计算
            batch_index=np.random.choice(m,batch_size,replace=True)
            X_batch=X[batch_index]
            y_batch=y[batch_index]
            y_train_batch=y_train[batch_index]

            j,dw1,dw2,db1,db2=self.loss(X_batch,y_batch)
            loss_list.append(j)


            self.w1+=-alpha*dw1
            self.w2+=-alpha*dw2
            self.b1+=-alpha*db1
            self.b2+=-alpha*db2

            if i%500==0:
                print("i=%d,loss=%f" %(i,j))
                train_acc=np.mean(y_train_batch==self.predict(X_batch))
                val_acc=np.mean(y_val==self.predict(X_val))
                accuracy_train.append(train_acc)
                accuracy_val.append(val_acc)

        return loss_list,accuracy_train,accuracy_val




    def predict(self,X_test):
        a2=sigmoid(X_test.dot(self.w1)+self.b1)
        a3=sigmoid(a2.dot(self.w2)+self.b2)

        y_pred=np.argmax(a3,axis=1)
        return y_pred
    pass

#梯度检测函数
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    fx=f(x)
    grad=np.zeros_like(x)
    #遍历x，对每一个求近似梯度
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        ix=it.multi_index
        oldval=x[ix]

        ixmax=oldval+h
        x[ix]=ixmax
        yxmax=f(x)

        ixmin=oldval-h
        x[ix]=ixmin
        yxmin=f(x)

        x[ix]=oldval#必须恢复原样
        grad[ix]=0.5*(yxmax-yxmin)/h

        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

#定义一下比较函数，用于两个梯度的对比
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#模拟一个小网络，判断模型的正确性

input_size=4
hidden_size=10
output_size=3
train_size=5

def initial_Model():
    np.random.seed(0)
    return NeuralNetwork(input_size,hidden_size,output_size)

def initial_Data():
    np.random.seed(1)
    XX=10*np.random.randn(train_size,input_size)
    yy=np.array([0, 1, 2, 2, 1])
    return XX,yy
net=initial_Model()
XX,yy=initial_Data()

yy_label=LabelBinarizer().fit_transform(yy)

loss,dw1,dw2,db1,db2=net.loss(XX,yy_label)
f=lambda w:net.loss(XX,yy_label)[0]#匿名函数,f(w)=loss
dw1_=eval_numerical_gradient(f,net.w1,False)
dw2_=eval_numerical_gradient(f,net.w2,False)
db1_=eval_numerical_gradient(f,net.b1,False)
db2_=eval_numerical_gradient(f,net.b2,False)
print ('%s max relative error: %e' % ('W1', rel_error(dw1, dw1_)))
print ('%s max relative error: %e' % ('W2', rel_error(dw2, dw2_)))
print ('%s max relative error: %e' % ('b1', rel_error(db1, db1_)))
print ('%s max relative error: %e' % ('b2', rel_error(db2, db2_)))



#训练模型
#先特征归一化
X_mean=np.mean(data,axis=0)
X=data-X_mean
y=target
#划分训练集和测试集
X_data,X_test,y_data,y_test=train_test_split(X,y,test_size=0.2)

#在训练集中划分训练集和验证集
X_train=X_data[0:1000]
y_train=y_data[0:1000]
X_val=X_data[1000:-1]
y_val=y_data[1000:-1]

#将训练标签二值化
y_train_label=LabelBinarizer().fit_transform(y_train)

classfiy=NeuralNetwork(X_train.shape[1],100,10)
print('start')
loss_list,accuracy_train,accuracy_val=classfiy.train(X_train,y_train_label,y_train,X_val,y_val)
print('end')



#可视化一下
plt.subplot(211)
plt.plot(loss_list)
plt.title('train loss')
plt.xlabel('iters')
plt.ylabel('loss')
plt.subplot(212)
plt.plot(accuracy_val,label='val_acc',color='red')
plt.plot(accuracy_train,label='train_acc',color='blue')
plt.legend('lower left')
plt.xlabel('iters')
plt.ylabel('accurancy')
plt.show()


#预测一下看准确性
y_pred=classfiy.predict(X_test)
accurancy=np.mean(y_test==y_pred)
print(accurancy)

