"""
线性回归，Python实现
拟合单输入单输出的一元一次方程
Time：2017年10月5日
"""
__author__ = '王震'
import  numpy as np
import random
def fit_liner(input,lables,w,b,learning_rate):
    """

    :param input:  真实数据x
    :param lables: 真实数据y
    :param w:       初始权重
    :param b:       初始偏置
    :param learning_rate: 学习速率
    :return:       更新的权重。初始值
    """
    x ,y= np.array(input),np.array(lables)
    size=x.size
    for i in range(1,size):
        w,b=compute_gradient(w,b,x[i],y[i],learning_rate)
    err=compute_error(y,w*x+b*np.ones(x.size))
    print('学习次数',size,'误差: ',err,'W:',w,'B:',b)
    return w,b
def compute_gradient(cur_w,cur_b,x,y,learning_rate):
    """
    梯度下降法更新参数
    :param cur_w: 当前的权重
    :param cur_b: 当前的偏置
    :param x:     x
    :param y:     y
    :param learning_rate: 学习速率
    :return: 更新后的权重w与偏置b
    """
    b_gradient = 0
    m_gradient = 0
    N = float(x.size)
    b_gradient = -(2/N)*(y-cur_w*x-cur_b)
    b_gradient = np.sum(b_gradient,axis=0)
    m_gradient = -(2/N)*x*(y-cur_w*x-cur_b)
    m_gradient = np.sum(m_gradient,axis=0)
    new_b = cur_b - (learning_rate * b_gradient)
    new_w = cur_w - (learning_rate * m_gradient)
    return new_w,new_b
def compute_error(real_data,predict_data):
    """
    误差计算：均方差
    :param real_data: 真实数据
    :param predict_data: 预测值
    :return: 平方差
    """
    totalError = 0
    labels=np.array(real_data)
    predict=np.array(predict_data)
    totalError = (labels-predict)**2
    return np.mean(totalError)
def creat_data_line(w,b,min,max,num):
    """
    y=w*x+b+noise
    :param w:  生成直线的斜率
    :param b:   生成直线的偏置
    :param min: x的最小值
    :param max: x的最大值
    :param num: 随机数的个数
    :return: x，y
    """
    x=np.zeros(num)
    y=np.zeros(num)
    for i in range(1,num):
        x[i]=random.uniform(min,max)
        y[i]=w*x[i]+(b+random.uniform(-0.02,0.02))
        #print(x[i],y[i])
    return x,y
#产生数据：y=x+3 x属于（0,5），20000个数据
x,y=creat_data_line(1,3,0,5,20000)
#训练模型
fit_liner(x,y,0.1,0.2,0.001)