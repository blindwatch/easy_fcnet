import numpy as np
import pandas as pd
import os
import math
from matplotlib import pyplot as plt

def _shuffle(x, y):  # shuffle函数,作用是在出错之后重新打乱原本的点，让每个点在每一轮尽量被扫到
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]

def _sigmoid(z): #sigmoid函数
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)


def _Relu(x):   #reLU函数，为避免神经元坏死采用slight
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.1 * x[i]
    x = np.array(x).reshape(-1, 1)
    return x


def _tanh(x):   #tanh激活函数
    return np.clip(2 / (1.0 + np.exp(- 2 * x)) - 1, 1e-8 - 1, 1 - 1e-8)


class fc_network:       #模型类定义
    def __init__(self):
        self.lr = 0         #学习率
        self.optimizer = 0      #优化器选择
        self.layer_f = [[]]     #神经层列表
        self.gradient_back = []     #反向传播梯度表
        self.weight = []            #权重表
        self.bias = []              #偏置表

    def set_lerning_rate(self, lr):     #设置学习率
        self.lr = lr
        return

    def set_optimizer(self, opt):   #设置激活函数
        if opt == 'sigmoid':
            self.optimizer = 1
        elif opt == "tanh":
            self.optimizer = 2
        elif opt == "relu":
            self.optimizer = 3

    def make_layer(self, input_n, output_n):    #新建一层神经网络
        self.layer_f.append(np.zeros((output_n, 1)))    #神经元矩阵
        self.gradient_back.append(np.zeros((output_n, 1)))  #反向梯度矩阵
        self.weight.append(np.random.random((output_n, input_n)))   #权重向量初始化
        self.bias.append(np.ones((output_n, 1)))        #偏置

    def cal_gradient(self, x):          #根据激活函数计算梯度
        if self.optimizer == 1:
            return _sigmoid(x) * (1 - _sigmoid(x))  #sigmoid的导数
        elif self.optimizer == 2:
            return 1 - np.square(np.tanh(x))        #tanh的导数
        elif self.optimizer == 3:
            for i in range(len(x)):         #relu的导数
                if x[i] > 0:
                    x[i] = 1
                else:
                    x[i] = -0.1
            x = np.array(x).reshape(-1, 1)
            return x

    def forward(self, x, y):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        self.layer_f[0] = x         #输入向量加入层列表
        w = 0           #i
        for i in range(len(self.weight) - 1):       #一直到倒数第二层，最后一层是softmax
            x_t = self.weight[i].dot(x) + self.bias[i]      #计算加权和
            #print(x_t)
            if self.optimizer == 1:             #更新反向传播梯度和激活下一层
                self.layer_f[i + 1] = _sigmoid(x_t)
                self.gradient_back[i] = _sigmoid(self.layer_f[i + 1]) * (1 - _sigmoid(self.layer_f[i + 1]))
            elif self.optimizer == 2:
                self.layer_f[i + 1] = _tanh(x_t)
                self.gradient_back[i] = 1 - np.square(np.tanh(self.layer_f[i + 1]))
            elif self.optimizer == 3:
                self.layer_f[i + 1] = _Relu(x_t)
                for j in range(len(self.layer_f[i + 1])):
                    self.gradient_back[i][j] = 1 if self.layer_f[i+1][j] > 0 else -0.1
            #print(self.layer_f[i + 1])
            x = self.layer_f[i + 1]
            w = w + 1
        s = np.exp((self.weight[w].dot(x) + self.bias[w]).reshape(-1, 1).astype(np.float))      #对最后一层做softmax并且记录梯度
        sm = np.sum(s)
        out = s / sm
        self.layer_f[w + 1] = out
        self.gradient_back[w] = out - y         #softmax梯度
        return out

    def backward(self):
        for i in range(len(self.gradient_back) - 1, -1, -1):        #反向传播梯度
            if i != 0:
                self.gradient_back[i - 1] = self.gradient_back[i - 1] * self.weight[i].transpose().dot(self.gradient_back[i])   #从这一层把偏导传到下一层
            gradient = self.gradient_back[i].dot(self.layer_f[i].transpose())       #计算W的导数
            self.bias[i] = self.bias[i] - self.lr * self.gradient_back[i]           #更新偏置
            self.weight[i] = self.weight[i] - self.lr * gradient                    #更新梯度


def question_1():       #第一问
    x = np.array([[3, 0.4], [1, 1], [3, 3], [2, 0.5], [3, 1], [1, 3], [1, 2], [2, 2], [3, 2]])
    y = np.array([[1, 0, 0], [1, 0, 0],[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    x_t = x
    y_t = y
    module1 = fc_network()
    module1.set_lerning_rate(0.5)
    module1.set_optimizer('relu')
    module1.make_layer(2, 4)
    module1.make_layer(4, 5)
    module1.make_layer(5, 3)
    it_time = 10000
    it_accuracy_500 = []
    for i in range(it_time):
        x_t, y_t = _shuffle(x_t, y_t)
        count = 0
        for j in range(len(x_t)):
            out = module1.forward(x_t[j], y_t[j])
            if np.argmax(out) == np.argmax(y_t[j]):
                count = count + 1
            module1.backward()
        accuracy = count / 9
        if (i+1) % 10 == 0:
            it_accuracy_500.append(accuracy)

    plt.rcParams['font.sans-serif'] = ['SimHei']        #画图
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(it_accuracy_500)
    plt.title('training accuracy per 10 times')
    plt.legend(['accuracy'])
    plt.savefig('accuracy.png')
    plt.show()
    plt.clf()

def question_2(): #第二问
    x = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1, 1, 1, 1]])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x_t = x
    y_t = y
    module1 = fc_network()
    module1.set_lerning_rate(0.5)
    module1.set_optimizer('tanh')
    module1.make_layer(9, 15)
    #module1.make_layer(3, 5)
    module1.make_layer(15, 3)
    it_time = 100
    it_accuracy_500 = []
    for i in range(it_time):
        x_t, y_t = _shuffle(x_t, y_t)
        count = 0
        for j in range(len(x_t)):
            out = module1.forward(x_t[j], y_t[j])
            if np.argmax(out) == np.argmax(y_t[j]):
                count = count + 1
            module1.backward()
        accuracy = count / 3
        #if (i+1) % 10 == 0:
        it_accuracy_500.append(accuracy)

    plt.rcParams['font.sans-serif'] = ['SimHei']        #画图
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(it_accuracy_500)
    plt.title('training accuracy per 1 times')
    plt.legend(['accuracy'])
    plt.savefig('accuracy.png')
    plt.show()
    plt.clf()

def preprocessing(): #第三问数据预处理，生成训练集和验证集
    data = pd.read_csv('./iris.csv', encoding='big5')
    data = data.iloc[:, 1:]
    data.replace('virginica', 2)
    data.replace('setosa', 0)
    data.replace('versicolor', 1)
    raw_data = data.to_numpy()
    X1, X2, X3 = np.vsplit(raw_data, [50, 100])
    X1_train, X1_val = np.vsplit(X1[:, :-1], [30])
    X2_train, X2_val = np.vsplit(X2[:, :-1], [30])
    X3_train, X3_val = np.vsplit(X3[:, :-1], [30])
    X_TRAIN = np.concatenate((X1_train, X2_train, X3_train), axis=0).astype(np.float)
    X_VAL = np.concatenate((X1_val, X2_val, X3_val), axis=0).astype(np.float)
    Y_TRAIN = np.zeros((90, 3), dtype=np.int)
    Y_VAL = np.zeros((60, 3), dtype=np.int)
    for i in range(90):
        if i < 30:
            Y_TRAIN[i][0] = 1
        elif i < 60:
            Y_TRAIN[i][1] = 1
        else:
            Y_TRAIN[i][2] = 1
    for i in range(60):
        if i < 20:
            Y_VAL[i][0] = 1
        elif i < 40:
            Y_VAL[i][1] = 1
        else:
            Y_VAL[i][2] = 1
    return X_TRAIN, Y_TRAIN, X_VAL, Y_VAL

def question_3():       #第三问
    X_T, Y_T, X_V, Y_V = preprocessing()
    x_t = X_T
    y_t = Y_T
    module1 = fc_network()
    module1.set_lerning_rate(0.05)
    module1.set_optimizer('tanh')
    module1.make_layer(4, 9)
    module1.make_layer(9, 5)
    module1.make_layer(5, 3)
    it_time = 5000
    it_train_accuracy = []
    it_val_prediction = []
    y_label = np.argmax(Y_V, axis=1)
    data = []
    for i in range(it_time):
        print(i)
        x_t, y_t = _shuffle(x_t, y_t)
        count = 0
        for j in range(len(x_t)):
            out = module1.forward(x_t[j], y_t[j])
            if np.argmax(out) == np.argmax(y_t[j]):
                count = count + 1
            module1.backward()
        t_accuracy = count / 90
        if (i+1) % 10 == 0:
            it_train_accuracy.append(t_accuracy)
    count = 0
    for i in range(len(X_V)):
        out = module1.forward(X_V[i], Y_V[i])
        it_val_prediction.append(np.argmax(out))
        if np.argmax(out) == np.argmax(Y_V[i]):
            count = count + 1
    it_val_accuracy = count / 60

    plt.rcParams['font.sans-serif'] = ['SimHei'] #画图
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(it_train_accuracy)
    plt.title('training accuracy per 100 times')
    plt.legend(['accuracy'])
    plt.savefig('accuracy.png')
    plt.show()
    plt.clf()

    for i in range(len(X_V)):
        data.append('data{0}:label={1},prediction:{2};;'.format(str(i + 1), str(y_label[i]), str(it_val_prediction[i])))
    file = open('predict.txt', 'w')  # 写入TXT文件
    for i in range(len(data)):
        s = data[i].replace(';', '\t')
        if (i + 1) % 4 == 0:
            s = s + '\n'
        file.write(s)
    s = 'accuracy:{0}'.format(str(round(it_val_accuracy, 2))) + '\n'
    file.write(s)
    file.close()
    print("保存成功")





#question_1()

#question_2()

question_3()


