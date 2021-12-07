# -*- coding = utf-8 -*-
# @Time :  2:58
# @Author : cjj
# @File : logistic regression.py
# @Software : PyCharm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import random

temp_dict = {
    'weights': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'size': [0.12,0.18,0.33,0.46,0.44,0.5,0.75,0.8,0.85,1.2],
    'obese' : [0,0,0,0,1,0,1,1,1,1]
}
data_mouse = pd.DataFrame(temp_dict)

#参数模块
iterations = 500
learningrate = 0.5


# mini batch
class Logitregression():
    def __init__(self):
        self.lr = learningrate
        self.iters = iterations
        self.w_lst = []
        self.loss_lst = []

    def initialize_weights(self, n_features):
        w = np.random.random((n_features, 1))
        b = 0
        # 把b的值放到w后面
        self.w = np.insert(w, n_features, b, axis=0)

    def true_prob(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def update_lr(self, iters):
        return learningrate - (learningrate - 0.05) / self.iters * iters

    def fit(self, x, y):
        m_samples, n_features = x.shape
        self.initialize_weights(n_features)
        # x也增加一列常数项
        tempone = np.ones((x.shape[0], 1))
        x = np.hstack((x, tempone))
        #         y = np.reshape(y(m_samples,1))
        # 梯度训练
        for i in range(self.iters):
            self.plot_process()
            # random_n = random.sample(range(m_samples), 5)
            # temp_x = x[random_n]
            # temp_y = y[random_n]
            h_x = x.dot(self.w)  # 5,2@2,1
            y_pred = self.true_prob(h_x)  # 5,1
            w_grad = x.T.dot(y_pred - y)  # 2,5@5,1
            self.w = self.w - self.lr * w_grad
            self.w_lst.append(self.w)
            total_loss = np.exp(sum(y * x.dot(self.w) - np.log(1 + np.exp(x.dot(self.w))))[0])
            self.loss_lst.append(total_loss)
            self.lr = self.update_lr(i)

    def predict(self, x):
        tempone = np.ones((x.shape[0], 1))
        x = np.hstack((x, tempone))
        return self.true_prob(x.dot(self.w))

    def plot_loss(self):
        plt.plot(range(self.iters), self.loss_lst)
        plt.show()

    def plot_process(self):
        plt.cla()
        plt.xlim(0, 1)
        plt.ylim(-0.1, 1.1)
        plt.scatter(data_mouse[data_mouse['obese'] == 0]['weights'], data_mouse[data_mouse['obese'] == 0]['obese'],
                    c='lightblue')
        plt.scatter(data_mouse[data_mouse['obese'] == 1]['weights'], data_mouse[data_mouse['obese'] == 1]['obese'],
                    c='darkblue')
        plt.plot(np.linspace(0, 1, 100), np.exp(np.linspace(0, self.w[0], 100) + self.w[1]) / (
                    1 + np.exp(np.linspace(0, self.w[0], 100) + self.w[1])), label='Linear Regression',
                 c='r')
        plt.xlabel('weights')
        plt.ylabel('obese')
        plt.pause(0.05)

#小白鼠模块
model_LR = Logitregression()
model_LR.fit(data_mouse['weights'].values.reshape(-1,1),data_mouse['obese'].values.reshape(-1,1))

plt.ioff()
plt.show()