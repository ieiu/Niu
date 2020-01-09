import torch
from torch import nn
import random


def getData(pow_num):
    x = []
    y = []
    num = 2 ** pow_num
    for i in range(num):
        for j in range(num):
            x.append([i, j])
            y.append(i ^ j)
    temp = [(x[i], y[i]) for i in range(len(x))]
    random.shuffle(temp)
    x = [temp[i][0] for i in range(len(temp))]
    y = [temp[i][1] for i in range(len(temp))]
    return x, y


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    #         self.emb.data = torch.zeros(8,4)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x).squeeze()
        return x


class Model_3(nn.Module):
    def __init__(self, emb_dim=3):
        super(Model_3, self).__init__()
        self.emb = nn.Embedding(8, emb_dim)
        self.model_1 = Model_1()
        self.linear3 = nn.Linear(emb_dim, 1)
        torch.nn.init.uniform_(self.emb.weight, a=0, b=1)
        torch.nn.init.uniform_(self.linear3.weight, a=0, b=8)

    def forward(self, x):
        #         print(x)
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.model_1(x)
        x = self.linear3(x).squeeze()
        return x


def train_1():
    x, y = getData(1)
    X = torch.Tensor(x)
    Y = torch.Tensor(y)
    model_1 = Model_1()
    optimizer = torch.optim.Adam(
        model_1.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    print('训练01模型')
    for epoch in range(6001):
        pre = model_1(X)
        loss = loss_fn(pre, Y)

        if epoch % 1000 == 0:
            print('epoch:%d loss:%f' % (epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model_1


def train_2(model_1):
    x, y = getData(3)
    train_x = x[:60]
    train_y = y[:60]
    test_x = x[60:]
    test_y = y[60:]
    X = torch.LongTensor(train_x)
    Y = torch.Tensor(train_y)
    model_3 = Model_3()
    model_3.model_1 = model_1
    model_3.model_1.requires_grad = False
    optimizer = torch.optim.Adam(
        model_3.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    print('模型2开始训练')
    for epoch in range(10000+1):
        pre = model_3(X)
        loss = loss_fn(pre, Y)

        if (epoch) % 1000 == 0:
            print('epoch:%d loss:%f' % (epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('训练集上的效果')
    pre = pre.round().detach().numpy().tolist()
    # for i in range(len(pre)):
    #     print(int(pre[i]), train_y[i])
    pre = model_3(torch.LongTensor(test_x))

    pre = pre.round().detach().numpy().tolist()
    right = 0
    all = 0
    for i in range(len(pre)):
        if int(pre[i]) == test_y[i]:
            right += 1
        all += 1
        # print(int(pre[i]), train_y[i])
    print('%d/%d' % (right, all))
    print(model_3.emb.weight)
    print(model_3.linear3.weight)

train_2(train_1())
