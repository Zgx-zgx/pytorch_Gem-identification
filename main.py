# 导入必要的库
import torch
import torchvision
import torchvision.transforms as transforms
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

batch_size = 16  # 批次大小
num_epochs = 3  # 训练轮数
learning_rate = 0.001  # 学习率

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 加载本地数据集
trainset = torchvision.datasets.ImageFolder(root='./dataset/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

testset = torchvision.datasets.ImageFolder(root='./dataset/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
import torch.nn as nn
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(8192, 2048)
        self.dropout_1=nn.Dropout(0.3)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout_2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, 87)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.flatten(1)
        x = self.fc(x)
        x=self.dropout_1(x)
        x=self.fc2(x)
        x=self.dropout_2(x)
        x = self.fc3(x)
        return x


model = CNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(  # 设置衰减系数
#     optimizer,
#     step_size=100,
#     gamma=0.99,
# )
def train():
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels).item() / batch_size
        if i%len(trainloader)==0: # 每一轮打印一次 正确率
          print('训练---[epoch:%d,   step:%2d] loss: %.3f acc: %.3f ' % (epoch + 1, i + 1, loss.item(), acc))
        total_loss += loss.item()
        total_acc += acc
    return total_loss / len(trainloader), total_acc / len(trainloader)


def test():
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == labels).item() / batch_size
            total_loss += loss.item()
            total_acc += acc
    return total_loss / len(testloader), total_acc / len(testloader)
testacc=[]
trainacc=[]
if __name__ == '__main__':

    for epoch in range(num_epochs):

        train_loss, train_acc = train()
        test_loss, test_acc = test()

        trainacc.append(train_acc*100)
        testacc.append(test_acc*100)
        print('Epoch: %d: train loss: %.3f train acc: %.3f test loss: %.3f test acc: %.3f' % (
        epoch + 1, train_loss, train_acc, test_loss, test_acc))
    torch.save(model.state_dict(), 'result.pth')

    plt.plot(testacc, color='blue', label='test_acc',linewidth=0.5)
    plt.plot(trainacc, color='red', label='train_acc',linewidth=0.5)
    plt.xlabel('epoch')
    plt.ylabel('accuracy rate ')
    plt.title('宝石--分类')  # 标题
    plt.legend()
    plt.savefig("result.jpg", dpi=300)  #保存图片
    plt.show()  # 显示图形