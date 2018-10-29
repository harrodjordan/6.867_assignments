import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 128)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 50)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(50, 10)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    
    return train_loss, accuracy

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def chelsea_train(num_points, verbose=True):
    # Training settings
    args = dict()
    args["seed"] = 1
    args["no_cuda"] = False
    args["log_interval"] = 50
    args["batch_size"] = 64
    args["test-batch-size"] = 1000
    args["epochs"] = 25
    args["lr"] = 0.01

    torch.manual_seed(args["seed"])
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_set = datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.ToTensor())
    train_size = train_set.train_data.shape[0]
    sample = np.random.choice(train_size, num_points, replace=False)
    train_set.train_data = train_set.train_data[sample]
    train_set.train_labels = train_set.train_labels[sample]
    
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args["batch_size"], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=args["test-batch-size"], shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args["lr"])

    for epoch in range(1, args["epochs"] + 1):
        avg_train_loss, train_acc = train(args, model, device, train_loader, optimizer)
        avg_test_loss, test_acc = test(args, model, device, test_loader)

        if verbose:
            print("Epoch {0}\nTrain Acc: {1:.2f}%, Loss: {2:.5f}\nTest Acc: {3:.2f}%, Loss: {4:.5f}".format(epoch, 100 * train_acc, avg_train_loss, 100 * test_acc, avg_test_loss))

    return avg_train_loss, train_acc, avg_test_loss, test_acc


def build_learning_curve():
    #-------------------------------------------------------------#
    # Write a function that takes builds a learning curve
    # from the results of running chelsea_train() for 
    # various values of `num_points`. You can change any of 
    # the inputs and outputs of this function at will.
    #-------------------------------------------------------------#

    num_points = [100, 500, 1000, 5000, 10000, 30000]
    #num_points=[500, 500]
    n_epochs = 5
    train_loss = np.zeros((6, 1))
    test_loss = np.zeros((6, 1))
    train_acc = np.zeros((6, 1))
    test_acc = np.zeros((6, 1))



    for i in range(len(num_points)):

        for k in range(n_epochs):

            train_loss_temp = 0
            test_loss_temp = 0
            train_acc_temp = 0 
            test_acc_temp = 0

            print("N = {0}\n Rounds = {1}\n".format(i, k))

            train_loss_temp,  train_acc_temp, test_loss_temp, test_acc_temp = chelsea_train(num_points[i], verbose=False)

            train_loss[i] += train_loss_temp
            test_loss[i] += test_loss_temp
            train_acc[i] += train_acc_temp
            test_acc[i] += test_acc_temp

    train_loss = train_loss/5
    train_acc = train_acc/5
    test_acc = test_acc/5
    test_loss = test_loss/5

    plt.figure()
    plt.plot(num_points, train_loss)
    plt.xlabel('N')
    plt.ylabel('Training Loss (Averaged Over 5 Rounds)')
    plt.title('Chelsea Learning Curve - Training Loss')
    plt.show()
    plt.savefig('Chelsea Learning Curve - Training Loss.png')

    plt.figure()
    plt.plot(num_points, train_acc)
    plt.xlabel('N')
    plt.ylabel('Training Accuracy (Averaged Over 5 Rounds')
    plt.title('Chelsea Learning Curve - Training Accuracy')
    plt.show()
    plt.savefig('Chelsea Learning Curve - Training Accuracy.png')

    plt.figure()
    plt.plot(num_points, test_loss)
    plt.xlabel('N')
    plt.ylabel('Test Loss (Averaged Over 5 Rounds')
    plt.title('Chelsea Learning Curve - Test Loss')
    plt.show()
    plt.savefig('Chelsea Learning Curve - Test Loss.png')

    plt.figure()
    plt.plot(num_points, test_acc)
    plt.xlabel('N')
    plt.ylabel('Test Accuracy (Averaged Over 5 Rounds')
    plt.title('Chelsea Learning Curve - Test Accuracy')
    plt.show()
    plt.savefig('Chelsea Learning Curve - Test Accuracy.png')





if __name__ == "__main__":
    build_learning_curve()
    #chelsea_train(100)
