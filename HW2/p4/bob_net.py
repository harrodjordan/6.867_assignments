import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        a
        pred = torch.round(output)
        correct = pred.eq(target.float().view_as(pred)).sum().item()
        sum_num_correct += correct
        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size))
            )
            sum_num_correct = 0
            sum_loss = 0
            num_batches_since_log = 0

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output.view(target.shape), target.float()) # sum up the mean square loss
            pred = torch.round(output) #round the prediction to the nearest class
            correct += pred.eq(target.float().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    ##################################################################
    # NOTE - DO NOT ONLY CHANGE THIS FILE
    ##################################################################

    #-------------------------------------------------------------#
    # SETUP - DO NOT CHANGE
    #-------------------------------------------------------------#
    args = dict()
    args["seed"] = 1
    args["no_cuda"] = False
    args["log_interval"] = 50
    args["batch_size"] = 64
    args["test-batch-size"] = 1000

    #-------------------------------------------------------------#
    # HYPERPARAMETERS - DO NOT CHANGE
    #-------------------------------------------------------------#
    params = dict()
    params["epochs"] = 10
    params["lr"] = 1e-2
    #-------------------------------------------------------------#

    torch.manual_seed(args["seed"])
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args["batch_size"], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args["test-batch-size"], shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params["lr"])

    for epoch in range(1, params["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()