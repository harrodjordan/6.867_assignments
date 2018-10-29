import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        sum_num_correct += correct
        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum_loss / num_batches_since_log, 
                100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size))
            )
            sum_num_correct = 0
            sum_loss = 0
            num_batches_since_log = 0

def test(model, device, test_loader, dataset_name="Test set"):
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
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_datasets():
    train_dataset = datasets.FashionMNIST('data', train=True, download=True,
                    transform=transforms.ToTensor())

    test_dataset = datasets.FashionMNIST('data', train=False,
                    transform=transforms.ToTensor())

    daniels_photos = data_utils.TensorDataset(
        torch.tensor(np.load('daniels_data/daniels_photos.npy')),
        torch.tensor(np.load('daniels_data/daniels_labels.npy')))

    return train_dataset, test_dataset, daniels_photos

def visualize(dataset, num_examples=10):
    random_selection = np.random.choice(len(dataset), num_examples)

    for i in random_selection:
        transforms.ToPILImage()(dataset[i][0]).show()

def training_procedure(train_dataset, test_dataset, daniels_photos):
    #-------------------------------------------------------------#
    # SETUP - DO NOT CHANGE
    #-------------------------------------------------------------#
    args = dict()
    args["seed"] = 73912
    args["no_cuda"] = False
    args["log_interval"] = 100
    args["batch_size"] = 32
    args["test-batch-size"] = 1000

    #-------------------------------------------------------------#
    # HYPERPARAMETERS - DO NOT CHANGE
    #-------------------------------------------------------------#
    params = dict()
    params["epochs"] = 10
    params["lr"] = 0.1

    torch.manual_seed(args["seed"])
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args["batch_size"], shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args["test-batch-size"], shuffle=True, **kwargs)

    daniels_photos_loader = torch.utils.data.DataLoader(daniels_photos,
                        batch_size=args["test-batch-size"], shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params["lr"])

    # Train the model
    for epoch in range(1, params["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    test(model, device, daniels_photos_loader, dataset_name="Daniel's photos")


if __name__ == '__main__':
    train_dataset, test_dataset, daniels_photos = load_datasets()

    #-----------------------------------------------------------------------#
    # NOTE - To visualize a dataset, uncomment one of the following lines
    #-----------------------------------------------------------------------#
    visualize(train_dataset)
    visualize(test_dataset)
    visualize(daniels_photos)

    training_procedure(train_dataset, test_dataset, daniels_photos)
