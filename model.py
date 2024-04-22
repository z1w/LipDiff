import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import flatten, conv2mat

class NeuralNet(nn.Module):
    def __init__(self, epochs, model_path, device):
        super(NeuralNet, self).__init__()
        self.epochs = epochs
        self.model_path = model_path
        self.device = device

    def train_model(self, train_loader, optimizer, criterion):
        n_total_steps = len(train_loader)

        # Cyclic LR with single triangle
        lr_peak_epoch = 5
        lr_schedule = np.interp(np.arange((self.epochs + 1) * n_total_steps),
                                [0, lr_peak_epoch * n_total_steps, self.epochs * n_total_steps],
                                [0, 1, 0])
        #scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        for epoch in range(self.epochs):
            total_loss = 0
            for i, (images, labels) in tqdm(enumerate(train_loader)):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                self.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()

                total_loss += float(loss)

                # if (i+1) % 100 == 0:
                #     print (f'Epoch [{epoch+1}/{self.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            if (epoch+1) % 5 == 0:
                # print every 5 epochs
                acc = self.evaluate(train_loader)
                print(f'Epoch [{epoch+1}/{self.epochs}] loss: {total_loss / n_total_steps}, train accuracy: {acc} %')

    def evaluate(self, test_loader):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(images)#.cpu().detach().numpy()
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        # print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))

class Mnist_model(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.l1 = nn.Linear(28*28, 256) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, 10)
        self.epochs=epochs

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
class Mnist_model3(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.l1 = nn.Linear(28*28, 128) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 64) 
        self.l3 = nn.Linear(64, 10)
        self.epochs=epochs

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
class Mnist_model_deep(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.l1 = nn.Linear(28*28, 128) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 10)
        self.epochs=epochs

    def forward(self, x):
        #5 layers
        x = x.reshape(-1, 28*28)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out

class Mnist_CNN(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.l1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16*14*14, 100)
        self.l3 = nn.Linear(100, 10)
        self.epochs=epochs
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = flatten(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
class Mnist_c2f(NeuralNet):
    def __init__(self, epochs, model_path, device, model):
        super().__init__(epochs, model_path, device)
        conv = model.l1
        W, out_shape, bias = conv2mat(conv,[28,28])
        self.l1 = nn.Linear(28*28, W.shape[0])
        self.l1.weight.data = W
        self.l1.bias.data = bias
        self.relu = nn.ReLU()
        l2 = model.l2
        self.l2 = nn.Linear(16*14*14, 100)
        self.l2.weight.data = l2.weight.data
        self.l2.bias.data = l2.bias.data
        l3 = model.l3
        self.l3 = nn.Linear(100, 10)
        self.l3.weight.data = l3.weight.data
        self.l3.bias.data = l3.bias.data

    def forward(self, x):
        #5 layers
        x = x.reshape(-1, 28*28)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class Mnist_CNN_large(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.l1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(8*14*14, 64)
        self.l3 = nn.Linear(64, 10)
        self.epochs=epochs
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = flatten(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
class Mnist_c2f_large(NeuralNet):
    def __init__(self, epochs, model_path, device, model):
        super().__init__(epochs, model_path, device)
        conv = model.l1
        W, out_shape, bias = conv2mat(conv,[28,28])
        self.l1 = nn.Linear(28*28, W.shape[0])
        self.l1.weight.data = W
        self.l1.bias.data = bias
        self.relu = nn.ReLU()
        l2 = model.l2
        self.l2 = nn.Linear(8*14*14, 64)
        self.l2.weight.data = l2.weight.data
        self.l2.bias.data = l2.bias.data
        l3 = model.l3
        self.l3 = nn.Linear(64, 10)
        self.l3.weight.data = l3.weight.data
        self.l3.bias.data = l3.bias.data

    def forward(self, x):
        #5 layers
        x = x.reshape(-1, 28*28)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
class cifar_tiny(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(32*8*8,512)
        self.l2 = nn.Linear(512,512)
        self.l3 = nn.Linear(512, 10)
        self.n1 = nn.BatchNorm2d(32)
        self.n2 = nn.BatchNorm2d(32)
        self.n3 = nn.BatchNorm2d(32)
        self.epochs=epochs
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.n1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.n2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.n3(out)
        out = flatten(out)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

class cifar_large(NeuralNet):
    def __init__(self,  epochs, model_path, device):
        super().__init__(epochs, model_path, device)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(32*8*8,512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 10)
        self.n1 = nn.BatchNorm2d(32)
        self.n2 = nn.BatchNorm2d(32)
        self.n3 = nn.BatchNorm2d(32)
        self.n4 = nn.BatchNorm2d(32)
        self.epochs=epochs
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.n1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.n2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.n3(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.n4(out)
        out = flatten(out)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out