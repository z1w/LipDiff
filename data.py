import torch
import torchvision
import torchvision.transforms as transforms

class MNIST_DATA:
    def __init__(self, train_batch_size, test_batch_size):
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

        test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())


        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=train_batch_size, 
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=test_batch_size, 
                                          shuffle=False)
        

class CIFAR10_DATA:
    def __init__(self, train_batch_size, test_batch_size):

        # download and create datasets
        train_transforms = [transforms.Resize((32, 32)),
                            # transforms.RandomCrop(input_dim, padding=2),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()]
        
        test_transforms = [transforms.Resize((32, 32)),
                           transforms.ToTensor()]

        train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transforms.Compose(train_transforms)
                                         )
        test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose(test_transforms)
                                         )

        #indices = list(range(5, len(test_dataset), 83))
        #indices = indices[0:-1]

        #test_dataset = torch.utils.data.Subset(test_dataset, indices)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=train_batch_size, 
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=test_batch_size, 
                                          shuffle=False)
        