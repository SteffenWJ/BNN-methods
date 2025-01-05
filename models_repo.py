import torch
import torch.nn as nn
import torch.optim as optim
import bayesian_torch.layers as bnn_layers # used in bayes by backprop

# Infor for bayesian_torch https://github.com/IntelLabs/bayesian-torch

# From December 3rd it is not longer maintained by IntelLabs
# It still works, shows good results

# The Tuple of the bnn_layers is not returned as there is another way to get the K-value
# that i am using, this was to make it easier to work with see traning code for more info

# This is being changed after January into a selection class so it is easier to use
# as most do not need all these and it can be optomised, but for my thesis it made sens as i need
# model after model and modification.

# More later 

class last_layer_BBBN_MNIST(nn.Module):
    #Bayes by backprop last layer model for MINIST
    def __init__(self):
        super(last_layer_BBBN_MNIST, self).__init__()
        self.cnn1 = nn.Conv2d(1, 32, 3, padding=1)
        self.cnn2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = bnn_layers.LinearReparameterization(64 * 7 * 7, 128)   # Bayesian FC layer
        self.fc2 = bnn_layers.LinearReparameterization(128, 10)  # Bayesian FC layer

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        
        x, _ = self.fc1(x)
        x = self.relu(x)
        x, _ = self.fc2(x) 
        return x
    
class last_layer_BBBN_CIFAR_10(nn.Module):
    #Bayes by backprop last layer model for CIFAR-10
    def __init__(self):
        super(last_layer_BBBN_CIFAR_10, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(256 * 4 * 4, 512)   # Bayesian FC layer
        self.BNN2 = bnn_layers.LinearReparameterization(512, 10)  # Bayesian FC layer

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 256 * 4 * 4)
        x, _ = self.BNN1(x) 
        x = self.relu(x)
        x, _ = self.BNN2(x)
        return x
    
class last_layer_BBBN_MNIST(nn.Module):
    #Bayes by backprop last layer model for MINIST
    def __init__(self):
        super(last_layer_BBBN_MNIST, self).__init__()
        self.cnn1 = nn.Conv2d(1, 32, 3, padding=1)
        self.cnn2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = bnn_layers.LinearReparameterization(64 * 7 * 7, 128)   # Bayesian FC layer
        self.fc2 = bnn_layers.LinearReparameterization(128, 10)  # Bayesian FC layer

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        x, _ = self.fc1(x)
        x = self.relu(x)
        x, _ = self.fc2(x)
        return x

    
class last_layer_BBBN_STL_10(nn.Module):
    #Bayes by backprop last layer model for STL-10
    def __init__(self):
        super(last_layer_BBBN_STL_10, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.cnn4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(512 * 6 * 6, 1024)   # Bayesian FC layer
        self.BNN2 = bnn_layers.LinearReparameterization(1024, 10)  # Bayesian FC layer

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x, _ = self.BNN1(x)
        x = self.relu(x)
        x, _ = self.BNN2(x)
        return x

class last_layer_BBBN_ImageNet(nn.Module):
    #Bayes by backprop last layer model for Redcued ImageNet-10
    def __init__(self):
        super(last_layer_BBBN_ImageNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.cnn4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(512 * 4 * 4, 1024)  # Bayesian FC layer
        self.BNN2 = bnn_layers.LinearReparameterization(1024, 10)  # Bayesian FC layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x, _ = self.BNN1(x)
        x = self.relu(x)
        x, _ = self.BNN2(x)
        return x

class FULL_BBBN_ImageNet(nn.Module):
    def __init__(self):
        #Bayes by backprop FULL model for Reduced ImageNet-10
        super(FULL_BBBN_ImageNet, self).__init__()
        self.cnn1 = bnn_layers.Conv2dReparameterization(3, 64, 3, padding=1)
        self.cnn2 = bnn_layers.Conv2dReparameterization(64, 128, 3, padding=1)
        self.cnn3 = bnn_layers.Conv2dReparameterization(128, 256, 3, padding=1)
        self.cnn4 = bnn_layers.Conv2dReparameterization(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = bnn_layers.LinearReparameterization(512 * 4 * 4, 1024)  # Bayesian FC layer
        self.fc2 = bnn_layers.LinearReparameterization(1024, 10)  # Bayesian FC layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x, _ = self.fc1(x)
        x = self.relu(x)
        x, _ = self.fc2(x)
        return x

class Dropout_LL_CNN_ImageNet(nn.Module):
    # Dropout Last Layer for Reduced ImageNet-10
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_LL_CNN_ImageNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.cnn4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.dropout = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Dropout_CNN_ImageNet(nn.Module):
    # Dropout Full model for Reduced ImageNet-10
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_CNN_ImageNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=drop_rate_conv)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=drop_rate_conv)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.dropout3 = nn.Dropout2d(p=drop_rate_conv)
        self.cnn4 = nn.Conv2d(256, 512, 3, padding=1)
        self.dropout4 = nn.Dropout2d(p=drop_rate_conv)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.dropout5 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout4(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x

class CNN_ImageNet(nn.Module):
    # Deterministic CNN model for Reduced ImageNet-10
    def __init__(self):
        super(CNN_ImageNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.cnn4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# I had a type o calling it BBBN and well it is BBBP tryign to fix it, but if you find BBBN i will change it.
class FULL_BBBP_MNIST(nn.Module):
    #Bayes by backprop FULL model for MINIST
    def __init__(self):
        super(FULL_BBBP_MNIST, self).__init__()
        self.conv1 = bnn_layers.Conv2dReparameterization(1, 32, 3, padding=1)
        self.conv2 = bnn_layers.Conv2dReparameterization(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = bnn_layers.LinearReparameterization(64 * 7 * 7, 128)  
        self.fc2 = bnn_layers.LinearReparameterization(128, 10) 

        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x, _ = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        x, _ = self.fc1(x)
        x = self.relu(x)
        x, _ = self.fc2(x)
        return x

class FULL_BBBP_CIFAR_10(nn.Module):
    #Bayes by backprop FULL model for CIFAR-10
    def __init__(self):
        super(FULL_BBBP_CIFAR_10, self).__init__()
        self.conv1 = bnn_layers.Conv2dReparameterization(3, 32, kernel_size=3, padding=1)
        self.conv2 = bnn_layers.Conv2dReparameterization(32, 64, kernel_size=3, padding=1)
        self.conv3 = bnn_layers.Conv2dReparameterization(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = bnn_layers.LinearReparameterization(128 * 4 * 4, 256)
        self.fc2 = bnn_layers.LinearReparameterization(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x , _ = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x , _ = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x , _ = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x , _ = self.fc1(x)
        x = self.relu(x)
        x , _ = self.fc2(x)
        return x
    
class FULL_BBBP_STL_10(nn.Module):
    #Bayes by backprop Full model for STL-10
    def __init__(self):
        super(FULL_BBBP_STL_10, self).__init__()
        self.cnn1 = bnn_layers.Conv2dReparameterization(3, 64, 3, padding=1)
        self.cnn2 = bnn_layers.Conv2dReparameterization(64, 128, 3, padding=1)
        self.cnn3 = bnn_layers.Conv2dReparameterization(128, 256, 3, padding=1)
        self.cnn4 = bnn_layers.Conv2dReparameterization(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(512 * 6 * 6, 1024)   
        self.BNN2 = bnn_layers.LinearReparameterization(1024, 10) 

        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x, _ = self.BNN1(x) 
        x = self.relu(x)
        x, _ = self.BNN2(x) 
        return x
    
    

class Dropout_LL_CNN_MNIST(nn.Module):
    # Dropout Last Layer for MNIST
    def __init__(self, drop_rate_fc=0.5):
        super(Dropout_LL_CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout3 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class Dropout_LL_CNN_CIFAR_10(nn.Module):
    # Dropout Last Layer for CIFAR-10
    def __init__(self, drop_rate_fc=0.5):
        super(Dropout_LL_CNN_CIFAR_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout4 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

class Dropout_LL_CNN_STL_10(nn.Module):
    # Dropout Last Layer for STL-10
    def __init__(self, drop_rate_fc=0.5):
        super(Dropout_LL_CNN_STL_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout5 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x
    
class Dropout_CNN_MNIST(nn.Module):
    # Dropout Full model for MNIST
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=drop_rate_conv)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=drop_rate_conv)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout3 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
    
class Dropout_CNN_CIFAR_10(nn.Module):
    # Dropout Full model for CIFAR-10
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_CNN_CIFAR_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=drop_rate_conv)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=drop_rate_conv)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout3 = nn.Dropout2d(p=drop_rate_conv)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout4 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

class Dropout_CNN_STL_10(nn.Module):
    # Dropout FUll model for STL-10
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_CNN_STL_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=drop_rate_conv)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=drop_rate_conv)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.dropout3 = nn.Dropout2d(p=drop_rate_conv)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.dropout4 = nn.Dropout2d(p=drop_rate_conv)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout5 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout4(x)

        x = x.view(-1, 512 * 6 * 6)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class CNN_MNIST(nn.Module):
    # Deterministic CNN model for MNIST
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN_CIFAR_10(nn.Module):
    # Deterministic CNN model for CIFAR-10
    def __init__(self):
        super(CNN_CIFAR_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN_STL_10(nn.Module):
    # Deterministic CNN model for STL-10
    def __init__(self):
        super(CNN_STL_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024) 
        self.fc2 = nn.Linear(1024, 10)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model_class, file_path):
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    return model

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def save_checkpoint(model, optimizer, file_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(model_class, optimizer_class, file_path):
    checkpoint = torch.load(file_path, weights_only=True)
    model = model_class()
    optimizer = optimizer_class(model.parameters())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #model.eval() #Removed as it is set during evaluation, or traning. Also Dropout need it to be in train mode
    print(f"Checkpoint loaded from {file_path}")
    return model, optimizer

#Debug code for the models
def test_model_VI_Standard(model, input_shape, device = 'cpu'):
    """
    Test the given model with a random input tensor of the specified shape.

    Parameters:
    model (torch.nn.Module) or ohters: The model to be tested.
    input_shape (tuple): The shape of the input tensor.

    Example input shapes:
    - MNIST shape: (1, 1, 28, 28)
    - CIFAR-10 shape: (1, 3, 32, 32)
    - STL-10 shape: (1, 3, 96, 96)

    Prints:
    - The name of the model class.
    - The shape of the input tensor.
    - The shape of the output tensor.
    """
    input_data = torch.randn(input_shape).to(device)
    with torch.no_grad():
        output = model(input_data)
    print(f"Testing {model.__class__.__name__}")
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")

    
# These classes where used for OOD test    
class last_layer_BBBN_CIFAR_10_class_9(nn.Module):
    #Bayes by backprop last layer model for CIFAR-10 with only 9 classes
    def __init__(self):
        super(last_layer_BBBN_CIFAR_10_class_9, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(256 * 4 * 4, 512)   
        self.BNN2 = bnn_layers.LinearReparameterization(512, 9) 

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 256 * 4 * 4)
        x, _ = self.BNN1(x)
        x = self.relu(x)
        x, _ = self.BNN2(x)
        return x
    
class last_layer_BBBN_STL_10_class_9(nn.Module):
    #Bayes by backprop last layer model for STL-10 with only 9 classes
    def __init__(self):
        super(last_layer_BBBN_STL_10_class_9, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, 3, padding=1)
        self.cnn4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(512 * 6 * 6, 1024)   
        self.BNN2 = bnn_layers.LinearReparameterization(1024, 9) 

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x, _ = self.BNN1(x) 
        x = self.relu(x)
        x, _ = self.BNN2(x) 
        return x

class FULL_BBBN_CIFAR_10_class_9(nn.Module):
    #Bayes by backprop FULL model for CIFAR-10 with only 9 classes
    def __init__(self):
        super(FULL_BBBN_CIFAR_10_class_9, self).__init__()
        self.conv1 = bnn_layers.Conv2dReparameterization(3, 32, kernel_size=3, padding=1)
        self.conv2 = bnn_layers.Conv2dReparameterization(32, 64, kernel_size=3, padding=1)
        self.conv3 = bnn_layers.Conv2dReparameterization(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = bnn_layers.LinearReparameterization(128 * 4 * 4, 256)
        self.fc2 = bnn_layers.LinearReparameterization(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x , _ = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x , _ = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x , _ = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x , _ = self.fc1(x)
        x = self.relu(x)
        x , _ = self.fc2(x)
        return x
    
class FULL_BBBN_STL_10_class_9(nn.Module):
    #Bayes by backprop last layer model for STL-10 with only 9 classes
    def __init__(self):
        super(FULL_BBBN_STL_10_class_9, self).__init__()
        self.cnn1 = bnn_layers.Conv2dReparameterization(3, 64, 3, padding=1)
        self.cnn2 = bnn_layers.Conv2dReparameterization(64, 128, 3, padding=1)
        self.cnn3 = bnn_layers.Conv2dReparameterization(128, 256, 3, padding=1)
        self.cnn4 = bnn_layers.Conv2dReparameterization(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.BNN1 = bnn_layers.LinearReparameterization(512 * 6 * 6, 1024) 
        self.BNN2 = bnn_layers.LinearReparameterization(1024, 9) 

        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x, _ = self.cnn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x, _ = self.BNN1(x) 
        x = self.relu(x)
        x, _ = self.BNN2(x)  
        return x
    
class Dropout_CNN_CIFAR_10_class_9(nn.Module):
    # Dropout Full model for CIFAR-10 with only 9 classes
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_CNN_CIFAR_10_class_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=drop_rate_conv)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=drop_rate_conv)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout3 = nn.Dropout2d(p=drop_rate_conv)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout4 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

class Dropout_CNN_STL_10_class_9(nn.Module):
    # Dropout Full model for STL-10 with only 9 classes
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_CNN_STL_10_class_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=drop_rate_conv)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=drop_rate_conv)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.dropout3 = nn.Dropout2d(p=drop_rate_conv)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.dropout4 = nn.Dropout2d(p=drop_rate_conv)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout5 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(1024, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout4(x)

        x = x.view(-1, 512 * 6 * 6)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class Dropout_LL_CNN_CIFAR_10_class_9(nn.Module):
    # Dropout Last Layer for CIFAR-10 with only 9 classes
    def __init__(self, drop_rate_fc=0.5):
        super(Dropout_LL_CNN_CIFAR_10_class_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout4 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

class Dropout_LL_CNN_STL_10_class_9(nn.Module):
    # Dropout Last Layer for STL-10 with only 9 classes
    def __init__(self, drop_rate_conv=0.2, drop_rate_fc=0.5):
        super(Dropout_LL_CNN_STL_10_class_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout5 = nn.Dropout(p=drop_rate_fc)
        self.fc2 = nn.Linear(1024, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class CNN_CIFAR_10_class_9(nn.Module):
    # Deterministic model for CIFAR-10 with only 9 classes
    def __init__(self):
        super(CNN_CIFAR_10_class_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN_STL_10_class_9(nn.Module):
    # Deterministic model for STL-10 with only 9 classes
    def __init__(self):
        super(CNN_STL_10_class_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024) 
        self.fc2 = nn.Linear(1024, 9)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512 * 6 * 6)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x