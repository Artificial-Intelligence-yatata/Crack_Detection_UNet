import glob
import os
import numpy # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib
import matplotlib.pyplot
import cv2
import random
import sklearn
import sklearn.model_selection

import torch
import torchvision
import torchsummary 





class Layer ():

    def __init__(self, type, activationFunction) -> None:
        super().__init__()
    
        self.layer = type
        self.activationFunction = activationFunction

    def forward (self, X):

        if self.activationFunction is not None:
            return self.activationFunction(self.layer(X))
        
        return self.layer(X)



class DoubleConvolutionBlock ():

    def __init__(self, channels_input, channels_output:int, mode, final_output = 1) -> None:
        super().__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        if (mode.lower() == "final"):
            self.final_output = final_output

        self.layers = []

        self.build(mode)

    
    def build (self, mode):

        layer1 = torch.nn.Conv2d(in_channels = self.channels_input, 
        out_channels = self.channels_output, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None)

        activation1 = torch.nn.functional.relu



        layer2 = torch.nn.Conv2d(in_channels = self.channels_output, 
        out_channels = self.channels_output, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None)
        activation2 = torch.nn.functional.relu


        if (mode == "Down"):
            #layer3 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)
            #activation3 = None
            self.layers.append(Layer(layer1, activation1))
            self.layers.append(Layer(layer2, activation2))

        elif (mode == "Up"):
            layer3 = torch.nn.ConvTranspose2d(self.channels_output, (self.channels_output)//2,
             kernel_size = (2, 2), stride=2, padding=0, dilation=1)
            activation3 = None
            
            self.layers.append(Layer(layer1, activation1))
            self.layers.append(Layer(layer2, activation2))
            self.layers.append(Layer(layer3, activation3))

        elif (mode == "Final"):

            layer3 = torch.nn.Conv2d(in_channels = self.channels_output, out_channels = self.final_output,
             kernel_size = (1,1),stride = 1, padding = 0, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None)            
            activation3 = None

            self.layers.append(Layer(layer1, activation1))
            self.layers.append(Layer(layer2, activation2))
            self.layers.append(Layer(layer3, activation3))

        
        

    def forward (self, X):

        for layer in self.layers:

            X = layer.forward(X)
        
        return X



class Model (torch.nn.Module):

    def __init__(self, channels_input = 3, channel_output = 3, img_size = (255, 255)) -> None:
        

        self.channels_input = channels_input
        self.channel_output = channel_output

        self.layers = []

        self.built = False
        self.trained = False
        
        
        self.problem = "segmentation"
        self.problem_type = "supervised"
                

    def build(self):

        
        # Encoder Chain
        # The contracting path follows the typical architecture of a convolutional network.
        # It consists of the repeated application of two 3×3 convolutions
        # (unpadded convolutions), each followed by a rectified linear unit (ReLU)
        # and a 2×2 max pooling operation with stride 2 for downsampling.
        # At each downsampling step we double the number of feature channels.

        self.block1 = DoubleConvolutionBlock(self.channels_input, 64, "Down")
        self.block2 = DoubleConvolutionBlock(64, 128, "Down")
        self.block3 = DoubleConvolutionBlock(128, 256, "Down")
        self.block4 = DoubleConvolutionBlock(256, 512, "Down")

    
        # Decoder Chain
        # Every step in the expansive path consists of an upsampling of the feature map
        # followed by a 2×2 convolution (“up-convolution”) that halves the number
        # of feature channels, a concatenation with the correspondingly cropped feature map
        # from the contracting path, and two 3×3 convolutions, each followed by a ReLU

        self.block5 = DoubleConvolutionBlock(512, 1024, "Up")
        
        self.block6 = DoubleConvolutionBlock(1024, 512, "Up")
        self.block7 = DoubleConvolutionBlock(512, 256, "Up")
        self.block8 = DoubleConvolutionBlock(256, 128, "Up")

        self.block9 = DoubleConvolutionBlock(128, 64, "Final", self.channel_output)
    

        self.built = True

    def forward (self, X):

        self.connections = []

        X = self.block1.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

        X = self.block2.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

        X = self.block3.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

        X = self.block4.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)




        X = self.block5.forward(X)
        
        
        if (X.shape != self.connections[-1].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-1].shape[2:])
        X = torch.cat((self.connections[-1], X), dim = 1)
        X = self.block6.forward(X)

        if (X.shape != self.connections[-2].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-2].shape[2:])
        X = torch.cat((self.connections[-2], X), dim = 1)
        X = self.block7.forward(X)
        
        if (X.shape != self.connections[-3].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-3].shape[2:])
        X = torch.cat((self.connections[-3], X), dim = 1)
        X = self.block8.forward(X)

        if (X.shape != self.connections[-4].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-4].shape[2:])
        X = torch.cat((self.connections[-4], X), dim = 1)
        
        
        return self.block9.forward(X)


    def build_UNet_modified(self) -> None:

        #TODO


        layer1 = torch.nn.Conv2d(in_channels = 3, out_channels = 127, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = False,
                                padding_mode = "same", device=None)
        activation1 = None

        layer2 = torch.nn.BatchNorm2d(32)
        activation2 = torch.nn.functional.relu


        layer3 = torch.nn.Conv2d(in_channels = 3, out_channels = 127, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = False,
                                padding_mode = "same", device=None)
        activation3 = None

        
        layer4 = torch.nn.BatchNorm2d(32)
        activation4 = torch.nn.functional.relu



        layer5 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        activation5 = None

        
        self.layers.append(Layer(layer1, activation1))
        self.layers.append(Layer(layer2, activation2))
        self.layers.append(Layer(layer3, activation3))
        self.layers.append(Layer(layer4, activation4))
        self.layers.append(Layer(layer5, activation5))

        #Down Block
    




        self.built = True


    def debbugFoward (self):

        batch_size = 5
        img_size = 160
        X = torch.randn((batch_size, self.channels_input, img_size, img_size))

        if (self.built == False):
            self.build()

        pred = self.forward(X)
        print("Input Shape: ", X.shape)
        print("Predict Shape: ", pred.shape)
        
        condition = (batch_size == pred.shape[0]) and (pred.shape[1] == self.channel_output)

        assert condition, AssertionError

        return True

    def __summary__(self):
        
        if(self.problem == "segmentation"):
            torchsummary.summary(self, (self.channels_input, 200, 200))
   
    def loadModel (self, path):
        self.load_state_dict(torch.load(path))
        
    def saveModel(self, path):
        torch.save(self.state_dict(), path)
        
    def eval(self):
        pass





class Layer ():

    def __init__(self, type, activationFunction) -> None:
        super().__init__()
    
        self.layer = type
        self.activationFunction = activationFunction

    def forward (self, X):

        if self.activationFunction is not None:
            return self.activationFunction(self.layer(X))
        
        return self.layer(X)



class DoubleConvolutionBlock ():

    def __init__(self, channels_input, channels_output:int, mode, final_output = 1) -> None:
        super().__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        if (mode.lower() == "final"):
            self.final_output = final_output

        self.layers = []

        self.build(mode)

    
    def build (self, mode):

        layer1 = torch.nn.Conv2d(in_channels = self.channels_input, 
        out_channels = self.channels_output, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None)

        activation1 = torch.nn.functional.relu



        layer2 = torch.nn.Conv2d(in_channels = self.channels_output, 
        out_channels = self.channels_output, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None)
        activation2 = torch.nn.functional.relu


        if (mode == "Down"):
            #layer3 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)
            #activation3 = None
            self.layers.append(Layer(layer1, activation1))
            self.layers.append(Layer(layer2, activation2))

        elif (mode == "Up"):
            layer3 = torch.nn.ConvTranspose2d(self.channels_output, (self.channels_output)//2,
             kernel_size = (2, 2), stride=2, padding=0, dilation=1)
            activation3 = None
            
            self.layers.append(Layer(layer1, activation1))
            self.layers.append(Layer(layer2, activation2))
            self.layers.append(Layer(layer3, activation3))

        elif (mode == "Final"):

            layer3 = torch.nn.Conv2d(in_channels = self.channels_output, out_channels = self.final_output,
             kernel_size = (1,1),stride = 1, padding = 0, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None)            
            activation3 = None

            self.layers.append(Layer(layer1, activation1))
            self.layers.append(Layer(layer2, activation2))
            self.layers.append(Layer(layer3, activation3))

        
        

    def forward (self, X):

        for layer in self.layers:

            X = layer.forward(X)
        
        return X



class Model (torch.nn.Module):

    def __init__(self, channels_input = 3, channel_output = 3, img_size = (255, 255)) -> None:
        

        self.channels_input = channels_input
        self.channel_output = channel_output

        self.layers = []

        self.built = False
        self.trained = False
        
        
        self.problem = "segmentation"
        self.problem_type = "supervised"
                

    def build_UNet(self):

        
        # Encoder Chain
        # The contracting path follows the typical architecture of a convolutional network.
        # It consists of the repeated application of two 3×3 convolutions
        # (unpadded convolutions), each followed by a rectified linear unit (ReLU)
        # and a 2×2 max pooling operation with stride 2 for downsampling.
        # At each downsampling step we double the number of feature channels.

        self.block1 = DoubleConvolutionBlock(self.channels_input, 64, "Down")
        self.block2 = DoubleConvolutionBlock(64, 128, "Down")
        self.block3 = DoubleConvolutionBlock(128, 256, "Down")
        self.block4 = DoubleConvolutionBlock(256, 512, "Down")

    
        # Decoder Chain
        # Every step in the expansive path consists of an upsampling of the feature map
        # followed by a 2×2 convolution (“up-convolution”) that halves the number
        # of feature channels, a concatenation with the correspondingly cropped feature map
        # from the contracting path, and two 3×3 convolutions, each followed by a ReLU

        self.block5 = DoubleConvolutionBlock(512, 1024, "Up")
        
        self.block6 = DoubleConvolutionBlock(1024, 512, "Up")
        self.block7 = DoubleConvolutionBlock(512, 256, "Up")
        self.block8 = DoubleConvolutionBlock(256, 128, "Up")

        self.block9 = DoubleConvolutionBlock(128, 64, "Final", self.channel_output)
    

        self.built = True

    def forward (self, X):

        self.connections = []

        X = self.block1.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

        X = self.block2.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

        X = self.block3.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

        X = self.block4.forward(X)
        self.connections.append(X)
        X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)




        X = self.block5.forward(X)
        
        
        if (X.shape != self.connections[-1].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-1].shape[2:])
        X = torch.cat((self.connections[-1], X), dim = 1)
        X = self.block6.forward(X)

        if (X.shape != self.connections[-2].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-2].shape[2:])
        X = torch.cat((self.connections[-2], X), dim = 1)
        X = self.block7.forward(X)
        
        if (X.shape != self.connections[-3].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-3].shape[2:])
        X = torch.cat((self.connections[-3], X), dim = 1)
        X = self.block8.forward(X)

        if (X.shape != self.connections[-4].shape):
            X = torchvision.transforms.functional.resize(X, size = self.connections[-4].shape[2:])
        X = torch.cat((self.connections[-4], X), dim = 1)
        
        
        return self.block9.forward(X)


    def build_UNet_modified(self) -> None:

        #TODO


        layer1 = torch.nn.Conv2d(in_channels = 3, out_channels = 127, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = False,
                                padding_mode = "same", device=None)
        activation1 = None

        layer2 = torch.nn.BatchNorm2d(32)
        activation2 = torch.nn.functional.relu


        layer3 = torch.nn.Conv2d(in_channels = 3, out_channels = 127, kernel_size = (3,3),
                                stride = 1, padding = 1, dilation = 1, bias = False,
                                padding_mode = "same", device=None)
        activation3 = None

        
        layer4 = torch.nn.BatchNorm2d(32)
        activation4 = torch.nn.functional.relu



        layer5 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        activation5 = None

        
        self.layers.append(Layer(layer1, activation1))
        self.layers.append(Layer(layer2, activation2))
        self.layers.append(Layer(layer3, activation3))
        self.layers.append(Layer(layer4, activation4))
        self.layers.append(Layer(layer5, activation5))

        #Down Block
    




        self.built = True


    def debbugFoward (self):

        batch_size = 5
        img_size = 160
        X = torch.randn((batch_size, self.channels_input, img_size, img_size))

        if (self.built == False):
            self.build()

        pred = self.forward(X)
        print("Input Shape: ", X.shape)
        print("Predict Shape: ", pred.shape)
        
        condition = (batch_size == pred.shape[0]) and (pred.shape[1] == self.channel_output)

        assert condition, AssertionError

        return True

    def __summary__(self):
        
        if(self.problem == "segmentation"):
            torchsummary.summary(self, (self.channels_input, 200, 200))
   
    def loadModel (self, path):
        self.load_state_dict(torch.load(path))
        
    def saveModel(self, path):
        torch.save(self.state_dict(), path)
        
    def eval(self):
        pass
