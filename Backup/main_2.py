import glob
import os
import numpy # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib
import matplotlib.pyplot
import cv2
import PIL
import random
import sklearn
import sklearn.model_selection
import collections
import torch
import torchvision
import torchinfo

import tqdm







class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):

        super(UNET, self).__init__()

        self.channels_input = 3
        self.channels_output = 1
        self.name = "unet-batch_norm"

        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                torch.nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torchvision.transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

















# Define the dataset class

class Dataset(torch.utils.data.Dataset):
	
    def __init__(self, data = None, data_directorys = None, image_directory = None, mask_directory = None, transform = None):
        
        # store the image and mask filepaths, and augmentation
        # transforms
        super(Dataset, self).__init__()

        if (data_directorys != None):
            self.data_directorys = data_directorys

        else:

            self.image_directory = image_directory
            self.mask_directory = mask_directory
            self.image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]
            self.mask_paths = [os.path.join(mask_directory, f) for f in os.listdir(mask_directory)]


            def match_files(dir1, dir2):
                files1 = [os.path.basename(f) for f in dir1]
                files2 = [os.path.basename(f) for f in dir2]
                matches = []
                #TODO check if all the files got a pair
                for file in set(files1).intersection(files2):
                    matches.append([os.path.join(dir1[files1.index(file)]), os.path.join(dir2[files2.index(file)])])
                return matches

            self.data_directorys = match_files(self.image_paths, self.mask_paths)
            '''
            self.data_directorys = []
            for i in range(len(self.image_paths)):
                self.data_directorys.append((self.image_paths[0], self.mask_paths[i]))
            '''
            '''
            self.data = []


            for i in range(len(self.image_paths)):
                imagePath = self.image_paths[i]
                maskPath = self.mask_paths[i]

                image = PIL.Image.open(imagePath).convert('RGB')
                mask = PIL.Image.open(maskPath).convert('L')

                # convert the PIL images to numpy arrays
                image = numpy.array(image)
                mask = numpy.array(mask)

                # normalize the image data to [0, 1] range
                #image = image / 255.

                self.data.append((image, mask))
            '''


        
        self.transform = transform

    def __len__(self):
        # return the number of total samples contained in the dataset
        if (self.data_directorys != None):
            return(len(self.data_directorys))
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):


        if (self.data_directorys != None):

            image, mask = self.data_directorys[idx]

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

            #image = PIL.Image.fromarray(numpy.uint8(image)).convert('RGB')
            #mask = PIL.Image.fromarray(numpy.uint8(mask)).convert('L')

            image = numpy.uint8(image)
            mask = numpy.uint8(mask)
            
            # Binarize mask
            threshold = 200
            mask = numpy.where(mask >= threshold, 1, 0)

            sample = (image, mask)

            # Apply transformations
            if (self.transform != None):
            
                sample = self.transform(sample)
            # This ensures the same transformation to both the image and the mask
            # Not like this below
            # image = self.transform(image)
            # mask = self.transform(mask)


            return sample


        else:

            # grab the image path from the current index
            imagePath = self.image_paths[idx]
            # load the image from disk, swap its channels from BGR to RGB,
            # and read the associated mask from disk in grayscale mode
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            image = PIL.Image.fromarray(numpy.uint8(image)).convert('RGB')
            mask = PIL.Image.fromarray(numpy.uint8(mask)).convert('L')

            sample = (image, mask)

        #sample = {'image': image, 'mask': mask}

            # Apply transformations
            if (self.transform != None):
                
                sample = self.transform(sample)
                # This ensures the same transformation to both the image and the mask
                # Not like this below
                # image = self.transform(image)
                # mask = self.transform(mask)
            return sample
        
        # Convert to PyTorch tensors
        
        # The transpose function in PyTorch is used to change
        # the dimension order of a tensor. In the case of an image,
        # the original order of dimensions is (height, width, channels),
        # but in PyTorch, the standard order is (channels, height, width).

        # Therefore, to convert an image to a PyTorch tensor, we need to
        # transpose the dimensions using the transpose function. The (2, 0, 1)
        # argument specifies the new order of dimensions. The first dimension,
        # which represents the channels in the image, is moved to the front,
        # and the other two dimensions are shifted back by one place.
        # This results in an image tensor with dimensions (channels, height, width),
        # which is the required format for input to most deep learning models in PyTorch.

        
        '''
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)  # Add channel dimension
        '''
        #(image, mask) = tuple(sample.items())

        




class Model (torch.nn.Module):

    def __init__(self):
        
        super(Model, self).__init__()
        
        self.name = None

        self.channels_input = None
        self.channels_output = None
        self.blocks = torch.nn.ModuleList()
    
    def forward(self, X):
        
        if (self.name == "unet"):
            self.connections = []

            X = self.blocks[0].forward(X)
            self.connections.append(X)
            X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

            X = self.blocks[1].forward(X)
            self.connections.append(X)
            X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

            X = self.blocks[2].forward(X)
            self.connections.append(X)
            X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)

            X = self.blocks[3].forward(X)
            self.connections.append(X)
            X = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)(X)




            X = self.blocks[4].forward(X)
            
            
            if (X.shape != self.connections[-1].shape):
                X = torchvision.transforms.functional.resize(X, size = self.connections[-1].shape[2:])
            X = torch.cat((self.connections[-1], X), dim = 1)

            X = self.blocks[5].forward(X)

            if (X.shape != self.connections[-2].shape):
                X = torchvision.transforms.functional.resize(X, size = self.connections[-2].shape[2:])
            X = torch.cat((self.connections[-2], X), dim = 1)

            X = self.blocks[6].forward(X)
            
            if (X.shape != self.connections[-3].shape):
                X = torchvision.transforms.functional.resize(X, size = self.connections[-3].shape[2:])
            X = torch.cat((self.connections[-3], X), dim = 1)

            X = self.blocks[7].forward(X)

            if (X.shape != self.connections[-4].shape):
                X = torchvision.transforms.functional.resize(X, size = self.connections[-4].shape[2:])
            X = torch.cat((self.connections[-4], X), dim = 1)
            
            
            return self.blocks[8].forward(X)
    


class DoubleConvolutionBlock (torch.nn.Module):

    def __init__(self, channels_input:int, channels_output:int, mode, final_output = 1) -> None:
        
        super(DoubleConvolutionBlock, self).__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        if (mode.lower() == "final"):
            self.final_output = final_output

        self.layers = torch.nn.ParameterList()

        self.build(mode)

    
    def build (self, mode):

        
        self.layers.append(torch.nn.Conv2d(in_channels = self.channels_input, 
                                    out_channels = self.channels_output, kernel_size = (3,3),
                                    stride = 1, padding = 1, dilation = 1, bias = True,
                                    padding_mode = "zeros", device=None))

        self.layers.append(torch.nn.ReLU())



        self.layers.append(torch.nn.Conv2d(in_channels = self.channels_output, 
                                    out_channels = self.channels_output, kernel_size = (3,3),
                                    stride = 1, padding = 1, dilation = 1, bias = True,
                                    padding_mode = "zeros", device=None))
        self.layers.append(torch.nn.ReLU())


        if (mode == "Down"):
            #layer3 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding=0, dilation=1)
            #activation3 = None
            #self.layers.append(Layer(self.layer1, self.activation1))
            #self.layers.append(Layer(self.layer2, self.activation2))
            pass

        elif (mode == "Up"):

            self.layers.append(torch.nn.ConvTranspose2d(self.channels_output, (self.channels_output)//2,
                                kernel_size = (2, 2), stride=2, padding=0, dilation=1))
            #self.activation3 = None
            
            #self.layers.append(Layer(self.layer1, self.activation1))
            #self.layers.append(Layer(self.layer2, self.activation2))
            #self.layers.append(Layer(self.layer3, self.activation3))

        elif (mode == "Final"):

            self.layers.append(torch.nn.Conv2d(in_channels = self.channels_output, out_channels = self.final_output,
                                kernel_size = (1,1),stride = 1, padding = 0, dilation = 1, bias = True,
                                padding_mode = "zeros", device=None))          
            #self.activation3 = None

            #self.layers.append(Layer(self.layer1, self.activation1))
            #self.layers.append(Layer(self.layer2, self.activation2))
            #self.layers.append(Layer(self.layer3, self.activation3)) 

    def forward (self, X):

        for layer in self.layers:

            X = layer(X)
        
        return X

class Layer (torch.nn.Module):

    def __init__(self, layer, activationFunction) -> None:
        super(Layer, self).__init__()
    
        self.layer = layer
        self.activationFunction = activationFunction

    def forward (self, X):

        if self.activationFunction is not None:
            return self.activationFunction(self.layer(X))
        
        return self.layer(X)






class Function_Loss(torch.nn.Module):
    def __init__(self, type, problem, problem_type, learning_rate = 0.001, mometum = 0.9):
        super(Function_Loss, self).__init__()

        #standard values
        self.problem = "segmentation"
        self.problem_type = "supervised"



    def forward(self, y_pred, y_true):
        loss = torch.abs(y_pred - y_true)
        avg_loss = torch.mean(loss)
        std_loss = torch.std(loss)
        return avg_loss, std_loss

    def __parameter_checker(self, type, problem, problem_type):


         if(problem != None):
            
            problem = problem.lower()

            #regression function loss
            if (problem == "mse"):
                pass
            elif(problem == "rmse"):
                pass
            elif():
                pass
            elif():
                pass
            elif():
                pass
            # segmentation function loss
            elif("dice_loss"):
                pass
            elif():
                pass
            elif():
                pass
            elif():
                pass
            elif("binary_cross_entropy_with_dice_loss"):
                pass
            elif():
                pass
            elif():
                pass
            elif():
                pass
            elif():
                pass
            elif():
                pass
                
                self.problem = problem
            else:
                print("ERROR -> Function Loss Paramater Checker: Problem")



        if(problem != None):
            
            problem = problem.lower()
            if (problem == "regression" or problem == "classification" or
                problem == "detection" or problem == "segmentation" or
                problem == "anomaly" or problem == "mix"):
            
                self.problem = problem
            else:
                print("ERROR -> Function Loss Paramater Checker: Problem")
        
                
        if(problem_type != None):
            
            problem_type = problem_type.lower()
            if (problem_type == "supervised" or problem_type == "unsupervised" or
                problem_type == "reinforcement" or problem_type == "semi-supervised" or
                problem_type == "transfer" or problem_type == "active" or 
                problem_type == "generative" or problem_type == "recommendation"):
            
                self.problem_type = problem_type
            else:
                print("ERROR -> Function Loss Paramater Checker: Problem Type")

   







class Iteration():
        
    
    def __init__(self, problem, problem_type, debugMode = True,
                 data_type = None,
                 transform = None, seed = None):
        
    
        #standard parameters
        self.data = None
        self.data_directory = None
        self.data_type = None
        
        self.data_train = None
        self.data_test = None

        self.model = Model()
        self.model_built = False
        self.model_trained = False

        self.problem_type = None
        self.problem = None

        self.setup_mark = False
        self.statistics_train = []
        self.batch_train = []

        self.debugMode = True
        self.device = "cpu"

        self.__parameter_checker(problem = problem, problem_type = problem_type, data_type = data_type) 
        
    
      
    #Setup Function
    #########################################################
    def setup(self, data = None, data_test = None, test_factor = 0.2, model = None,
                channels_input = 3, channels_output = 3, device = "cpu"):
        
        if (self.debugMode == True):
            print("Setup: Initialization")

        self.__parameter_checker(data = data, model = model, test_factor = test_factor,
                                channels_input = channels_input, channels_output = channels_output,
                                device = device)
        
        self.__data_setup(data, test_factor)
        self.__model_setup(model)
    
    
        self.setup_saveConfig()


        self.setup_mark = True

        if (self.debugMode == True):
            print("Setup: Complete")
    


    #Analisys Section
    def __parameter_checker(self, data = None, model = None, data_type = None, test_factor = None,
                          channels_input = None, channels_output = None,
                          problem = None, problem_type = None, device = None, debugMode = None):
        
        if (data != None):
            pass
        
        if (model != None):
            
            model = model.lower()
            if (model == "unet"):
                self.model.name = model
            if (model == "unet-batch_norm"):
                self.model.name = model
            else:
                print("ERROR -> Paramater Checker: Model")
        
        if (channels_input != None):
            if (isinstance(channels_input, int) == True):
                self.model.channels_input = channels_input
            else:
                print("ERROR -> Paramater Checker: Channels Input")

        if (channels_output != None):
            if (isinstance(channels_output, int) == True):
                self.model.channels_output = channels_output
            else:
                print("ERROR -> Paramater Checker: Channels Output")


        if (test_factor != None):
            self.test_factor = test_factor
            if (test_factor > 1.0 or test_factor < 0.0):
                print("ERROR -> Paramater Checker: Model")


        if (data_type != None):
        
            data_type.lower()
            if (data_type == "alphanumeric" or data_type == "image" or
                data_type == "audio" or data_type == "video" or
                data_type == "mix"):
        
                self.data_type = data_type
            else:
                print("ERROR -> Paramater Checker: Data Type")
        
        if(problem != None):
            
            problem = problem.lower()
            if (problem == "regression" or problem == "classification" or
                problem == "detection" or problem == "segmentation" or
                problem == "anomaly" or problem == "mix"):
            
                self.problem = problem
            else:
                print("ERROR -> Paramater Checker: Problem")
        
        
        
        
        if(problem_type != None):
            
            problem_type = problem_type.lower()
            if (problem_type == "supervised" or problem_type == "unsupervised" or
                problem_type == "reinforcement" or problem_type == "semi-supervised" or
                problem_type == "transfer" or problem_type == "active" or 
                problem_type == "generative" or problem_type == "recommendation"):
            
                self.problem_type = problem_type
            else:
                print("ERROR -> Paramater Checker: Problem Type")
                
                
        if (device != None):
            
            if (device == "gpu" and torch.cuda.is_available()):
                self.device = torch.device("cuda:0")
            elif(device == "cpu"):
                self.device = torch.device("cpu")
            else:
                print("ERROR -> Paramater Checker: Device")


        if (debugMode != None):
            if(debugMode == True or debugMode == False):
                self.debugMode = debugMode
      

    def analysis_pre(self, image_size = 256, plot_image_resolution = False):

        if (self.debugMode == True):
            print("Analysis Mode: Initialization")


        if(self.setup_mark == False):
            print("ERROR -> Analysis Pre: Setup Mark")
            return
        self.__analysis_data(plot_image_resolution = plot_image_resolution)
        self.__analysis_model(image_size = image_size)


        if (self.debugMode == True):
            print("Analysis Mode: Complete")


    #Data Setup Section
    #########################################################
    def __data_setup(self, data, test_factor, image_size = 256):
        
        self.__data_augmentation(image_size)
        self.__data_generate(data)
        self.__data_preprocessing()
        self.__data_train_test_split(test_factor)
        
    
    def __data_generate(self, data):
        
        if (self.debugMode == True):
            print("-> Data Generation: Initialization")

        start_time = time.process_time()

        if os.path.isdir(data):
             
            self.data_directory = data

            if(self.debugMode == True):
                print("--> {0} is a Directory".format(data))
            

            if (self.data_type == "alphanumeric"):
                pass
        
            elif (self.data_type == "image" and self.problem_type == "supervised"):
                
                # Must have image data and class (Y) data
                
                if (self.problem == "classification"):
                    pass
                elif (self.problem == "detection"):
                    pass
                elif (self.problem == "segmentation"):
                    
                    image_directory = data + "/Images"
                    mask_directory = data + "/Masks"

                    if (os.path.isdir(image_directory) == True):
                        if(self.debugMode == True):
                            print("--> {0} is Image Directory".format(image_directory))
                    else:
                        if(self.debugMode == True):
                            print("--> {0} is Not Image Directory".format(image_directory))
                        raise

                    if (os.path.isdir(mask_directory) == True):
                        if (self.debugMode == True):
                            print("--> {0} is Mask Directory".format(mask_directory))
                    else:
                        if(self.debugMode == True):
                            print("--> {0} is Not Mask Directory".format(mask_directory))
                        raise

                    if (self.debugMode == True):
                        print("--> Generation for Supervised, Image, Segmentation")
                    
                    # Create a dataset using the ImageFolder class
                    #self.data =  torchvision.datasets.ImageFolder(data, transform=self.transformation)
                    self.data = Dataset(image_directory = data + "/Images", mask_directory = data + "/Masks", transform = self.transformation)
    
        else:

            if (self.debugMode == True):
                print("--> {0} is not Directory".format(data))

        if (self.debugMode == True):
            print("-> Data Generation: Complete")
        

        end_time = time.process_time()

        self.data_loading_time = end_time - start_time
            
        print("Data Loading Time: {0} minutes, {1:.3f} seconds".format(int(self.data_loading_time/60), self.data_loading_time%60))

    def __data_augmentation(self, image_size):




        class transformation_dict_to_PIL_tuple(object):
            """
            Rescale the image in a sample to a given size.

            Args:
                output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
            """

            def __init__(self):
                pass

            def __call__(self, sample):

                if (isinstance(sample, dict)):
                    image, mask = sample['image'], sample['mask']
                    sample = (image, mask)
                    print("Dict")
                    print(isinstance(sample, dict))
                    print(isinstance(sample, tuple))
                elif (isinstance(sample, tuple)):
                    print("Tuple")
                    print(isinstance(sample, dict))
                    print(isinstance(sample, tuple))

                return sample




        class Resize(object):
            """
            Rescale the image in a sample to a given size.

            Args:
                output_size (tuple or int): Desired output size. If tuple, output is
                    matched to output_size. If int, smaller of image edges is matched
                    to output_size keeping aspect ratio the same.
            """

            def __init__(self, problem, problem_type, output_size):

                assert isinstance(output_size, (int, tuple))
                self.output_size = output_size

                self.problem = problem
                self.problem_type = problem_type

            def __call__(self, sample):


                if (self.problem_type == "supervised"):

                        #Sample X and Y

                        if (self.problem == "segmentation"):

                            image, mask = sample


                            #print("Init: Image Shape {}".format(image.size))
                            #print("Init: Mask Shape {}".format(mask.size))

                            #Format for PIL.Image.Image variable
                            #h, w = image.shape[:2]
                            #w, h = image.size

                            h, w, c = image.shape

                            if isinstance(self.output_size, int):

                                if h > w:
                                    new_h, new_w = self.output_size * h / w, self.output_size
                                else:
                                    new_h, new_w = self.output_size, self.output_size * w / h

                            else:
                                new_h, new_w = self.output_size

                            new_h, new_w = int(new_h), int(new_w)


                            # Convert numpy arrays to PyTorch Tensors
                            image_tensor = torchvision.transforms.ToTensor()(image)
                            mask_tensor = torchvision.transforms.ToTensor()(mask)

                            #image = torchvision.transforms.functional.resize(image, (new_h, new_w))
                            #mask = torchvision.transforms.functional.resize(mask, (new_h, new_w))

                            resize_operation = torchvision.transforms.Resize(self.output_size)

                            resized_image_tensor = resize_operation(image_tensor)
                            resized_mask_tensor = resize_operation(mask_tensor)


                            '''
                            #from skimage import transform
                            #TODO new way
                            #img = transform.resize(image, (new_h, new_w))

                            #img = PIL.Image.fromarray(image)
                            img = image.resize((new_w, new_h))

                            # h and w are swapped for landmarks because for images,
                            # x and y axes are axis 1 and 0 respectively

                            # Convert PIL mask to numpy array
                            mask = numpy.array(mask)
                            mask = mask * [new_w / w, new_h / h]
                            mask = PIL.Image.fromarray(mask)
                            '''


                            #print("Resize: Image Shape {}".format(img.size))
                            #print("Resize: Mask Shape {}".format(mask.size))

                            return (resized_image_tensor, resized_mask_tensor)


        class RandomCrop(object):
            """
            Crop randomly the image in a sample.
            Args:
                output_size (tuple or int): Desired output size. If int, square crop
                is made.
            """

            def __init__(self, problem, problem_type, output_size):

                assert isinstance(output_size, (int, tuple))

                if isinstance(output_size, int):
                    self.output_size = (output_size, output_size)
                else:
                    assert len(output_size) == 2
                    self.output_size = output_size
                
                self.problem = problem
                self.problem_type = problem_type


            def __call__(self, sample):
                

                if (self.problem_type == "supervised"):

                        #Sample X and Y

                        if (self.problem == "segmentation"):                     

                            '''
                            image, mask = sample
                            #Format for PIL.Image.Image variable
                            #h, w = image.shape[:2]
                            w, h = image.size
                            new_h, new_w = self.output_size

                            top = numpy.random.randint(0, h - new_h)
                            left = numpy.random.randint(0, w - new_w)

                            # center crop
                            #top = (new_h - self.output_size) // 2
                            #left = (new_w - self.output_size) // 2
                            if (isinstance(self.output_size, int) == True):
                                image = image.crop((left, top, left + self.output_size, top + self.output_size))
                                mask = mask.crop((left, top, left + self.output_size, top + self.output_size))
                            elif (isinstance(self.output_size, tuple) == True):
                                image = image.crop((left, top, left + self.output_size[0], top + self.output_size[1]))
                                mask = mask.crop((left, top, left + self.output_size[0], top + self.output_size[1]))

                            '''
                            '''
                            image = image[top: top + new_h,
                                        left: left + new_w]

                            mask = mask - [left, top]
                            '''
                            '''
                            '''



                            image, mask = sample

                            
                            operation = torchvision.transforms.RandomCrop(size = self.output_size)
                            image = operation(image)
                            mask = operation (mask)

                            #print("RandomCrop: Image Shape {}".format(image.size))
                            #print("RadomCrop: Mask Shape {}".format(mask.size))
                            return (image, mask)
                else:
                    print("RandomCrop WTF")


        class CenterCrop(object):
            """
            Crop randomly the image in a sample.
            Args:
                output_size (tuple or int): Desired output size. If int, square crop
                is made.
            """

            def __init__(self, problem, problem_type, output_size):

                assert isinstance(output_size, (int, tuple))

                if isinstance(output_size, int):
                    self.output_size = (output_size, output_size)
                else:
                    assert len(output_size) == 2
                    self.output_size = output_size
                
                self.problem = problem
                self.problem_type = problem_type


            def __call__(self, sample):
                

                if (self.problem_type == "supervised"):

                        #Sample X and Y

                        if (self.problem == "segmentation"):                     

                            image, mask = sample
                            #Format for PIL.Image.Image variable
                            #h, w = image.shape[:2]
                            w, h = image.size
                            new_h, new_w = self.output_size

                            # center crop
                            top = (new_h - self.output_size) // 2
                            left = (new_w - self.output_size) // 2
                            image = image.crop((left, top, left + self.output_size, top + self.output_size))
                            mask = mask.crop((left, top, left + self.output_size, top + self.output_size))
        
                            '''
                            image = image[top: top + new_h,
                                        left: left + new_w]

                            mask = mask - [left, top]
                            '''

                            return (image, mask)
                else:
                    print("CenterCrop WTF")




        class RandomRotation(object):
            """
            Rotate the image randomly within a certain range.

            Args:
                degrees (int): Range of degrees to select from.
            """

            def __init__(self, degrees):
                self.degrees = degrees

            def __call__(self, sample):
                #print("RandomRotation")


                if isinstance(sample, tuple):
                    image, mask = sample
                else:
                    print("Not Tuple")
                    raise

                angle = random.uniform(-self.degrees, self.degrees)

                #img = PIL.Image.fromarray(image)
                image = image.rotate(angle)

                mask = numpy.asarray(mask)
                #new_mask.reshape(224, 224, 1)
                mask = mask.reshape(-1, 2)

                # rotate mask
                angle = angle * 3.14159 / 180
                c, s = numpy.cos(angle), numpy.sin(angle)
                rotation_matrix = numpy.array([[c, -s], [s, c]])
                new_mask = numpy.dot(mask, rotation_matrix)

                mask = PIL.Image.fromarray(new_mask.reshape(224, 224))

                print(type(image))
                print(type(mask))


                #print("RandomRotation: Image Shape {}".format(image.size))
                #print("RandomRotation: Mask Shape {}".format(mask.size))

                return (image, mask)


        class HorizontalFlip(object):
            """
            Flip the image and landmarks horizontally.
            """

            def __init__(self, problem, problem_type):

                self.problem = problem
                self.problem_type = problem_type

            def __call__(self, sample):

                if (self.problem_type == "supervised"):

                        #Sample X and Y

                        if (self.problem == "segmentation"):

                            image, mask = sample
                            
                            image = numpy.fliplr(image)
                            #mask = numpy.asarray(mask)

                            image = PIL.Image.fromarray(image)
                            #h, w = image.shape[:2]
                            w, h = image.size



                            mask = numpy.array(mask) # convert to numpy array
                            
                            #mask = w - mask # invert mask horizontally

                            #mask[:, 0] = w - mask[:, 0]
                            mask = mask[:, ::-1]
                            #mask[:, 0] = image.shape[1] - mask[:, 0]

                            mask = PIL.Image.fromarray(mask.reshape(224, 224))
                            #print("HorizontalFlip: Image Shape {}".format(image.size))
                            #print("HorizontalFlip: Mask Shape {}".format(mask.size))

                            return (image, mask)


        class VerticalFlip(object):
            """
            Flip the image and landmarks vertically.
            """
            def __init__(self, problem, problem_type):

                self.problem = problem
                self.problem_type = problem_type

            def __call__(self, sample):

                if (self.problem_type == "supervised"):

                        #Sample X and Y

                        if (self.problem == "segmentation"):

                            image, mask = sample

                            image = numpy.flipud(image)
                            #mask[:, 1] = image.shape[0] - mask[:, 1]



                            
                            print(image.shape)
                            image = PIL.Image.fromarray(image)
                            #h, w = image.shape[:2]
                            w, h = image.size



                            mask = numpy.array(mask) # convert to numpy array
                            #mask = h - mask # invert mask horizontally
                            print(mask.shape)
                            mask[:, 1] = h - mask[:, 1]


                            #mask[:, 0] = w - mask[:, 0]
                            #mask[:, 0] = image.shape[1] - mask[:, 0]

                            mask = PIL.Image.fromarray(mask.reshape(224, 224))
                            print("VerticalFlip: Image Shape {}".format(image.size))
                            print("VerticalFlip: Mask Shape {}".format(mask.size))

                            return (image, mask)



        class ToTensor(object):
            """Convert ndarrays in sample to Tensors."""

            def __init__(self, problem, problem_type):

                self.problem = problem
                self.problem_type = problem_type

            def __call__(self, sample):

                if (self.problem_type == "supervised"):

                    #Sample X and Y

                    if (self.problem == "segmentation"):

                        image, mask = sample
                        if (isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor)):
                            return sample

                        #print("ToTensor: Image Shape {}".format(image.size))
                        #print("ToTensor: Mask Shape {}".format(mask.size))
                        # swap color axis because
                        # numpy image: H x W x C
                        # torch image: C x H x W
                        #image = image.transpose((2, 0, 1))

                        #image = image.transpose((0,1))
                        #image = torch.from_numpy(image)
                        #mask = torch.from_numpy(mask)

                        #print("Final: Image Shape {}".format(image.size))
                        #print("Final: Mask Shape {}".format(mask.size))

                        transform = torchvision.transforms.ToTensor()
                        image = transform(image)
                        mask = transform(mask)
                        #image = image.transpose((2, 0, 1))
                        
                        #print("Final: Image  {}".format(type(image.size)))
                        #print("Final: Mask  {}".format(type(mask.size)))

                        return (image, mask)



        # Define the transforms
        '''
        self.transformation = torchvision.transforms.Compose([Resize(problem = self.problem, problem_type = self.problem_type, output_size = image_size),
                                                            RandomCrop(problem = self.problem, problem_type = self.problem_type, output_size = 224),
                                                            RandomRotation(degrees = 10),
                                                            HorizontalFlip(problem = self.problem, problem_type = self.problem_type),
                                                            VerticalFlip(problem = self.problem, problem_type = self.problem_type),
                                                            ToTensor(problem = self.problem, problem_type = self.problem_type)
                                                            ])
        '''
        
        self.transformation = torchvision.transforms.Compose([Resize(problem = self.problem, problem_type = self.problem_type, output_size = image_size),
                                                            RandomCrop(problem = self.problem, problem_type = self.problem_type, output_size = 224),
                                                            #HorizontalFlip(problem = self.problem, problem_type = self.problem_type),
                                                            ToTensor(problem = self.problem, problem_type = self.problem_type)
                                                            ])

        
        '''
        self.transformation = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                                            torchvision.transforms.CenterCrop(224),
                                                            torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.RandomVerticalFlip(),
                                                            torchvision.transforms.RandomRotation(10),
                                                            torchvision.transforms.ToTensor(),
                                                            
                                                            ])
        '''
        '''
        torchvision.transforms.Normalize(
                                                                mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]
                                                            )
        Output different form input, so normalize diferent shapes para Image e para masks. Nao sei como separar
        '''

    def __data_preprocessing(self):
        pass
       
    
    def __data_train_test_split(self, test_factor):

        if (self.debugMode == True):
            print("--> Train/Test Split: Initialization")

        # Split your dataset into training and testing subsets
        test_size = int(test_factor * self.data.__len__())
        train_size = self.data.__len__() - test_size


        

        #self.data_train, self.data_test = torch.utils.data.random_split(self.data, [train_size, test_size])




        # create the train and test subsets
        '''
        train_subset, test_subset = torch.utils.data.random_split(self.data, [train_size, test_size])


        train_directorys = [self.data.data_directorys[i] for i in train_subset.indices]
        print("train_directorys: {}".format(train_directorys))
        test_directorys = [self.data.data_directorys[i] for i in test_subset.indices]

        self.data_train = Dataset(data_directorys = train_directorys, transform = self.transformation)
        self.data_test = Dataset(data_directorys = test_directorys, transform = self.transformation)
        '''

        def random_sublist(lst, num):
            sublist = random.sample(lst, num)
            remaining = [elem for elem in lst if elem not in sublist]
            return sublist, remaining
        
        train_list, test_list = random_sublist(self.data.data_directorys, train_size)
        #print("train_directorys: {}".format(train_list))
        self.data_train = Dataset(data_directorys = train_list, transform = self.transformation)
        self.data_test = Dataset(data_directorys = test_list, transform = self.transformation)
        
        #self.data_test = Dataset(data_directorys = self.data.data_directorys[i for i in test_subset.indices], transform = self.transformation)


        if (self.debugMode == True):
            print("--> Train/Test Split: Test Factor = {0} Train Size = {1} Test Size = {2}".format(
                test_factor, self.data_train.__len__(), self.data_test.__len__()))

        if (self.debugMode == True):
            print("--> Train/Test Split: Train Data Type = {0} Train Data Type = {1}".format(type(self.data_train), type(self.data_test)))

        '''
        self.data_train.to(self.device)
        self.data_test.to(self.device)

        

        if (self.debugMode == True):
            print("--> Train/Test Split: Train Data Device = {0} Train Data Device = {1}".format(
                self.data_train.get_device(), self.data_test.get_device()))

        '''
        if (self.debugMode == True):
            print("--> Train/Test Split: Complete")


    def __get_custom_sample(self, data, index, original = False):

        if (self.setup_mark != True):
            raise
            return




        image, mask = data.data_directorys[index]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        image = PIL.Image.fromarray(numpy.uint8(image)).convert('RGB')
        mask = PIL.Image.fromarray(numpy.uint8(mask)).convert('L')
            
        sample = (image, mask)

   
        # Apply transformations
        if (data.transform != None and original == False):
            
            sample = data.transform(sample)
        
        return sample


    def __analysis_data(self, n_train_sampler = 2, n_test_sampler = 2, plot_image_resolution = False):
        
        if (plot_image_resolution):

            def plot_image_sizes(dataset, num_samples=1000):
                """Plots a bar plot of the most used image sizes returned by the dataset's __getitem__ method."""
                
                # Collect the image sizes in a dictionary
                sizes = collections.defaultdict(int)
                for i in range(num_samples):
                    img, mask = self.__get_custom_sample(dataset, i, original = True)
                    img = numpy.array(img)
                    sizes[img.shape] += 1
                
                # Sort the dictionary by count and convert it to a list of tuples
                sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
                
                # Extract the size and count from each tuple
                labels, values = zip(*sorted_sizes)
                
                # Plot the bar chart
                matplotlib.pyplot.bar(range(len(labels)), values)
                matplotlib.pyplot.xticks(range(len(labels)), labels)
                matplotlib.pyplot.xlabel("Image Size")
                matplotlib.pyplot.ylabel("Frequency")
                matplotlib.pyplot.title("Image Size Distribution")
                matplotlib.pyplot.show()


            plot_image_sizes(self.data, num_samples = self.data.__len__())

        self.__train_test_analysis(n_train_sampler = n_train_sampler, n_test_sampler = n_test_sampler)
    

    def __train_test_analysis(self, n_train_sampler, n_test_sampler):



        # Train/Test Pie Chart
        fig, axes = matplotlib.pyplot.subplots(1,1, figsize = (4,4))
        labels = "Train Data", "Test Data"
        sizes = [len(self.data_train), (len(self.data_test))]
       
        def labeling_func(size):
            relative = size/(len(self.data_train) + (len(self.data_test)))
            return "{0}\n{1:.2f}%".format(size, relative)
            
        axes.pie(sizes, labels = labels)
        #TODO put sizes absolute value in pie chart


        #wedges, texts, autotexts = axes.pie(sizes, labels=labels, autopct = lambda size: labeling_func(size))
        axes.set_title("Train/Test Chart with Test Factor: {0}".format(self.test_factor))
        #axes.legend(wedges, labels, title = "Data", loc = "best")


        # Train and Test Sampler
        if (self.data_type == "image"):

            if (self.problem_type == "supervised"):

                fig, axes = matplotlib.pyplot.subplots(n_train_sampler, 3, figsize = (3*3,4*n_train_sampler))

                for i in range(0, n_train_sampler, 1):

                    index = random.randint(0, self.data_train.__len__())

                    sample = list(self.data_train.__getitem__(index))

                    # Assume image_tensor is a PyTorch tensor object with shape (C, H, W)
                    # Swap the order of dimensions to be (H, W, C) and convert to a NumPy array
                    sample[0] = sample[0].permute(1, 2, 0).numpy()
                    sample[1] = sample[1].permute(1, 2, 0).numpy()


                    original = list(self.__get_custom_sample(data = self.data_train, index = index, original = True))

                    axes[i][0].imshow(original[0])
                    axes[i][0].set_title("Original Sample: {}".format(index))

                    axes[i][1].imshow(sample[0])
                    axes[i][1].set_title("Train X: {0}\n Size {1}".format(index, sample[0].shape))

                    axes[i][2].imshow(sample[1], cmap = "gray")
                    axes[i][2].set_title("Train Y: {0}\n Size {1}".format(index, sample[1].shape))
                

                fig, axes = matplotlib.pyplot.subplots(n_test_sampler, 3, figsize = (3*3,4*n_train_sampler))

                for i in range(0, n_test_sampler, 1):

                    index = random.randint(0, self.data_test.__len__())
                    sample = list(self.data_test.__getitem__(index))

                    # Assume image_tensor is a PyTorch tensor object with shape (C, H, W)
                    # Swap the order of dimensions to be (H, W, C) and convert to a NumPy array
                    sample[0] = sample[0].permute(1, 2, 0).numpy()
                    sample[1] = sample[1].permute(1, 2, 0).numpy()


                    original = list(self.__get_custom_sample(data = self.data_test, index = index, original = True))

                    axes[i][0].imshow(original[0])
                    axes[i][0].set_title("Original Sample: {}".format(index))

                    axes[i][1].imshow(sample[0])
                    axes[i][1].set_title("Test X: {0}\n Size {1}".format(index, sample[0].shape))

                    axes[i][2].imshow(sample[1], cmap = "gray")
                    axes[i][2].set_title("Test Y: {0}\n Size {1}".format(index, sample[1].shape))






    #Model Setup Section
    #########################################################
    def __model_setup(self, model):
        self.__model_build()
        
        
    def __model_build(self):

        if (self.debugMode == True):
            print("-> Model Build: Initialization")

        if (self.model.name == "unet"):
            self.__model_build_UNet()
        elif (self.model.name == "unet-batch_norm"):
            self.model = UNET(in_channels=3, out_channels=1).to(self.device)
            if (self.debugMode == True):
                print("-> Model Build: U-Net with Batch Normalization")

        #self.model.blocks = torch.tensor(self.model.blocks)
        self.model_built = True
        self.model.to(self.device)
        
        #print("-> Model Device: {}".format(next(self.model.parameters()).device))

        if (self.debugMode == True):
            print("-> Model Build: Complete")
        
    def __model_build_UNet(self):
        
        if (self.debugMode == True):
            print("--> Build U-Net: Initialization")
        
        # Encoder Chain
        # The contracting path follows the typical architecture of a convolutional network.
        # It consists of the repeated application of two 3×3 convolutions
        # (unpadded convolutions), each followed by a rectified linear unit (ReLU)
        # and a 2×2 max pooling operation with stride 2 for downsampling.
        # At each downsampling step we double the number of feature channels.

        self.model.blocks.append(DoubleConvolutionBlock(self.model.channels_input, 64, "Down"))
        self.model.blocks.append(DoubleConvolutionBlock(64, 128, "Down"))
        self.model.blocks.append(DoubleConvolutionBlock(128, 256, "Down"))
        self.model.blocks.append(DoubleConvolutionBlock(256, 512, "Down"))

    
        # Decoder Chain
        # Every step in the expansive path consists of an upsampling of the feature map
        # followed by a 2×2 convolution (“up-convolution”) that halves the number
        # of feature channels, a concatenation with the correspondingly cropped feature map
        # from the contracting path, and two 3×3 convolutions, each followed by a ReLU

        self.model.blocks.append(DoubleConvolutionBlock(512, 1024, "Up"))
        
        self.model.blocks.append(DoubleConvolutionBlock(1024, 512, "Up"))
        self.model.blocks.append(DoubleConvolutionBlock(512, 256, "Up"))
        self.model.blocks.append(DoubleConvolutionBlock(256, 128, "Up"))

        self.model.blocks.append(DoubleConvolutionBlock(128, 64, "Final", self.model.channels_output))


        self.model.name = "unet"
        '''
        for i in range(0, len(self.model.blocks), 1):
            self.model.blocks[i] = torch.nn.Parameter(self.blocks[i])
        '''
        if (self.debugMode == True):
            print("--> Build U-Net: Complete")

    def __model_forward(self, X):
    
        return self.model.forward(X)


    def __model_debug(self, image_size):
        

        batch_size = 5

        X = torch.randn((batch_size, self.model.channels_input, image_size, image_size), device=self.device)

        #X.to(self.device)
        print("--> Debug: Input Device {}".format(X.device))
        print("--> Debug: Model Device {}".format(next(self.model.parameters()).device))
        if (self.model_built == False):
            print("ERROR -> Debug: Model Not Built Yet")

        pred = self.__model_forward(X)
        print("--> Debug: Input Shape ", X.shape)
        print("--> Debug: Predict Shape ", pred.shape)
        
        condition = (batch_size == pred.shape[0]) and (pred.shape[1] == self.model.channels_output)

        assert condition, AssertionError
        print("--> Debug: Prediction OK")
        

    def __analysis_model(self, image_size):

        if(self.debugMode == True):
            print("-> Analysis Model: Initialization")


        # Check if the model has trainable parameters
        params = list(self.model.parameters())

        if(self.debugMode == True):
            print("--> Model Trainable Parameters: {0}".format(len(params)))

        # Model Layers   
        if(self.problem == "segmentation"):
            #torchsummary.summary(self, (self.channels_input, 200, 200))
            pass
    
        #print(params)

        batch_size = 32
        print(torchinfo.summary(self.model, input_size=(1, self.model.channels_input, image_size, image_size), depth = 20))



        
        print("For Input Image Size: {0}".format(image_size))
        def compute_max_depth(shape, max_depth=20, print_out=True):
            
            shapes = []
            shapes.append(shape)

            for level in range(1, max_depth):

                if shape % 2 ** level == 0 and shape / 2 ** level > 1:
                    shapes.append(shape / 2 ** level)
                    if print_out:
                        print(f'Level {level}: {shape / 2 ** level}')
                else:
                    if print_out:
                        print(f'Max-level: {level - 1}')
                    break

            return shapes

        print(compute_max_depth(image_size, print_out=True))




        self.__model_debug(image_size = image_size)


        if(self.debugMode == True):
            print("-> Analysis Model: Complete")

    def __save_model(self):
        torch.save(self.state_dict(), path)
    
    def __load_model(self):
        pass


    #Train Model Section
    #########################################################
    def train(self, batch_size = 32, n_epochs = 5, KFolds = 5,
            function_loss = None, optimizer = None, multiple_processes = 2,
            best_batch_show = True):
        

        self.batch_size = batch_size
        self.function_loss = function_loss
        self.optimizer = optimizer
        
        print("Model Training: Initialization")
        
        train_start_time = time.process_time()

        # Create a KFold object with the desired number of folds
        kFold = sklearn.model_selection.KFold(n_splits = KFolds, shuffle=True)


        scaler = torch.cuda.amp.GradScaler()

        # Iterate over the folds
        for fold, (train_idx, valid_idx) in enumerate(kFold.split(self.data_train)):
    
            fold_loss = 999.99
            fold_acc = 0.00

            print("Fold {0}/{1}".format(fold+1, KFolds))

            # Create the data loaders for the fold using the SubsetRandomSampler
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    
            train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=batch_size, sampler=train_sampler)
            valid_loader = torch.utils.data.DataLoader(self.data_train, batch_size=batch_size, sampler=valid_sampler)

            for batch in train_loader:
                #print(batch)
                break
            # Train the model using the data from this fold
            #self.model.forward(train_loader)


            for epoch in range (0, n_epochs, 1):
 
                time_epoch, epoch_loss, epoch_acc = self.__train_epoch(fold, epoch, train_loader, scaler = scaler)
                self.statistics_train.append([fold+1, epoch+1, epoch_loss, epoch_acc])

                fold_loss += epoch_loss
                fold_acc += epoch_acc

                print("---> Epoch {0} of {1} | Time: -> {2}:{3}:{4:.2} Loss -> {2} Acc -> {3}".format(
                    epoch + 1, n_epochs, int(time_epoch/(24*60)), int(time_epoch/60), time_epoch%60, epoch_loss, epoch_acc))
            
             # Adjust metrics to get average loss and accuracy per batch 
            fold_loss_avg = fold_loss / len(train_loader)
            fold_acc_avg = fold_acc / len(train_loader)
            
            print("Average Fold {0} Loss: {1} Acc: {2}}]".format(fold + 1, loss.item(), acc))
        

        train_finish_time =  time.process_time() - train_start_time

        self.model_trained = True

    def __train_batch(self, data):



        
        def intersection_over_union(y_pred, y_true):
            y_pred = y_pred > 0.5  # binarize predictions
            y_true = y_true > 0.5  # binarize true masks
            y_pred = y_pred.to(torch.long)  # convert to long data type
            y_true = y_true.to(torch.long)  # convert to long data type
            intersection = torch.logical_and(y_pred, y_true).sum(dim=[2, 3])  # sum over height and width dimensions
            union = torch.logical_or(y_pred, y_true).sum(dim=[2, 3])  # sum over height and width dimensions
            iou = intersection.float() / union.float()
            return iou.mean(dim=[0, 1])  # average over batch and channel dimensions



        def accuracy (y_pred, label):
            acc = (y_pred == label).sum().item()/len(y_pred)
            return acc
                
        if (self.problem_type == "supervised" and
            self.problem == "classification"):
            
        
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        
            # track history if only in train
            with torch.set_grad_enabled(True):
                
                # 1. Forward pass
                # forward + backward + optimize
                '''
                if(self.model.baseModel is None):
                    y_pred = self.model.forward(inputs)
                else:
                    y_pred = self.model.baseModel(inputs)
                '''
                y_pred = self.__model_forward(inputs)
                                
                # 2. Calculate  and accumulate loss
                loss = self.function_loss(y_pred, labels)
                
                # 3. Optimizer zero grad
                self.optimizer.zero_grad()
                
                # 4. Loss backward
                loss.backward()
                
                # 5. Optimizer step
                self.optimizer.step()
            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            
            #y_pred_class = torch.argmax(y_pred, dim=1)
            acc = (y_pred_class == labels).sum().item()/len(y_pred)
        
            print("Train Batch: Loss: {0} Acc: {1}}]".format(loss.item(), acc))
            return loss.item(), acc
        
        
        if (self.problem_type == "supervised" and
            self.problem == "segmentation"):
            
            # get the inputs; data is a list of [inputs, labels]
            #data_tqdm = tqdm.tqdm(data)
            #inputs, masks = data_tqdm
            inputs, masks, *other = data

            #masks = (masks > 1).float()
            #print("Inputs: {}".format(len(inputs)))
            #print(inputs[0])
            #print(inputs[1])
            #print(inputs[2])
            #print("masks: {}".format(len(masks)))
            #print(masks[0])
            #inputs = inputs.to(self.device)
            #masks = masks.to(self.device)
            print("Inputs Type {0} Mask Type {1}".format(type(inputs), type(masks)))
            print("Inputs Shape {0} Mask Shape {1}".format(inputs.size(), masks.size()))
            print("Inputs DEVICE {0} Mask Device {1}".format(inputs.device, masks.device))
            
            
            #print("inputs: {0}".format(inputs.shape))
            #print("masks: {0}".format(masks.shape))
            #print("mask 1: {0}".format(masks[0]))
          
        
            # track history if only in train
            with torch.set_grad_enabled(True):
                
                # 1. Forward pass
                # forward + backward + optimize
                '''
                if(self.model.baseModel is None):
                    y_pred = self.model.forward(inputs)
                else:
                    y_pred = self.model.baseModel(inputs)
                '''
                y_pred = self.__model_forward(inputs)
                #print("Y_pred: {0}".format(y_pred.shape))
                #print("Y_pred: {0}".format(y_pred))
                
                # 2. Calculate  and accumulate loss
                loss = self.function_loss(y_pred, masks)
                
                # 3. Optimizer zero grad
                self.optimizer.zero_grad()
                
                # 4. Loss backward
                loss.backward()
                
                # 5. Optimizer step
                self.optimizer.step()
            
            #y_pred_class = torch.argmax(y_pred, dim=1)
            iou = intersection_over_union(y_pred, masks)
            return loss.item(), iou


    def __train_batch_mixed_precision(self, data, scaler):

        # forward
        with torch.cuda.amp.autocast():

            inputs, masks, *other = data

            y_pred = self.__model_forward(inputs)
            #print("Batch: y_pred {0}".format(type(y_pred)))
            #print("Batch: y_pred len {0}".format(len(y_pred)))
            #print("Batch: masks {0}".format(type(y_pred)))
            #print("Batch: masks len {0}".format(len(masks)))
            loss = self.function_loss(y_pred.float(), masks.float())
            
            acc = self.__check_accuracy(mode = "accuracy", y_pred = y_pred, y_true = masks)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        return loss.item(), acc



                    
    def __train_epoch(self, kFold, epoch, data, scaler):

        epoch_start_time = time.process_time()
        epoch_loss = 0
        epoch_acc = 0
        
        data_tqdm = tqdm.tqdm(data) #Nice Progress Bar
        data_number_total = 0

        for batch_i, data_batch in enumerate(data_tqdm, 1):

            #print("Batch Train")
            #print(len(data_batch))
            #print(type(data_batch[0]))
            #print(data_batch[0])
            
            #data_batch = torch.tensor(data_batch)
            #data_batch.to(self.device)


            data_number_total += self.batch_size

            data_batch = [d.to(self.device) for d in data_batch]
            loss, acc = self.__train_batch_mixed_precision(data_batch, scaler = scaler)

            # update tqdm loop
            data_tqdm.set_postfix(Batch = batch_i, Loss = loss, Accuracy = acc)

            self.statistics_batch_train.append([data_number_total, loss, acc])
            #print("Batch: {0} Loss: {1} Acc: {2}".format(batch_i, loss, acc))
            epoch_loss += loss
            epoch_acc += acc

        # Adjust metrics to get average loss and accuracy per batch 
        epoch_loss_avg = epoch_loss / len(data)
        epoch_acc_avg = epoch_acc / len(data)


        
        return time.process_time() - epoch_start_time, epoch_loss_avg, epoch_acc_avg



    def __check_accuracy(self, mode, y_pred = None, y_true = None):

        if(self.problem_type == "supervised"):

            # Must have y_pred and y_true

            if (self.problem == "segmentation" and self.data_type == "image"):

                # y are masks

                if(mode == "accuracy"):
                    '''
                    Computes the accuracy between two binary masks
                    '''
                    # flatten the tensors
                    y_pred = y_pred.view(-1)
                    y_true = y_true.view(-1)

                    y_pred = torch.round(torch.sigmoid(y_pred)).long()
                    y_true = y_true.long()


                    # calculate the number of correct predictions
                    correct = torch.sum(y_pred == y_true).item()

                    # calculate the total number of predictions
                    total = y_true.shape[0]

                    # calculate the accuracy
                    accuracy = float(correct/ total)

                
                    return accuracy
                else:
                    print("X ERROR: Check Accuracy")

            else:
                print("X ERROR: Check Accuracy")

        else:
            print("X ERROR: Check Accuracy")


    def eval_model(self, n_predictions = 5, new_data = None, debug_path = None):


        if (self.debugMode == True):

            print("Model Evaluation: {0}".format(self.mode.name))

        #make n_predictions and show images, mask and predictions

        if (self.data_type == "image"):

            if (self.problem == "segmentation"):

                fig, axes = matplotlib.pyplot.subplots(n_predictions, 3, figsize = (3*3, 3*n_predictions))

                for i in range(0, n_predictions, 1):

                    image, mask = self.data_test.__getitem__(random.randint(0, self.data_test.__len__()))

                    axes[i][0].imshow(image)
                    axes[i][0].set_title("Input Image: {0}\n Size {1}".format(index, image.shape))

                    axes[i][1].imshow(mask)
                    axes[i][0].set_title("True Mask: {0}\n Size {1}".format(index, mask.shape))

                    predict = self.__model_forward(image)

                    # assume `predict` is a PyTorch tensor containing an image
                    predict = predict.permute(1, 2, 0).cpu().numpy() # convert to numpy array
                    axes[i][1].imshow(predict)
                    axes[i][0].set_title("Model Prediction\n Size {0}".format(predict.shape))


        #plot learning curves
        # score by sample
        batch_data = numpy.array(self.statistics_batch_train)

        X = batch_data[:,0]
        loss_train = batch_data[:,1]
        acc_train = batch_data[:,2]
        
        # score by epoch

        # score by fold


        # if regression
        # R2 score
        

        # if (classifier)
        # Confusion Matrix training data
        # cnfusion matrix test data
        # COnsequentemnte, precioson, recal e f1 score para trianing e test
        # ROC CUrver
        # Mean Average Precision (mAP): This metric is commonly used to evaluate object detection models, but can also be used for image segmentation problems. It calculates the precision and recall values for different IoU thresholds, and then calculates the average precision across all IoU thresholds.


        # if segmentation
        # pixel accuracy
        # IoU
        # Dice



        # bar plot de outras métricas
        # if regresion
        #   MSE, RMSE, etc
        # TODO em train obter multiplas metricas 
   


        pass


#                   , savePath = None, debbugMode = False,
#                    debbugFile_Path = None):
    
    def pil_to_tensor(image):
        """
        Converts a PIL Image to a PyTorch tensor.
        Args:
            image (PIL Image): Image to be converted to tensor.
        Returns:
            tensor (Tensor): Converted PyTorch tensor.
        """
        tensor = to_tensor(image)
        return tensor


    def tensor_to_pil(tensor):
        """
        Converts a PyTorch tensor to a PIL Image.
        Args:
            tensor (Tensor): Tensor to be converted to PIL Image.
        Returns:
            image (PIL Image): Converted PIL Image.
        """
        image = to_pil_image(tensor)
        return image


    def setup_saveConfig(self):
        pass





    








    