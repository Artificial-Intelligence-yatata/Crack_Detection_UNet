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



class customDataset(torch.utils.data.Dataset):
    
    def __init__(self, data = None, data_test = None, for_training = True,
                 data_type = None, debugMode = False,
                 transform = None, seed = None):
        
        
        self.problem = "segmentation"
        self.problem_type = "supervised"
        
                
        self.data_loaded = False
        self.debugMode = debugMode
        
        self.parameter_check(data = data, data_test = data_test,
                             for_training = for_training,
                             data_type = data_type, seed = seed)
        
        
        self.multiple_processes = 1
        
        try:
            self.transform = transform
        except NameError:
            self.transform = None
        
      
    def parameter_check(self, data, data_test, for_training, data_type, seed):

        if (self.debugMode == True):
            print("Dataset Parameter Check Initialization")

        data_type.lower()
        if (data_type != "alphanumeric" or data_type != "image" or
            data_type != "audio" or data_type != "video" or
            data_type != "mix"):
            
            
            #TODO raise error 
            pass
        
        self.data_type = data_type



        self.data_test = data_test

        
        
        self.for_training = for_training
        
        
        
        if (data != None):
            self.generate_dataset(data, self.data_type)


        if (self.debugMode == True):
            print("Dataset Parameter Check Complete") 
            
    def initiate_process(self, data_test, test_factor, batch_size, shuffle, drop_last, seed):
        
        self.test_factor = test_factor


        if(self.debugMode == True):
            print("Dataset Initial Process Initialization")

        if (data_test is None):
            self.train_test_split(test_factor = test_factor, seed = seed)
        
        else:
            self.data_train = self.data
            self.data_test = data_test
            
            self.train_size = len(self.data_train)
            self.test_size = len(self.data_test)
            self.test_factor = len(self.data_test)/len(self.data)



        self.preprocessing()
        
        self.init_dataloader(for_training = self.for_training, batch_size = batch_size, multiple_processes = self.multiple_processes,
                             shuffle = shuffle, drop_last = drop_last)
        
        
        if(self.debugMode == True):
            print("Dataset Initial Process Complete")

    def generate_dataset (self, data, data_type):
        

        classes = []
        image_paths = []
        mask_paths = []



        if (self.debugMode == True):
          print("Dataset Generation Initialization")

        """
        Data can be a dataset, data_directory, a Database, 
        """
        start_time = time.process_time()
        
        if os.path.isdir(data):
             
            if(self.debugMode == True):
               print("{0} is a Directory".format(data))
            
            if (data_type == "alphanumeric"):
                pass
        
            elif (data_type == "image" and self.problem_type == "supervised"):
                
                # Must have image data and class (Y) data
                
                if (self.debugMode == True):
                    print("Generation for Supervised, Image")
                
                if (self.problem == "classification"):
                    pass
                elif (self.problem == "detection"):
                    pass
                elif (self.problem == "segmentation"):
                    
                    # must have Binary classes (Mask of Yes Data and Mask of No Data)
                    # define directory paths


                    if (self.debugMode == True):
                        print("Generation for Segmentation")

                    data_directory = data
                    
                    classes = list(map(os.path.basename,[f.path for f in os.scandir(data_directory) if f.is_dir()]))
    
                    image_folders = ["images", "imgs", "image", "Images", "Imgs", "Image"]
                    mask_folders = ["masks", "mks", "mask", "Masks", "Mks", "Mask"]
            
                     # initialize empty lists for image and mask paths
                    image_paths = []
                    mask_paths = []
                    
                    # iterate through image directories and add image paths to list
                    for classe in classes:
                        classe_directory = os.path.join(data_directory, classe)
                        
                        for image_folder in image_folders:
                            path = os.path.join(classe_directory, image_folder)
                            
                            if os.path.exists(path):
                                                               
                                image_paths.extend(glob.glob(path + '/*.[jJ][pP][gG]', recursive=False))
                                image_paths.extend(glob.glob(path + '/*.[jJ][pP][eE][gG]', recursive=False))
                                image_paths.extend(glob.glob(path +'/*.png', recursive=False))
                                image_paths.extend(glob.glob(path + '/*.webp', recursive=False))

                        # iterate through mask directories and add mask paths to list
                        for mask_folder in mask_folders:
                            path = os.path.join(classe_directory, mask_folder)
                            if os.path.exists(path):
                                mask_paths.extend(glob.glob(path + '/*.[jJ][pP][gG]', recursive=False))
                                mask_paths.extend(glob.glob(path + '/*.[jJ][pP][eE][gG]', recursive=False))
                                mask_paths.extend(glob.glob(path +'/*.png', recursive=False))
                                mask_paths.extend(glob.glob(path + '/*.webp', recursive=False))

                    # create dataframe from image and mask paths
                    data = {'X': image_paths, 'Y': mask_paths}

                    if (self.debugMode == True):
                        print("Classes: {0}".format(classes))
                        print("Images Number: {0}".format(len(image_paths)))
                        print("Masks Number: {0}".format(len(mask_paths)))


                    try:
                      dataset = pandas.DataFrame(data)
                    except:
                      print("Erro Generating Dataset")                 

                    fig, axes = matplotlib.pyplot.subplots(1,1, figsize=(5,5))
                    axes.bar(["X", "Y"], [dataset["X"].count(), dataset["Y"].count()])

                    axes.set_title("Sample Number")
                    axes.set_xlabel("X and Y Data")
                    axes.set_ylabel("Sample Count")
                    matplotlib.pyplot.show()

                elif (self.problem == "mix"):
                    pass
        
            elif (data_type == "audio"):
                pass
            elif (data_type == "video"):
                pass
            elif (data_type == "mix"):
                pass
            
            
            self.data = dataset
            
            end_time = time.process_time()

            self.data_loading_time = end_time - start_time
            
            print("Data Loading Time: {0} minutes, {1:.3f} seconds".format(int(self.data_loading_time/60), self.data_loading_time%60))
            
        else:

            if (self.debugMode == True):
                print("{0} is not Directory".format(data))
            
            # TODO check if dataset
            self.data = data
            #TODO check if connection
        
        self.data_loaded = True
        
        if (self.debugMode == True):
          print("D  Dataset Generation Complete")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        if (self.data_type == "image" and self.problem == "segmentation" and
            self.problem_type == "supervised"):
            
            X = self.data.loc[index]["X"]
            Y = self.data.loc[index]["Y"]
            
            X = numpy.array(Image.open(X_directory).convert("RGB"))
            Y = numpy.array(Image.open(Y_directory).convert("L"), dtype=np.float32)
            Y[Y == 255.0] = 1.0
            
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        

        

        return X, Y
    
    
    def get_random_sample (self, sample_number = 2, img_show = False, img_transform = False):
        
        if (self.data_type == "image" and self.problem == "segmentation" and
            self.problem_type == "supervised"):
            
            
            samples_index = numpy.random.randint(self.__len__(), size = sample_number)
            
            print(samples_index)
            
            X_directory = self.data.loc[samples_index]["X"]
            Y_directory = self.data.loc[samples_index]["Y"]
        
        
            if (img_transform == True):
                pass
        
            if (img_show == True):
                
                fig, axes = matplotlib.pyplot.subplots(nrows=sample_number, ncols=2, figsize = (8,5*sample_number))
                
                
                for i in range(0, len(samples_index), 1):
                    
                    
                    X_img = cv2.imread(X_directory[samples_index[i]])
                    Y_img = cv2.imread(Y_directory[samples_index[i]])
                
                    axes[i][0].imshow(X_img)
                    axes[i][1].imshow(Y_img)
                    
                    axes[i][0].set_title(X_directory[samples_index[i]], fontsize=8)
                    axes[i][1].set_title(Y_directory[samples_index[i]], fontsize=8)
                
                matplotlib.pyplot.show()
        
        else:
            pass
        
        return X_directory, Y_directory
    
    def analize_sample (self, index = None, sample = None, with_model = False):
        
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        laplacian = numpy.array(cv2.Laplacian(image, cv2.CV_64F))
        
    def evaluate_data (self):
        
        start_time = time.process_time()
        
        if (self.data_loaded == False):
            #TODO raise error
            return
        
        if (self.data_type == "image"):
            
            keys = ["height", "width", "channels", "quality"]
            keys_special = ["histogram", ""]
            
            
            def img_info(image_path):
                
                image = cv2.imread(image_path)
                height, width, channels = image.shape
                
                hist_blue = cv2.calcHist([image], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([image], [1], None, [256], [0, 256])
                hist_red = cv2.calcHist([image], [2], None, [256], [0, 256])
                quality = 1
                
                data = {
                    "height":height,
                    "width":width,
                    "channels":channels,
                    "histogram":[hist_red, hist_green, hist_blue],
                    "quality":quality
                }
                return data
            
            
            if (self.problem_type == "supervised"):
                
                X_info = numpy.array(self.data["X"].apply(img_info))
                Y_info = numpy.array(self.data["Y"].apply(img_info))
            
                X_features = []
                Y_features = []
            
                for key in keys:


                    X_features.append([len([data.get(key) for data in X_info]),
                                       numpy.max([data.get(key) for data in X_info]),
                                       numpy.min([data.get(key) for data in X_info]),
                                       numpy.mean([data.get(key) for data in X_info]),
                                       numpy.median([data.get(key) for data in X_info]),
                                       numpy.std([data.get(key) for data in X_info]),
                                       numpy.unique([data.get(key) for data in X_info])])


                    Y_features.append([len([data.get(key) for data in Y_info]),
                                       numpy.max([data.get(key) for data in Y_info]),
                                       numpy.min([data.get(key) for data in Y_info]),
                                       numpy.mean([data.get(key) for data in Y_info]),
                                       numpy.median([data.get(key) for data in Y_info]),
                                       numpy.std([data.get(key) for data in Y_info]),
                                       numpy.unique([data.get(key) for data in Y_info])])



                fig, axes = matplotlib.pyplot.subplots(2,2, figsize = (10,8))



                X_histogram_average = numpy.mean([data.get("histogram") for data in X_info], axis = 0)            
                Y_histogram_average = numpy.mean([data.get("histogram") for data in Y_info], axis = 0)
                
                axes[0][0].plot(X_histogram_average[0], color = "red")
                axes[0][0].plot(X_histogram_average[1], color = "green")
                axes[0][0].plot(X_histogram_average[2], color = "blue")
                
                axes[0][0].set_title("Average RGB Histogram X")
                
                axes[0][1].plot(Y_histogram_average[0], color = "red")
                axes[0][1].plot(Y_histogram_average[1], color = "green")
                axes[0][1].plot(Y_histogram_average[2], color = "blue")
                
                axes[0][1].set_title("Average RGB Histogram Y")
                
                
                X_img = cv2.imread(self.data.loc[random.randint(0, self.__len__())]["X"])
                # Convert to grayscale
                gray = cv2.cvtColor(X_img, cv2.COLOR_BGR2GRAY)                
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                
                axes[1][0].imshow(laplacian, cmap = "gray")
                axes[1][0].set_title("Laplacian")


                dataset_X = pandas.DataFrame(X_features, 
                                           columns = ["count", "max","min","average","median","std", "unique"],
                                           index = keys)

                dataset_Y = pandas.DataFrame(X_features, 
                                           columns = ["count", "max","min","average","median","std", "unique"],
                                           index = keys)

                delta = time.process_time() - start_time
                print("Data Evaluation Time: {0} hours, {1} minutes, {1:.3f} seconds".format(int(delta/24), int(delta/60), delta%60))


                matplotlib.pyplot.show()
                
                return dataset_X, dataset_Y
            
    def preprocessing (self):
        
        if(self.debugMode == True):
          print("Preprocessing Process Initialization")
        
        
        def data_augementation (self):
            pass
        def data_sythesis(self):
            pass


        if(self.debugMode == True):
          print("Preprocessing Process Completed")
        
        pass

   
    def train_test_split(self, test_factor, seed):
        
        if(self.debugMode == True):
            print("Dataset Train_Test Split Initialization")
            print("Test Factor: ".format(test_factor))

        self.test_factor = test_factor
        
        if (seed is None):
                X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
                    self.data["X"],
                    self.data["Y"],
                    test_size = test_factor)
                
        else:
                X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
                    self.data["X"],
                    self.data["Y"],
                    test_size = test_factor,
                    random_state = seed)
            
        
        self.data_train = pandas.concat([X_train, Y_train], axis = 1)
        self.data_test = pandas.concat([X_test, Y_test], axis = 1)
            
        self.train_size = len(self.data_train)
        self.test_size = len(self.data_test)

        
        fig, axes = matplotlib.pyplot.subplots(1,1, figsize = (3,3))
        plot = axes.bar(["Train", "Test"], [self.train_size, self.test_size])
        axes.set_title("Train Test Split Factor: {}".format(self.test_factor))
        axes.set_xlabel("Datasets")
        axes.set_ylabel("Sample Count")
        
        '''
        for rect, label in plot:
            height = rect.get_height()
            top = height
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, top , ha="center", va="bottom")
        '''
        #axes.set_grid()
        #axes.legend()
        matplotlib.pyplot.show()


        if(self.debugMode == True):
          print("Dataset Train_Test Split Complete")
            
            
    def init_dataloader(self, for_training = True, batch_size = 32, multiple_processes = 2, shuffle = True, drop_last = False):           

        if (for_training == True):
            
            class Auxiliar(torch.utils.data.Dataset):
                def __init__(self, data):
                    self.data = data
        
                def __getitem__(self, index):
                    x, y = self.data[index]
                    return (x, y)
    
                def __len__(self):
                    return len(self.data)
            
            train = Auxiliar(self.data_train)
            test = Auxiliar(self.data_test)

            self.Dataloader_train = torch.utils.data.DataLoader(train,
                                                  batch_size = batch_size,
                                                  num_workers = multiple_processes,
                                                  shuffle = shuffle,
                                                  drop_last = drop_last)

            self.Dataloader_test = torch.utils.data.DataLoader(test,
                                                  batch_size = batch_size,
                                                  num_workers = multiple_processes,
                                                  shuffle = shuffle,
                                                  drop_last = drop_last)


        elif (for_training == False):

          #Not for training. For predicting
          self.Dataloader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size = 1,
                                                  num_workers = multiple_processes,
                                                  shuffle = shuffle,
                                                  drop_last = drop_last)

        print("DataLoader Process Complete")
 