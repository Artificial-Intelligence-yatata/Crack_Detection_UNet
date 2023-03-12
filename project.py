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



class Project():
    
    def __init__ (self, model, dataset, problem, problem_type, multiple_processes = 1,
                  debugMode = False):  
        
        
        self.parameter_check(problem, problem_type, multiple_processes)
        
        self.model = model
        self.dataset = dataset    
        self.debugMode = debugMode
        self.savePath = None
        
        
    def parameter_check(self, problem, problem_type, multiple_processes):
        
        problem = problem.lower()
        if (problem != "regression" or problem != "classification" or
            problem != "detection" or problem != "segmentation" or
            problem != "anomaly" or problem != "mix"):
            
            #TODO
            pass
        
        self.problem = problem
        
        problem_type = problem_type.lower()
        if (problem_type != "supervised" or problem_type != "unsupervised" or
            problem_type != "reinforcement" or problem_type != "semi-supervised" or
            problem_type != "transfer" or problem_type != "active" or 
            problem_type != "generative" or problem_type != "recommendation"):
            
            #TODO
            pass
        
        self.problem_type = problem_type
        

        self.multiple_processes = multiple_processes
        
    def setup(self, test_factor = 0.2, seed = None):




      if(self.debugMode == True):
        print("Project Setup Initialization")

      if (not(type(test_factor) == int or type(test_factor) == float) and
          (test_factor > 1.0 or test_factor < 0.0)):

          #TODO error
          print("Test Factor Error")
          pass

      

      self.dataset.problem = self.problem
      self.dataset.problem_type = self.problem_type
      self.dataset.multiple_processes = self.multiple_processes
      self.dataset.debugMode = self.debugMode



      self.dataset.initiate_process(data_test = self.dataset.data_test,
                                    test_factor = test_factor, 
                                    batch_size = 32, shuffle = True, drop_last = False, seed = seed)
      


      self.model.setup(debugMode = self.debugMode, problem = self.problem,
                       problem_type = self.problem_type)
      self.model.build()


      if(self.debugMode == True):
        print("Setup Complete")


    def problem_summary(self):
      
      self.model.__summary__()
      self.model.debbugFoward()
        
    def train_batch(self, data):
        
        if (self.problem_type == "supervised" and
            self.problem_type == "classification"):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        
            # track history if only in train
            with torch.set_grad_enabled(True):
                
                # 1. Forward pass
                # forward + backward + optimize
                if(self.model.baseModel is None):
                    y_pred = self.model.forward(inputs)
                else:
                    y_pred = self.model.baseModel(inputs)
                    
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
        
        return loss.item(), acc
    
    
    
    def train_fold(self, n_epoch, train_loader, best_batch_show = True):
        
        #Epochs
        for epoch in range (0, n_epoch, 1):

            epoch_start = time.time()

            train_loss = []
            train_acc = []
            best_batch_acc = 0.0
            best_batch_loss = 999
            best_batch_iteration = 0

            data_10 = 0
            i = 1

            #train iteration
            if (best_batch_show is True):
                print("Epoch {0}".format(epoch+1), end = ": ")


            for batch_i, data in enumerate(train_loader):

                loss, acc = self.train_batch(data)
                data_10 += len(data)

                # print statistics
                if (best_batch_show is not True):
                    print("Epoch:{0} Batch:{1} Loss = {2:0.4f} Accuracy = {3:0.4f}".format(epoch+1, batch_i+1, loss, acc))

                elif (best_batch_show is True):

                    if (best_batch_acc < acc and best_batch_loss > loss):
                        best_batch_acc = acc
                        best_batch_loss = loss
                        best_batch_iteration = batch_i

                    if (data_10 > self.dataset.trainSize/10 * i):
                        print("{0}-".format(i), end = "")
                        i += 1

                self.train_scores.append([epoch+1, data_10, loss, acc])

                train_loss.append(loss)
                train_acc.append(acc)
    
    
    def trainModel (self, batch_size = 32, n_epochs = 5, KFolds = 0,
                    function_loss = None, optimizer = None,
                    learning_rate = None, device = None, multiple_processes = 2,
                    best_batch_show = True, savePath = None, debbugMode = False,
                    debbugFile_Path = None):
        
        self.nEpoch = n_epochs
        self.KFolds = KFolds

        
                
        self.train_scores = []            #scores of model as number sample increase
        self.train_scores_epochs = []     #scores of model as epoch increase
        self.cv_scores_epochs = []  

        train_start = time.process_time()
        

        print("************************************************************")
        print("Model Training on: {0}".format(self.device))

        if (self.KFolds > 0):
            
            cv_splits = sklearn.model_selection.KFold(n_splits = KFolds, shuffle=True)
            
            for fold, (train_index, cv_index) in enumerate(cv_splits.split(numpy.arange(len(self.dataset.data_train)))):
                
                print("Fold: {0}/{1}".format(fold, self.KFolds),sep = "\n")
                
                train_sampler = torch.utils.dataSubsetRandomSampler(train_index)
                cv_sampler = torch.utils.dataSubsetRandomSampler(cv_index)
                
                train_loader = torch.utils.data.DataLoader(self.dataset.dataset_train, batch_size = self.dataset.batch_size, sampler = train_sampler, drop_last = False)
                cv_loader = torch.utils.data.DataLoader(self.dataset.dataset_train, batch_size = self.dataset.batch_size, sampler = cv_sampler, drop_last = False)
                
                
                self.train_fold(n_epoch = n_epochs, train_loader = train_loader, best_batch_show = best_batch_show)
                
                #Validation iteration

                if (best_batch_show is True):
                    print("Epoch {0}".format(epoch+1), end = ": ")
                
                for batch_i, data in enumerate(cv_loader):
                    loss, acc = self.train_batch(data)
                    
                    cv_loss.append(loss)
                    cv_acc.append(acc)

             
                if (best_batch_show is True):
                    print("\nBest_Batch:{0} Best_Batch_Loss = {1:0.4f} Best_Batch_Accuracy = {2:0.4f}".format(best_batch_iteration+1, best_batch_loss, best_batch_acc))
            
                # Adjust metrics to get average loss and accuracy per batch 
                train_loss_avg = sum(train_loss) / len(self.dataset.Dataloader_train)
                train_acc_avg = sum(train_acc) / len(self.dataset.Dataloader_train)

                self.train_scores_epochs.append([epoch+1, train_acc, train_loss, float((time.time() - epoch_start) // 60), (time.time() - epoch_start) % 60])

                # print statistics
                print("Epoch:{0} Loss_Avg = {1:.4f} Accuracy_Avg = {2:.4f} Time Lapsed = {3}min {4:.2f}seg\n".format(
                    epoch+1, train_loss_avg, train_acc_avg, (time.time() - epoch_start) // 60, (time.time() - epoch_start) % 60))
        

        
        elif (self.KFolds <= 0):
            #no cv
            ###################################################

            self.train_fold(n_epoch = n_epochs, train_loader = train_loader, best_batch_show = best_batch_show)

            if (best_batch_show is True):
                print("\nBest_Batch:{0} Best_Batch_Loss = {1:0.4f} Best_Batch_Accuracy = {2:0.4f}".format(best_batch_iteration+1, best_batch_loss, best_batch_acc))

            # Adjust metrics to get average loss and accuracy per batch 
            train_loss_avg = sum(train_loss) / len(self.dataset.Dataloader_train)
            train_acc_avg = sum(train_acc) / len(self.dataset.Dataloader_train)

            self.train_scores_epochs.append([epoch+1, train_acc, train_loss, float((time.time() - epoch_start) // 60), (time.time() - epoch_start) % 60])


            # print statistics
            print("Epoch:{0} Loss_Avg = {1:.4f} Accuracy_Avg = {2:.4f} Time Lapsed = {3}min {4:.2f}seg\n".format(
                    epoch+1, train_loss_avg, train_acc_avg, (time.time() - epoch_start) // 60, (time.time() - epoch_start) % 60))


            
            
            
        delta_training = time.process_time() - train_start
        print("Model Training Time: {0} hours, {1} minutes, {2:.3f} seconds, {} seconds".format(
                int(delta_training/24),int(delta_training/60), delta_training%60)) 
        print("************************************************************")
    
    
        
        self.model.trained = True
        if (savePath is not None):
            self.savePath = savePath
            self.model.saveModel(savePath)


        if (debbugMode is True and debbugFile_Path is not None):
            file_object = open(debbugFile_Path, 'w')
            for data in self.train_scores_epochs:
                file_object.write(str(data[0]))
                file_object.write('|')
                file_object.write(str(data[1]))
                file_object.write('|')
                file_object.write(str(data[2]))
                file_object.write('|')
                file_object.write(str(data[3]))
                file_object.write('|')
                file_object.write(str(data[4]))

                file_object.write("\n")
            #Close the file
            file_object.close()



        #self.plot_learning_curve()
    
    