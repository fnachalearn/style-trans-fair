'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC, SVC
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau



class Net(nn.Module):
    def __init__(self, number_of_classes, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * ((((input_shape[0]-4)//2)-4)//2) * ((((input_shape[1]-4)//2)-4)//2), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, number_of_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class model (BaseEstimator):
    def __init__(self, number_of_classes, input_shape):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''

        self.epochs = 10
        self.batch_size = 4
        self.initial_learning_rate = 0.001

        self.num_labels=number_of_classes
        self.is_trained=False
        self.enc = OneHotEncoder(handle_unknown='ignore')
        

        self.__model = Net(number_of_classes, input_shape)

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        '''
                here the imput -X is an np.array() of shape [number of images, height, width, channel]
                               -y is an np.array() of shape[number of images,]
                what we did is to resize all the images and flatten them from 3D to 1D and
                after this we transform the list of flatten image as array. And then we pass it into 
                the fit function.
                
         '''


        X = torch.from_numpy(X.reshape(X.shape[0], X.shape[-1], X.shape[2], X.shape[1])).to(torch.float32)
        y = torch.from_numpy(y).to(torch.long)

        train_dataset = TensorDataset(X, y)
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__model.parameters(), lr=self.initial_learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader)*self.epochs)

        running_loss = RunningLoss()

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            print("epoch", epoch)
            running_loss.reset()
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))
            for i, data in pbar:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.__model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                optimizer.step()
                scheduler.step()


                # set the running_loss at tqdm bar 
                running_loss.update(loss.item())
                pbar.set_description(f"loss: {running_loss.get():.4f}")

                

        print('Finished Training')
        

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        
        '''
                here the imputs :  -X is an np.array() of shape [number of images, height, width]
                                 - y is an np.array() of shape[number of images,]
                what we did is to resize all the image and flatten them from 3D to 1D and
                after this we transform the list of flatten image as array. And then we pass it into 
                the predict function.
                
         '''
        X = torch.from_numpy(X.reshape(X.shape[0], X.shape[-1], X.shape[1], X.shape[2])).to(torch.float32)
        result = self.__model(X).detach().numpy()

        # # Run inference on CPU
        # with tf.device('/cpu:0'):
        #     result = self.__model.predict(X)

        return np.argmax(result, axis=1)
            


    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

class RunningLoss():
    def __init__(self):
        self.reset()
        self.count = 0

    def reset(self):
        self.loss = 0
        self.count = 0
    
    def update(self, loss):
        self.loss += loss
        self.count += 1
    
    def get(self):
        return self.loss / self.count