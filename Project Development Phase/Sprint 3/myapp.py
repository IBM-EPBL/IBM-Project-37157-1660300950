import numpy as np
#Importing libraries required for the model
import tensorflow as tf
import keras
import keras.backend as K

from tensorflow.keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.applications import *
from keras.preprocessing import *
from keras.preprocessing . image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras . models import Sequential
from keras . layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization , Dropout
from keras . utils . np_utils import to_categorical
from sklearn . model_selection import train_test_split

#For plotting charts used for data visualizations
import matplotlib . pyplot as pit

#Libraries for Locating and loading data
import glob
from PIL import Image
import os
from os import listdir
#Setting path to our dataset folder
dirName = 'C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data'
folders = listdir (dirName)
#Getting the names for all the folders containing data
def getListOfFiles (dirName) :

# create a list of sub directories and files (if any)
# names in the given directory
    listOfFile = os . listdir(dirName)
    allFiles = list( )
    for fol_name in listOfFile:
        fullPath = os . path . join (dirName, fol_name)
        allFiles . append ( fullPath)
    return allFiles

Folders = getListOfFiles (dirName)
len (Folders)
subfolders = []
for num in range (len (Folders ) ) :
    sub_fols = getListOfFiles (Folders[num] )
    subfolders+=sub_fols
#Now, the subfolders contains the address to all our data folders for each class
subfolders
#Loading the data and pre processing it to make it in trainable format
#1III

#X data will includes the data generated for each image
#Y data will include a id no, unique for every different species, so are having 6 classes
#there for we will get 6 ids = [0 , 1, 2,3,4,5]
#That will be tha label we're classifying.
X_data=[]
Y_data=[]
id_no=0

#to make a list of tuples, where we'll store the info about the image, category and species
found = []
#itering in all folders under Augmented data folder
for paths in subfolders:
#setting folder path for each unique class and category
    files = glob . glob (paths + "/*. jpg")
    #adding tuples to the list that contain folder name and subfolder name
    found . append((paths . split('\\' ) [-2] , paths . split( '\\' ) [-1]))

#itering all files under the folder one by one
    for myFile in files:
        img = Image. open (myFile)
        img=img.resize( (224, 224) , Image. ANTIALIAS) # resizes image without ratio
        #convert the images to numpy arrays

        img = np . array (img)
        if img . shape == ( 224, 224, 3) :
    # Add the numpy image to matrix with all data
            X_data . append (img)
            Y_data . append (id_no)
    id_no+=1 
#to see our
    print (X_data)
    print (Y_data)

    #converting lists to np arrays again
    X = np . array (X_data)
    Y = np. array (Y_data)
    # Print shapes to see if they are correct
    print ( "x-shape" , X. shape, "y shape", Y. shape)

    X = X.astype ( 'float32' ) /255.0
    #The keras Library offers a function called to_categorical() that you can use to one hot encode
    #so we can use the to categorical () function directly
    y_cat = to_categorical (Y_data, len(subfolders) )
    print ("X shape" , X, "y_cat shape", y_cat)
    print ("X shape" , X. shape, "y_cat shape" , y_cat . shape)

    X_train,X_test,y_train,y_test=train_test_split(X,y_cat,test_size=0.2)
    print("The model has "+str(len(X_train))+" inputs")