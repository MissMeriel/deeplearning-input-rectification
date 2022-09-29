import numpy as np
# import keras
from tensorflow import keras
import os, csv
# import kornia
from PIL import Image
# import copy, cv2
# from scipy import stats
# from pathlib import Path
# import skimage.io as sio
# import pandas as pd
# import random

class DatasetGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def  __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.feature = feature
        self.shuffle = shuffle
        self.training_dir = 'H:/BeamNG_DAVE2_racetracks/'
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print(int(np.floor(len(self.list_IDs) / self.batch_size)))
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #
        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y1, y2 = self.data_generation(index)
        return X, y1, y2

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    def data_generation(self, i):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        training_dir = "{}training_images_industrial-racetrackstartinggate{}".format(self.training_dir, i)
        X, y1, y2 = self.process_training_dir(training_dir)
        return X, y1, y2

    def process_csv(self, filename):
        global path_to_trainingdir
        hashmap = {}
        with open(filename) as csvfile:
            metadata = csv.reader(csvfile, delimiter=',')
            next(metadata)
            for row in metadata:
                imgfile = row[0].replace("\\", "/")
                hashmap[imgfile] = row[1:]
        return hashmap

    # retrieve number of samples in dataset
    def get_dataset_size(self, dir_filename):
        dir1_files = os.listdir(dir_filename)
        dir1_files.remove("data.csv")
        return len(dir1_files)
    
    # assumes training directory has 10,000 samples
    # resulting size: (10000, 150, 200, 3)
    def process_training_dir(self, trainingdir, size=(150, 200)):
        td = [i for i in os.listdir(trainingdir) if '.jpg' in i]
        td = [i for i in td if 'collection_trajectory.jpg' not in i]
        # td.remove("data.csv")
        samples = self.get_dataset_size(trainingdir)
        X_train = np.empty((samples, *size, 3))
        steering_Y_train = np.empty((samples))
        throttle_Y_train = np.empty((samples))
        hashmap = self.process_csv("{}/data.csv".format(trainingdir))
        for index, img in enumerate(td):
            img_file = "{}/{}".format(trainingdir, img)
            X_train[index] = Image.open(img_file).resize(size[::-1])
            steering_Y_train[index] = float(hashmap[img][1])
            throttle_Y_train[index] = float(hashmap[img][2])
        return np.asarray(X_train), np.asarray(steering_Y_train), np.asarray(throttle_Y_train)

    def process_img_dir(self, trainingdir, size=(150, 200)):
        td = [i for i in os.listdir(trainingdir) if '.jpg' in i]
        # td = [i for i in os.listdir(trainingdir) if '.png' in i]
        td = [i for i in td if 'collection_trajectory.jpg' not in i]
        # td.remove("data.csv")
        samples = self.get_dataset_size(trainingdir)
        X_train = np.empty((samples, *size, 3))
        for index, img in enumerate(td):
            img_file = "{}/{}".format(trainingdir, img)
            X_train[index] = Image.open(img_file).resize(size[::-1])
        return np.asarray(X_train)