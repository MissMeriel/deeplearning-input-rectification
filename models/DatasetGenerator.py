import numpy as np
import keras
import os, cv2, csv
from DAVE2 import DAVE2Model
from DAVE2pytorch import DAVE2PytorchModel
import kornia

from PIL import Image
import copy
from scipy import stats
# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import torch.utils.data as data
from pathlib import Path
import skimage.io as sio
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import random

from torchvision.transforms import Compose, ToTensor, PILToTensor, functional as transforms


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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        training_dir = "{}training_images_industrial-racetrackstartinggate{}".format(self.training_dir, i)
        m = Model()
        X, y1, y2 = self.process_training_dir(training_dir, m)
        print(X.shape, y1.shape)
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
    def process_training_dir(self, trainingdir, m):
        td = os.listdir(trainingdir)
        td.remove("data.csv")
        size = self.get_dataset_size(trainingdir)
        X_train = np.empty((size, 150, 200, 3))
        steering_Y_train = np.empty((size))
        throttle_Y_train = np.empty((size))
        hashmap = self.process_csv("{}\\data.csv".format(trainingdir))
        for index, img in enumerate(td):
            img_file = "{}/{}".format(trainingdir, img)
            image = Image.open(img_file)
            image = m.process_image(np.asarray(image))
            X_train[index] = image.copy()
            steering_Y_train[index] = float(hashmap[img][1])
            throttle_Y_train[index] =  float(hashmap[img][2])
        # return np.asarray(X_train), np.asarray(steering_Y_train), np.asarray(throttle_Y_train)
        return X_train, steering_Y_train, throttle_Y_train
        # print(len(X_train))
        # print(len(steering_Y_train))
        # return np.asarray(X_train), np.asarray(steering_Y_train)

    def process_enumerated_training_dirs(self, filename_root, trainingdir_indices, m):
        X_train = np.empty((10000 * len(trainingdir_indices), 150, 200, 3))
        steering_Y_train = np.empty((10000 * len(trainingdir_indices)))
        throttle_Y_train = np.empty((10000 * len(trainingdir_indices)))
        for i in trainingdir_indices:
            trainingdir = "{}{}".format(filename_root, i)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}\\data.csv".format(trainingdir))
            for index, img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                # image = cv2.imwrite(img_file)
                image = m.process_image(np.asarray(image))
                # plt.imshow(np.reshape(image, m.input_shape))
                # plt.pause(0.00001)
                adjusted_index = index + i * 10000
                X_train[adjusted_index] = image.copy()
                steering_Y_train[adjusted_index] = float(hashmap[img][1])
                throttle_Y_train[adjusted_index] = float(hashmap[img][2])
            return X_train, steering_Y_train, throttle_Y_train

    def process_all_training_dirs(self, m):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/'
        dirs = os.listdir(rootdir)
        sizes = []
        dirs = dirs[:1]
        for d in dirs:
            sizes.append(self.get_dataset_size(rootdir + d))
        X_train = np.empty((sum(sizes), 150, 200, 3))
        steering_Y_train = np.empty((sum(sizes)))
        throttle_Y_train = np.empty((sum(sizes)))
        adjusted_index = 0
        for i,d in enumerate(dirs[:1]):
            trainingdir = "{}{}".format(rootdir, d)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}/data.csv".format(trainingdir))
            print("size of {}: {}".format(d, sizes[i]))
            print("adjusted_index start", adjusted_index)
            for index,img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                image = m.process_image(np.asarray(image))
                adjusted_index = sum(sizes[:i]) + index
                # print("adjusted_index", adjusted_index)
                X_train[adjusted_index] = image.copy()
                steering_Y_train[adjusted_index] = copy.deepcopy(float(hashmap[img][1]))
                throttle_Y_train[adjusted_index] = copy.deepcopy(float(hashmap[img][2]))
            print("adjusted_index end", adjusted_index)
        return X_train, steering_Y_train, throttle_Y_train

    def process_all_training_dirs_with_2D_output(self):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/PID/'
        dirs = os.listdir(rootdir)
        dirs = [d for d in dirs if "100K" not in d]
        sizes = []
        print(len(dirs), "dirs:", dirs)
        for d in dirs:
            sizes.append(self.get_dataset_size(rootdir + d))
        X_train = np.empty((sum(sizes), 150, 200, 3))
        # Y_train = np.empty((sum(sizes), 2))
        y_all = np.empty((sum(sizes), 2))
        y_steering = np.empty((sum(sizes), 1))
        y_throttle = np.empty((sum(sizes), 1))
        adjusted_index = 0
        m = DAVE2Model()
        for i,d in enumerate(dirs):
            trainingdir = "{}{}".format(rootdir, d)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}/data.csv".format(trainingdir))
            print("size of {}: {}".format(d, sizes[i]))
            print("adjusted_index start", adjusted_index)
            for index,img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                image = m.process_image(np.asarray(image))
                adjusted_index = sum(sizes[:i]) + index
                # print("adjusted_index", adjusted_index)
                X_train[adjusted_index] = image.copy()
                # output vector is [steering, throttle]
                y_all[adjusted_index] = [copy.deepcopy(float(hashmap[img][1])),
                                           copy.deepcopy(float(hashmap[img][2]))]
                y_steering[adjusted_index] = copy.deepcopy(float(hashmap[img][1]))
                y_throttle[adjusted_index] = copy.deepcopy(float(hashmap[img][2]))
            print("adjusted_index end", adjusted_index)
        return X_train, y_all, y_steering, y_throttle

    def get_endnumber(self, filename):
        if ".jpg" in filename:
            img_filename = filename.replace(".jpg", "")
            l = img_filename.split("_")[-1]
            return float(l)
        elif "training_images_" in filename:
            # img_filename = filename.replace("racetrackstartinggate", "")
            l = filename.split("racetrackstartinggate")[-1]
            return float(l)

    def sort_dirs(self, dirs):
        dirs.sort(reverse=False, key=self.get_endnumber)
        return dirs

    def process_all_training_dirs_with_2D_output_and_multi_input(self, m):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/PID/'
        dirs = os.listdir(rootdir)
        dirs = self.sort_dirs(dirs)
        sizes = []
        for d in dirs:
            sizes.append(self.get_dataset_size(rootdir + d))
        X_train = np.empty((sum(sizes), 150, 200, 3))
        # Y_train = np.empty((sum(sizes), 2))
        X_kph = np.zeros((sum(sizes), 1))
        y_all = np.empty((sum(sizes), 2))
        y_steering = np.empty((sum(sizes), 2))
        y_throttle = np.empty((sum(sizes), 2))
        adjusted_index = 0
        for i,d in enumerate(dirs):
            trainingdir = "{}{}".format(rootdir, d)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}/data.csv".format(trainingdir))
            print("size of {}: {}".format(d, sizes[i]))
            print("adjusted_index start", adjusted_index)
            for index,img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                image = m.process_image(np.asarray(image))
                adjusted_index = sum(sizes[:i]) + index
                # print("adjusted_index", adjusted_index)
                X_train[adjusted_index] = image.copy()
                # output vector is [steering, throttle]
                y_all[adjusted_index] = [copy.deepcopy(float(hashmap[img][1])),
                                           copy.deepcopy(float(hashmap[img][2]))]
                y_steering[adjusted_index] = copy.deepcopy(float(hashmap[img][1]))
                y_throttle[adjusted_index] = copy.deepcopy(float(hashmap[img][2]))
                X_kph[adjusted_index] = copy.deepcopy(float(hashmap[img][-1]) * 3.6) # m/s converted to kph
            print("adjusted_index end", adjusted_index)
        return X_train, X_kph, y_all, y_steering, y_throttle

    def process_csv_with_random_selection(self, filename, count):
        global path_to_trainingdir
        hashmap = {}
        with open(filename) as csvfile:
            metadata = csv.reader(csvfile, delimiter=',')
            next(metadata)
            for row in metadata:
                imgfile = row[0].replace("\\", "/")
                if abs(float(row[2])) < 0.1:
                    if count > 70000:
                        hashmap[imgfile] = row[1:]
                    else:
                        count += 1
                else:
                    hashmap[imgfile] = row[1:]
        return hashmap, count

    def process_all_training_dirs_with_random_selection(self, m):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/'
        dirs = os.listdir(rootdir)
        sizes = []
        for d in dirs:
            sizes.append(self.get_dataset_size(rootdir + d))
        dataset_size = sum(sizes) - 70000 + 1
        print("dataset_size", dataset_size)
        print("dirs", dirs)
        X_train = np.empty((dataset_size, 150, 200, 3))
        steering_Y_train = np.empty((dataset_size))
        throttle_Y_train = np.empty((dataset_size))
        count = 0
        trainindex = 0
        totalcount = 0
        for i,d in enumerate(dirs):
            trainingdir = "{}{}".format(rootdir, d)
            print(trainingdir)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap, count = self.process_csv_with_random_selection("{}\\data.csv".format(trainingdir), count)
            for index,img in enumerate(hashmap.keys()):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                image = m.process_image(np.asarray(image))
                X_train[trainindex] = image.copy()
                steering_Y_train[trainindex] = copy.deepcopy(float(hashmap[img][1]))
                throttle_Y_train[trainindex] = copy.deepcopy(float(hashmap[img][2]))
                trainindex += 1
        return X_train, steering_Y_train, throttle_Y_train

    ##################################################
    # PYTORCH DATASETS
    ##################################################

    def process_all_training_dirs_pytorch(self):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/PID/'
        dirs = os.listdir(rootdir)
        sizes = [self.get_dataset_size(rootdir + d) for d in dirs]
        print("dirs:", dirs)
        # exit(0)
        X_train = np.empty((sum(sizes), 150, 200, 3))
        # Y_train = np.empty((sum(sizes), 2))
        # y_all = np.empty((sum(sizes), 2))
        y_steering = np.empty((sum(sizes), 1))
        # y_throttle = np.empty((sum(sizes), 1))
        adjusted_index = 0
        m = DAVE2Model()
        for i,d in enumerate(dirs):
            trainingdir = "{}{}".format(rootdir, d)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}/data.csv".format(trainingdir))
            print("size of {}: {}".format(d, sizes[i]))
            print("adjusted_index start", adjusted_index)
            for index,img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                image = m.process_image(np.asarray(image))
                adjusted_index = sum(sizes[:i]) + index
                # print("adjusted_index", adjusted_index)
                X_train[adjusted_index] = image.copy()
                # output vector is [steering, throttle]
                # y_all[adjusted_index] = [copy.deepcopy(float(hashmap[img][1])),
                #                            copy.deepcopy(float(hashmap[img][2]))]
                y_steering[adjusted_index] = copy.deepcopy(float(hashmap[img][1]))
                # y_throttle[adjusted_index] = copy.deepcopy(float(hashmap[img][2]))
            print("adjusted_index end", adjusted_index)
        # return X_train, y_all, y_steering, y_throttle
        return X_train, y_steering

    ##################################################
    # DATASET REASSEMBLY
    ##################################################

    def row_hashmap_to_string(self, hashmap, new_filename, columns):
        for entry in hashmap[1:]:
            new_filename = "{},{}".format(new_filename,entry)
        return new_filename + "\n"

    def restructure_training_set(self):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/PID/'
        new_rootdir = 'H:/BeamNG_DAVE2_racetracks_all/restruct2/'
        dirs = os.listdir(rootdir)
        sizes = [self.get_dataset_size(rootdir + d) for d in dirs]
        # print(f"{dirs=}")
        # print(f"{sizes=}")
        adjusted_index = 0
        main_df = pd.DataFrame()
        with open(f"{new_rootdir}/data.csv", 'w') as f:
            f.write("filename,timestamp,steering_input,throttle_input,brake_input,driveshaft,engine_load,fog_lights,fuel,lowpressure,oil,oil_temperature,parkingbrake,rpm,water_temperature,wheelspeed\n")
            for i,d in enumerate(dirs):
                trainingdir = "{}{}".format(rootdir, d)
                td = os.listdir(trainingdir)
                td.remove("data.csv")
                # print(f"{td=}")
                hashmap = self.process_csv("{}/data.csv".format(trainingdir))
                print("size of {}: {}".format(d, sizes[i]))
                print("adjusted_index start", adjusted_index)
                df = pd.read_csv("{}/data.csv".format(trainingdir))
                # new_df = pd.DataFrame([], columns=df.columns)
                for index, img in enumerate(td):
                    img_file = "{}/{}".format(trainingdir, img)
                    image = Image.open(img_file)
                    adjusted_index = sum(sizes[:i]) + index
                    # print(f"{adjusted_index=}")
                    new_img_filename = "{}hopper_industrial_{}.jpg".format(new_rootdir, adjusted_index)
                    # print(f"{new_img_filename=}")
                    image.save(new_img_filename)
                    row_string = self.row_hashmap_to_string(hashmap[img], f"hopper_industrial_{adjusted_index}.jpg", df.columns)
                    f.write(row_string)
                # df_index = df.index[df['filename'] == img]
                # df.loc[df_index, 'filename'] = "hopper_industrial_{}.jpg".format(adjusted_index)
                # print("\n", img, "\n", df[df['filename'] == img])
                # row = df[df['filename'] == img]
                # row['filename']= "hopper_industrial_{}.jpg".format(adjusted_index)
                # print(df[df['filename'] == f"hopper_industrial_{adjusted_index}.jpg"])
                # new_df.append(row)
                # row = row.to_string(header=False, index=False,index_names=False).split('\n')
                # row = [','.join(ele.split()) for ele in row]
                # f.write("{}\n".format(row))
            # main_df = pd.concat([main_df, new_df])
        # main_df.to_csv(f"{new_rootdir}data.csv", index=False)
        print("Finished dataset restruct")

def stripleftchars(s):
    # print(f"{s=}")
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

class DataSequence(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform

        image_paths = []
        for p in Path(root).iterdir():
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                image_paths.append(p)
        image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
        self.image_paths = image_paths
        # print(f"{self.image_paths=}")
        self.df = pd.read_csv(f"{self.root}/data.csv")
        self.cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.image_paths[idx]
        image = sio.imread(img_name)

        df_index = self.df.index[self.df['filename'] == img_name.name]
        y_thro = self.df.loc[df_index, 'throttle_input'].array[0]
        y_steer = self.df.loc[df_index, 'steering_input'].array[0]
        y = [y_steer, y_thro]
        # torch.stack(y, dim=1)
        y = torch.tensor(y_steer)

        # plt.title(f"steering_input={y_steer.array[0]}")
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.01)

        if self.transform:
            image = self.transform(image).float()
        # print(f"{img_name.name=} {y_steer=}")
        # print(f"{image=}")
        # print(f"{type(image)=}")
        # print(self.df)
        # print(y_steer.array[0])

        # sample = {"image": image, "steering_input": y_steer.array[0]}
        sample = {"image": image, "steering_input": y}

        self.cache[idx] = sample
        return sample

class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root, image_size=(100,100), transform=None, robustification=False, noise_level=10):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.size = 0
        self.image_size = image_size
        image_paths_hashmap = {}
        all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        for p in Path(root).iterdir():
            if p.is_dir() and "_YESM" in str(p): #"_NO" not in str(p) and "YQWHF3" not in str(p):
                self.dirs.append("{}/{}".format(p.parent,p.stem.replace("_YESM", "")))
                image_paths = []
                try:
                    self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                except FileNotFoundError as e:
                    print(e, "\nNo data.csv in directory")
                    continue
                for pp in Path(p).iterdir():
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "collection_trajectory" not in pp.name:
                        image_paths.append(pp)
                        all_image_paths.append(pp)
                image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
                image_paths_hashmap[p] = copy.deepcopy(image_paths)
                self.size += len(image_paths)
        print("Finished intaking image paths!")
        self.image_paths_hashmap = image_paths_hashmap
        self.all_image_paths = all_image_paths
        # self.df = pd.read_csv(f"{self.root}/data.csv")
        self.cache = {}
        self.robustification = robustification
        self.noise_level = noise_level

    def get_total_samples(self):
        return self.size

    def get_directories(self):
        return self.dirs
        
    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)
        image = image.resize(self.image_size)
        orig_image = self.transform(image)
        pathobj = Path(img_name)
        # print(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df['filename'] == img_name.name]
        orig_y_steer = df.loc[df_index, 'steering_input'].item()
        y_throttle = df.loc[df_index, 'throttle_input'].item()
        y_steer = copy.deepcopy(orig_y_steer)
        if self.robustification:
            # add noise
            image = copy.deepcopy(orig_image)
            image = torch.clamp(image + torch.randn(*image.shape) / self.noise_level, 0, 1)
            if random.random() > 0.5:
                # flip image
                # plt.imshow(image.permute(1,2,0))
                # plt.pause(0.01)
                image = torch.flip(image, (2,))
                # plt.imshow(image.permute(1,2,0))
                # plt.pause(0.01)
                y_steer = -orig_y_steer
            if random.random() > 0.5:
                # blur
                gauss = kornia.filters.GaussianBlur2d((5, 5), (5.5, 5.5))
                image = gauss(image[None])[0]
                # image = kornia.filters.blur_pool2d(image[None], 3)[0]
                # image = kornia.filters.max_blur_pool2d(image[None], 3, ceil_mode=True)[0]
                # image = kornia.filters.median_blur(image, (3, 3))
                # image = kornia.filters.median_blur(image, (10, 10))
                # image = kornia.filters.box_blur(image, (3, 3))
                # image = kornia.filters.box_blur(image, (5, 5))
                # image = kornia.resize(image, image.shape[2:])
                # plt.imshow(image.permute(1,2,0))
                # plt.pause(0.01)
        else:
            # if type(image) == Image.Image:
            t = Compose([ToTensor()])
            image = t(image).float()
            # image = torch.from_numpy(image).permute(2,0,1) / 127.5 - 1

        # vvvvvv uncomment below for value-image debugging vvvvvv
        # plt.title(f"{img_name}\nsteering_input={y_steer.array[0]}", fontsize=7)
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.01)

        sample = {"image": image, "steering_input": torch.FloatTensor([y_steer]), "throttle_input": torch.FloatTensor([y_throttle]), "all": torch.FloatTensor([y_steer, y_throttle])}
        orig_sample = {"image": orig_image, "steering_input": torch.FloatTensor([orig_y_steer]), "throttle_input": torch.FloatTensor([y_throttle]), "all": torch.FloatTensor([orig_y_steer, y_throttle])}
        self.cache[idx] = orig_sample
        return sample

    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            arr = df['steering_input'].to_numpy()
            # print("len(arr)=", len(arr))
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            # print(f"Retrieved dataframe {key=}")
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments

    ##################################################
    # ANALYSIS METHODS
    ##################################################

    # Moments are 1=mean 2=variance 3=skewness, 4=kurtosis
    def get_distribution_moments(self, arr):
        moments = {}
        moments['shape'] = np.asarray(arr).shape
        moments['mean'] = np.mean(arr)
        moments['median'] = np.median(arr)
        moments['var'] = np.var(arr)
        moments['skew'] = stats.skew(arr)
        moments['kurtosis'] = stats.kurtosis(arr)
        moments['max'] = max(arr)
        moments['min'] = min(arr)
        return moments
