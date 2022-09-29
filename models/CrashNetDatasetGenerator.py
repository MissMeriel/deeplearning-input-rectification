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

def stripleftchars(s):
    # print(f"{s=}")
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root, image_size=(100,100), transform=None, robustification=False, noise_level=10):
        self.root = root
        self.transform = transform
        self.size = 0
        self.image_size = image_size
        image_paths_hashmap = {}
        all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        for p in Path(root).iterdir():
            if p.is_dir():
                self.dirs.append("{}/{}".format(p.parent,p.stem))
                image_paths = []
                try:
                    self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.txt")
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
        df_index = df.index[df['IMG'] == img_name.name]
        dists = df.loc[df_index, 'DISTS'].item()
        dists = dists.replace("[","").replace("]","")
        dists = np.fromstring(dists, dtype=float, sep=' ')
        dists = np.flip(dists)
        if self.robustification:
            # add noise
            image = copy.deepcopy(orig_image)
            image = torch.clamp(image + torch.randn(*image.shape) / self.noise_level, 0, 1)
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

        sample = {"image": image, "dists": torch.FloatTensor([dists])}
        orig_sample = {"image": image, "dists": torch.FloatTensor([dists])}
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
