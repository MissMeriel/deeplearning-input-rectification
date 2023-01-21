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
from wand.image import Image as WandImage
from io import BytesIO
import skimage

def stripleftchars(s):
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

def striplastchars(s):
    s = s.split("-")[-1]
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root, RRL_dir, image_size=(100,100), transform=None,
                 robustification=False, noise_level=10, effect=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.RRL_dir = RRL_dir
        self.effect = effect
        self.transform = transform
        self.robustification = robustification
        self.noise_level = noise_level
        self.size = 0
        self.image_size = image_size
        self.image_paths_hashmap = {}
        self.all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        self.process_basemodel_dirs()
        self.process_RRL_dir()
        self.cache = {}

    def process_basemodel_dirs(self):
        marker = "_YES"
        for p in Path(self.root).iterdir():
            if p.is_dir() and marker in str(p):  # "_NO" not in str(p) and "YQWHF3" not in str(p):
                self.dirs.append("{}/{}".format(p.parent, p.stem.replace(marker, "")))
                image_paths = []
                try:
                    self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                except FileNotFoundError as e:
                    print(e, "\nNo data.csv in directory")
                    continue
                for pp in Path(p).iterdir():
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg",
                                             ".bmp"] and "collection_trajectory" not in pp.name:
                        image_paths.append(pp)
                        self.all_image_paths.append(pp)
                image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
                self.image_paths_hashmap[p] = copy.deepcopy(image_paths)
                self.size += len(image_paths)

    def process_RRL_dir(self):
        skipcount = 0
        if self.RRL_dir is not None and Path(self.RRL_dir).is_dir():
            for p in Path(self.RRL_dir).iterdir():
                if p.is_dir() and "tb_logs_DDPG" not in str(p):
                    ep = int(str(p.stem).replace("ep", ""))
                    if ep % 8 != 0:
                        self.dirs.append("{}/{}".format(p.parent, p.stem))
                        image_paths = []
                        try:
                            self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                        except FileNotFoundError as e:
                            print(e, "\nNo data.csv in directory")
                            continue
                        for pp in Path(p).iterdir():
                            if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                                image_paths.append(pp)
                                self.all_image_paths.append(pp)
                        image_paths.sort(key=lambda p: int(striplastchars(p.stem)))
                        self.image_paths_hashmap[p] = copy.deepcopy(image_paths)
                        self.size += len(image_paths)
                    else:
                        skipcount += 1
                        print(f"Skipping {str(p)}, {skipcount=} ")

    def get_total_samples(self):
        return self.size

    def get_directories(self):
        return self.dirs
        
    def __len__(self):
        return len(self.all_image_paths)


    def fisheye(self, image):
        with WandImage.from_array(image) as img:
            # img.format = 'bmp'
            img.virtual_pixel = 'transparent'
            img.distort('barrel', (0.1, 0.0, -0.05, 1.0))
            img.alpha_channel = False
        #     img_buffer = np.asarray(bytearray(img.make_blob()), dtype='uint8')
        # bytesio = BytesIO(img_buffer)
        # img = skimage.io.imread(bytesio)
            img = np.array(img, dtype='uint8')
            # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            # plt.imshow(img)
            # plt.show()
            return img

    def depth_estimation(self, image):
        pass

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)
        if "RRL" not in str(img_name):
            # apply transformation
            if self.effect == "resize":
                image = image.resize(self.image_size)
        # image = cv2.imread(img_name.__str__())
        # image = cv2.resize(image, self.image_size) / 255
        # image = self.fisheye(image)
        orig_image = self.transform(image)
        pathobj = Path(img_name)
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
