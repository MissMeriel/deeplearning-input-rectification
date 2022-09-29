import numpy as np
import pandas as pd
import scipy.stats as stats
import skimage.io as sio
import torch
import torch.utils.data as data

from pathlib import Path


class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.size = 0
        image_paths_hashmap = {}
        all_image_paths = []
        self.dfs_hashmap = {}
        for p in Path(root).expanduser().resolve().iterdir():
            if "100K" not in str(p):
                image_paths = []
                self.dfs_hashmap[f"{p}"] = pd.read_csv(p / "data.csv")
                for pp in p.iterdir():
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                        image_paths.append(pp)
                        all_image_paths.append(pp)
                image_paths_hashmap[p] = image_paths
                self.size += len(image_paths)
        print("Finished intaking image paths!!")
        self.image_paths_hashmap = image_paths_hashmap
        self.all_image_paths = all_image_paths
        self.cache = {}

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image = sio.imread(img_name)
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df["filename"] == img_name.name]
        y_steer = df.loc[df_index, "steering_input"]
        if self.transform:
            image = self.transform(image)
        # sample = {"image": image, "steering_input": y_steer.array.item()}
        sample = (image, torch.FloatTensor([y_steer.item()]))
        self.cache[idx] = sample
        return sample

    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            arr = df["steering_input"].to_numpy()
            print(f"{len(arr)=}")
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            print(f"Retrieved dataframe {key=}")
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments

    ##################################################
    # ANALYSIS METHODS
    ##################################################

    # Moments are 1=mean 2=variance 3=skewness, 4=kurtosis
    def get_distribution_moments(self, arr):
        moments = {}
        moments["shape"] = np.asarray(arr).shape
        moments["mean"] = np.mean(arr)
        moments["median"] = np.median(arr)
        moments["var"] = np.var(arr)
        moments["skew"] = stats.skew(arr)
        moments["kurtosis"] = stats.kurtosis(arr)
        moments["max"] = max(arr)
        moments["min"] = min(arr)
        return moments
