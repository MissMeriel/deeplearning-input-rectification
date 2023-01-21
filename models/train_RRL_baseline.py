import numpy as np
import argparse
import os
from RRLDatasetGenerator import MultiDirectoryDataSequence
import time

from DAVE2pytorch import DAVE2v3

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
import torch.nn as nn
from torch.utils import data

from PIL import Image
import PIL
import matplotlib.pyplot as plt
import csv
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json
import h5py

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='parent directory of base model dataset', default="H:/BeamNG_DeepBillboard_dataset2")
    parser.add_argument('RRL_dir', help='parent directory of RRL dataset', default="H:/RRL-results/RLtrain-max200epi-DDPGhuman-0.05evaleps-bigimg-1_19-20_43-IW6ZTT/RLtrain-max200epi-DDPGhuman-0.05evaleps-bigimg-1_19-20_43-IW6ZTT")
    args = parser.parse_args()
    return args

def train_baseline_model():
    start_time = time.time()
    input_shape = (270, 480)
    model = DAVE2v3(input_shape=input_shape)
    BATCH_SIZE = 64
    NB_EPOCH = 100
    lr = 1e-4
    robustification = True
    noise_level = 20
    args = parse_arguments()
    dataset = MultiDirectoryDataSequence(args.dataset, args.RRL_dir, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=robustification, noise_level=noise_level,
                                         effect="resize") #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples())
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
    print("Processed datapoint paths in {0:.3g} sec.".format(time.time() - start_time))

    iteration = f'{model._get_name()}-baseplusRRL-resinc{input_shape[0]}x{input_shape[1]}-Rturn-lr1e4-{NB_EPOCH}epoch-batch{BATCH_SIZE}-lossMSE-{int(dataset.get_total_samples()/1000)}Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr) #, betas=(0.9, 0.999), eps=1e-08)
    for epoch in range(NB_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image'].float().to(device)
            y = hashmap['steering_input'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            optimizer.zero_grad()

            outputs = model(x)
            # loss = F.mse_loss(outputs.flatten(), y)
            loss = F.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            logfreq = 20
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq))
                running_loss = 0.0
                # from torchvision.utils import save_image
                # save_image(x, "test_david.png", nrow=8)
            # if len(running_loss) > 10:
            #     running_loss[-10:]
        print(f"Finished {epoch=}")
        model_name = f"H:/GitHub/DAVE2-Keras/model-{iteration}-epoch{epoch}.pt"
        print(f"Saving model to {model_name}")
        torch.save(model, model_name)
    print('Finished Training')

    model_name = f'H:/GitHub/DAVE2-Keras/model-{iteration}.pt'
    torch.save(model, model_name)

    # delete models from previous epochs
    print("Deleting models from previous epochs...")
    for epoch in range(NB_EPOCH):
        os.remove(f"H:/GitHub/DAVE2-Keras/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    time_to_train = time.time() - start_time
    print("Time to train: {0:.1g} min".format(time_to_train / 60.0))
    # save metainformation about training
    with open(f'H:/GitHub/DAVE2-Keras/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.get_total_samples()}\n"
                f"{NB_EPOCH=}\n"
                f"{lr=}\n"
                f"{BATCH_SIZE=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{robustification=}\n"
                f"{noise_level=}\n"
                f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")
        # f.write(train_summary)


if __name__ == '__main__':
    train_baseline_model()
