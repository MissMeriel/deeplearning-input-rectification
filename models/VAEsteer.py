import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torchvision.utils import make_grid, save_image
from typing import List

# meriels dependencies
from DatasetGenerator import MultiDirectoryDataSequence
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, Normalize
from torch.autograd import Variable
import argparse

NAME = "vae"
Path("models").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/iter").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/epoch").mkdir(exist_ok=True, parents=True)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Model(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 16, 3, 2),
            # nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, 2),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 2),
            # nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 2),
            # nn.BatchNorm2d(256),
            nn.ELU(),
        )
        encoder_output_shape = self.encoder(torch.ones(1, 3, self.input_shape[0], self.input_shape[1])).shape[1:]
        encoder_output_size = np.product(encoder_output_shape)
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_var = nn.Linear(encoder_output_size, latent_dim)
        print(f"\n{encoder_output_shape=}\n{encoder_output_size=}")

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, encoder_output_size),
            Reshape((-1, *encoder_output_shape)),
            nn.ConvTranspose2d(256, 128, 3, 2, padding=(0,0)),#, output_padding=(1,0)), #0, padding=(1)),
            # nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 0), #, padding=(0, 1)),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 4, 1, padding=(1, 0)),  # , padding=(0, 1)),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 0), #, padding=(1, 0)),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            # nn.ConvTranspose2d(32, 32, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, padding=(1,0)),
            # nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, 3, 2),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.ConvTranspose2d(8, 8, 3, 1, padding=(1,0)),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.ConvTranspose2d(8, 3, 3, 2, padding=(0,1)),
            # nn.BatchNorm2d(3),
            # nn.ELU(),
            # nn.ConvTranspose2d(3, 3, 1, 1, padding=(0,1)),
            # nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        print("decoder shape =", self.decoder(torch.ones(1, self.latent_dim)).shape)
        print("decoder size =", np.product(encoder_output_shape))

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x = x.flatten(1)
        result = self.encoder(x).flatten(1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)
        y = y[...,:-1]
        # y = y.view(-1, 1, 200, 200)
        y = y.view(-1, 3, self.input_shape[0], self.input_shape[1])
        return y

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(self.fc_var.weight.device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)


def loss_fn(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    # recons_loss = F.binary_cross_entropy(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss


def train(model, data_loader, num_epochs=300, device=torch.device("cpu"), sample_interval=200, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    #TODO: adjust gamma if it plateaus at some values
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # kld_weight = 0.01 * data_loader.batch_size / len(data_loader.dataset)
    kld_weight = data_loader.batch_size / len(data_loader.dataset)
    all_losses = []
    start_t = time.time()
    z = torch.randn(100, model.latent_dim).to(device)
    model = model.eval()
    # grid = make_grid(model.decode(z).detach().cpu(), 10)
    # save_image(grid, f"samples_{NAME}/epoch/0.png")
    for epoch in range(1, num_epochs + 1):
        epoch_start_t = time.time()
        losses = []
        model = model.train()
        for i, hashmap in enumerate(data_loader, start=1):
            optimizer.zero_grad()
            x = hashmap['image'].float().to(device)
            y = hashmap['steering_input'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)
            recons, x, mu, log_var = model(x)
            loss, rloss, kloss = loss_fn(recons, x, mu, log_var, kld_weight)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if (i % 100) == 0:
                iter_end_t = time.time()
                # epoch, samples, epoch elapsed time,
                print(
                    f"{epoch} {i} [{iter_end_t - epoch_start_t:.1f}]: {losses[-1]:.4f} {np.mean(losses[-10:]):.4f} {rloss.item():.4f} {kloss.item():.4f}"
                )
            batches_done = (epoch - 1) * len(data_loader) + i
            # if batches_done % sample_interval == 0:
                # model = model.eval()
                # save_image(
                #     model.sample(25).detach().cpu(),
                #     f"samples_{NAME}/iter/%d.png" % batches_done,
                #     nrow=5,
                # )
                # model = model.train()
        epoch_end_t = time.time()
        print(f"epoch time: {epoch_end_t - epoch_start_t} seconds")
        print(f"total time: {epoch_end_t - start_t} seconds")
        all_losses.append(losses)
        lr_scheduler.step()
        model = model.eval()
        save_image(
            model.sample(25).detach().cpu(),
            f"samples_{NAME}/iter/%d.png" % batches_done,
            nrow=5,
        )
        grid = make_grid(model.decode(z).detach().cpu(), 10)
        save_image(grid, f"samples_{NAME}/epoch/{epoch}.png")
    end_t = time.time()
    print(f"total time: {end_t - start_t} seconds")
    return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='parent directory of training dataset')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    print(args)
    start_time = time.time()
    BATCH_SIZE = 64
    NB_EPOCH = 100
    lr = 1e-4
    robustification = False
    noise_level = 20
    model = Model(input_shape=(135,240), latent_dim=512)
    dataset = MultiDirectoryDataSequence(args.dataset, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=robustification, noise_level=noise_level) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples())
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
    print("time to load dataset: {}".format(time.time() - start_time))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    model = model.to(device)
    model = train(model, trainloader, device=device, num_epochs=NB_EPOCH, sample_interval=20000, lr=lr)
    model = model.to(torch.device("cpu"))
    model = model.eval()
    model_filename = "VAEsteer-4thattempt-nobatchnorm.pt"
    torch.save(model, model_filename)


if __name__ == "__main__":
    main()