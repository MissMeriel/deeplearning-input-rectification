# import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# adapted from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# env = gym.make('CartPole-v1').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# globals
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super().__init__()
        self.input_shape = (h,w)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.dropout = nn.Dropout()

        size = np.product(nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)(
            torch.zeros(1, 3, *self.input_shape)).shape)

        self.lin1 = nn.Linear(in_features=size, out_features=100, bias=True)
        self.lin2 = nn.Linear(in_features=100, out_features=50, bias=True)
        self.lin3 = nn.Linear(in_features=50, out_features=10, bias=True)
        self.lin4 = nn.Linear(in_features=10, out_features=2, bias=True)
        self.lin4 = nn.Linear(in_features=10, out_features=3, bias=True)


    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = self.conv5(x)
        x = F.elu(x)
        x = x.flatten(1)
        x = self.lin1(x)
        #x = self.dropout(x)
        x = F.elu(x)
        x = self.lin2(x)
        #x = self.dropout(x)
        #x = F.elu(x)
        x = self.lin3(x)
        #x = self.dropout(x)
        #x = F.elu(x)
        x = self.lin4(x)
        x = torch.tanh(x)
        # x = 2 * torch.atan(x)
        return x


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
screen_height, screen_width = 135, 240
n_actions = 3
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=0.005)  # , betas=(0.9, 0.999), eps=1e-08)
#optimizer = optim.RMSprop(policy_net.parameters())
loss = 0


def save_policies(epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"target_net-{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"policy_net-{epoch}.pt")



def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state = state.to(device)
            policy_net_prediction = policy_net(state)
            # print(f"policy_net_prediction={policy_net_prediction[0]}")
            # print(f"select_action() Returning {policy_net_prediction.max(1)[1].view(1, 1)}")
            return policy_net_prediction.max(1)[1].view(1, 1)
    else:
        # print(f"select_action() Returning {torch.tensor([[random.randrange(n_actions)]])}")
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(save=False):
    # plt.figure(2)
    plt.figure(2, figsize=(2,2), dpi=80)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    # plt.xticks(list(range(1, max(durations_t) + 1)), [str(i) for i in range(1, max(durations_t) + 1)])
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    if save:
        plt.savefig("training_performance.jpg")



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()