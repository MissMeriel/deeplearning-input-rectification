import shutil
import os
import string
import cv2
import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import logging
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

import scipy.misc
import copy
import torch
import torchvision.transforms as transforms
import statistics, math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import PIL
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


'''
Generative Adversarial Imitation Learning (GAIL)
SB3 Docs: https://stable-baselines3.readthedocs.io/en/v0.11.1/guide/imitation.html
Example: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py
'''

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class CustomNetwork(nn.Module):
    """
    See documentation page 56: https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor␣
    ˓→(e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy␣
    ˓→network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value␣
    ˓→network
    """

    def __init__(self,
                orig_expert):
        super(CustomNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # self.latent_dim_pi = last_layer_dim_pi
        # self.latent_dim_vf = last_layer_dim_vf
        # Policy network
        self.policy_net = orig_expert #nn.Sequential(nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU())
        # Value network
        self.value_net = None #nn.Sequential(nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU())

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified␣
        ˓→network.
        If all layers are shared, then ``latent_policy == latent_value``
        """
        # print(f"{type(features)=}")
        # print(f"{features.shape=}")
        transform = T.Compose([T.ToTensor()])
        features = transform(features[0])[None]
        # print(f"{features.shape=}")
        prediction = self.policy_net(features.permute(0,2,3,1))
        print(f"{prediction=}")
        return prediction.detach().item(), None

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return None #self.value_net(features)


def parse_args():
    return

def main():
    global base_filename, default_color, default_scenario, road_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"deletelater-policymonitorGAIL1-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/beamng_env.py", newdir)
    PATH = f"{newdir}/policymonitor1"
    import gym
    from dist_shift_env import FaultInducingEnv
    env = FaultInducingEnv(image_shape=(3, 135, 240), filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents', beamnginstance="BeamNG.tech",
                 # port=64156, road_id="12146", reverse=False)
                 # port=64156, road_id="12667", reverse=False)
                 # port=64156, road_id="10784", reverse=False)
                 # port=64156, road_id="10673", reverse=False)
                 port=64156, road_id="15425", reverse=False, newdir=newdir)
    from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    start_time = time.time()

    print("Loading expert...")
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    orig_expert = torch.load(model_filename, map_location=torch.device('cpu')).eval()
    expert = CustomNetwork(orig_expert)
    rng = np.random.default_rng(0)
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)
    # Create an evaluation callback with the same env, called every 10000 iterations
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path=f"./{newdir}",
        log_path=f"./{newdir}",
        eval_freq=10000,
    )
    callbacks.append(eval_callback)

    kwargs = {}
    kwargs["callback"] = callbacks
    # Train for a certain number of timesteps
    model.learn(
        total_timesteps=5e5, tb_log_name="GAIL1-sb3_car_run_" + str(time.time()), **kwargs
    )

    # Save policy weights
    model.save(f"{PATH}-GAIL1-sb3_car_policy.pt")
    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()