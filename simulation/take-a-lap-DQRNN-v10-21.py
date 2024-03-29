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

'''
Actor-Critic Learner with continuous action space
actor-critic model based on Microsoft example:
https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/22-DeepRL/CartPole-RL-TF.ipynb
DQN from microsoft example:
https://microsoft.github.io/AirSim/reinforcement_learning/
'''

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def parse_args():
    return

def main():
    global base_filename, default_color, default_scenario, road_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"RLtrain-v1021badmodel-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/beamng_env.py", newdir)
    PATH = f"{newdir}/SB3-v1021badmodel-DQN6action-rew5000"
    import gym
    from beamng_env import CarEnv
    env = CarEnv(image_shape=(1, 84, 150), model="DQN", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents',
                 # beamnginstance="BeamNG.researchINSTANCE3", port=64356, road_id="13091", reverse=False)
                 beamnginstance="BeamNG.researchINSTANCE3", port=64356, road_id="13306", reverse=False)

    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    start_time = time.time()

    # Initialize RL algorithm type and parameters
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=0.00025,
        verbose=1,
        batch_size=32,
        train_freq=4,
        target_update_interval=1000,
        learning_starts=200000,
        buffer_size=5000,
        max_grad_norm=10,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        device="cuda",
        tensorboard_log=f"./{newdir}/tb_logs_v1021-badmodel/",
    )
    model.load("RLtrain-v102badmodel-10_21-15_17-HGT33S/best_model")
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
        total_timesteps=5e5, tb_log_name="dqn_v1021badmodel_sb3_car_run_" + str(time.time()), **kwargs
    )

    # Save policy weights
    model.save(f"{PATH}-dqn_v1021badmodel_sb3_car_policy.pt")
    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()