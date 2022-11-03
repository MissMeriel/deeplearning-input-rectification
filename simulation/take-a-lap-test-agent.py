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
from beamng_env import CarEnv
from stable_baselines3.common.env_checker import check_env

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
    newdir = f"RLtest-v1021-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/beamng_env.py", newdir)
    PATH = f"{newdir}/SB3-v1021-DQN6action-rew5000"
    # env = CarEnv(image_shape=(1, 84, 150), model="DQN", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents')

    # setup v102
    # env = CarEnv(image_shape=(1, 84, 150), model="DQN", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents',
    #              beamnginstance="BeamNG.researchINSTANCE2", port=64756, road_id="13091", reverse=True)
    # model = DQN.load("RLtrain-v102badmodel-10_23-19_18-TIFKMT/best_model")

    # setup v1033
    # env = CarEnv(image_shape=(1, 84, 150), model="DQN", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents',
    #              beamnginstance="BeamNG.researchINSTANCE2", port=64756, road_id="12146", reverse=False)
    # model = DQN.load("RLtrain-v1033badmodel-10_23-19_16-2WKWKB/best_model")

    # setup v1021
    env = CarEnv(image_shape=(1, 84, 150), model="DQN", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance="BeamNG.researchINSTANCE2", port=64756, road_id="13091", reverse=False)
    model = DQN.load("RLtrain-v1021badmodel-10_25-11_12-8EEXOJ/best_model")

    check_env(env)
    # test model loaded properly
    # testinput = np.random.random((1, 84, 150))
    # pred = model.predict(testinput)
    # setpoint, steer = get_action(pred)
    # print(f"{pred=}\t{setpoint=}\t{steer=}")
    start_time = time.time()



    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    done, crashed = False, False
    while not done or not crashed:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, state = env.step(action)
        episode_reward += reward
        crashed = state.get('collision', True)
        print(f"{action=}\t{done=}\t{crashed=}")
        if done or crashed:
            print("Reward:", episode_reward, "Collision?", state.get('collision', True))
            episode_reward = 0.0
            obs = env.reset()

def get_action(action):
    action = action[0].item()
    if action == 0:
        steer = 0
    elif action == 1:
        steer = 0.5
    elif action == 2:
        steer = -0.5
    elif action == 3:
        steer = 0.25
    elif action == 4:
        steer = -0.25
    elif action == 5:
        steer = 1.0
    else:
        steer = -1.0

    if abs(steer) > 0.2:
        setpoint = 30
    else:
        setpoint = 40

    return setpoint, steer

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()