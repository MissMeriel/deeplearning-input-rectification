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

def main():
    global base_filename, default_color, default_scenario, road_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"RLtrain-max200epi-DDPGhuman-0.05evaleps-bigimg-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/DDPGHumanenv4.py", newdir)
    PATH = f"{newdir}/evalexpert"

    from DDPGHumanenv4 import CarEnv
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=(3, 270, 480), model="DDPGMLPSingle", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance="BeamNG.research", port=64156, scenario="hirochi_raceway", road_id="9039", reverse=False,
                 base_model=model_filename, test_model=False, seg=0)

    start_time = time.time()
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.000025 * np.ones(n_actions))
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=0.001,
        verbose=1,
        batch_size=32,
        train_freq=1,
        learning_starts=2000000,
        buffer_size=50000,
        device="cuda",
        tensorboard_log=f"./{newdir}/tb_logs_DDPG/",
    )
    # Create an evaluation callback with the same env, called every 10000 iterations
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnMaxEpisodes
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=50, verbose=1)
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=200, verbose=1)
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        callback_after_eval=callback_max_episodes,
        n_eval_episodes=5,
        best_model_save_path=f"./{newdir}",
        log_path=f"./{newdir}",
        eval_freq=10000,
        verbose=1
    )
    callbacks.append(eval_callback)
    callbacks.append(callback_max_episodes)
    kwargs = {}
    kwargs["callback"] = callbacks
    # Train for a certain number of timesteps
    model.learn(total_timesteps=5e10, tb_log_name="DDPG-sb3_car_run_" + str(time.time()), **kwargs)

    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()