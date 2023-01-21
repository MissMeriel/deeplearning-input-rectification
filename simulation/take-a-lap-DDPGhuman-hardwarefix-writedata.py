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

def main(obs_shape=(3, 135, 240), scenario="hirochi_raceway", road_id="9039", seg=None, label="Rturn"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"F:/RRL-results/RLtrain{label}-resinc-max200-0.05eval-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/DDPGHumanenv4writedata.py", newdir)

    from DDPGHumanenv4writedata import CarEnv
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model="DDPGMLPSingle", filepathroot=newdir, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance="BeamNG.research", port=64156, scenario=scenario, road_id=road_id, reverse=False,
                 base_model=model_filename, test_model=False, seg=seg)

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
        tensorboard_log=f"{newdir}/tb_logs_DDPG/",
    )
    # Create an evaluation callback with the same env, called every 10000 iterations
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, EveryNTimesteps
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=200, verbose=1)
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_max_episodes,
        callback_after_eval=callback_max_episodes,
        n_eval_episodes=1,
        best_model_save_path=f"{newdir}",
        log_path=f"{newdir}",
        eval_freq=20000,
        verbose=1
    )
    #everyN_callback = EveryNTimesteps(n_steps=50, callback=callback_max_episodes)
    callbacks.append(eval_callback)
    callbacks.append(callback_max_episodes)
    #callbacks.append(everyN_callback)
    kwargs = {}
    kwargs["callback"] = callbacks
    # Train for a certain number of timesteps
    model.learn(total_timesteps=5e10, tb_log_name="DDPG-sb3_car_run_" + str(time.time()), **kwargs)

    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")
    env.close()

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    #main(obs_shape=(3, 270, 480), scenario="hirochi_raceway", road_id="9039", seg=0, label="Rturn")
    main(obs_shape=(3, 270, 480), scenario="west_coast_usa", road_id="12930", seg=None, label="Lturn")
    main(obs_shape=(3, 270, 480), scenario="automation_test_track", road_id="8185", seg=None, label="straight")
    main(obs_shape=(3, 270, 480), scenario="west_coast_usa", road_id="10988", seg=1, label="windy")