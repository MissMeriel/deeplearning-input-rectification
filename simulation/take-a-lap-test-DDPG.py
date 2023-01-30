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
    print(f"{device=}")
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"RLtest-resdec-straight-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/TestEnv.py", newdir)
    PATH = f"{newdir}/"

    from TestEnv import CarEnv
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=(3, 54, 96), model="DDPGMLPSingle", filepathroot=newdir, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance="BeamNG.researchINSTANCE2", port=64156, scenario="automation_test_track", road_id="8185", reverse=False,
                 base_model=model_filename, test_model=True, seg=None, transf="fisheye")
    # main(obs_shape=(3, 270, 480), scenario="hirochi_raceway", road_id="9039", seg=0, label="Rturn")
    # main(obs_shape=(3, 270, 480), scenario="west_coast_usa", road_id="12930", seg=None, label="Lturn")
    # main(obs_shape=(3, 270, 480), scenario="automation_test_track", road_id="8185", seg=None, label="straight")
    # main(obs_shape=(3, 270, 480), scenario="west_coast_usa", road_id="10988", seg=1, label="windy")
    start_time = time.time()
    from stable_baselines3 import DDPG

    model = DDPG.load("F:/RRL-results/RLtrainstraight-resdec-max200-0.05eval-1_25-10_34-3DPMZU/best_model", print_system_info=True)
    print(model.policy.actor_target.mu)
    print("\n\n", model.policy.actor.mu)
    # pred = model.predict(testinput)
    # print(f"{pred=}\t{setpoint=}\t{steer=}")
    start_time = time.time()
    obs = env.reset()
    episode_reward = 0
    done, crashed = False, False
    episodes = 0
    results = []
    while not done or not crashed:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, state = env.step(action)
        crashed = state.get('collision', True)
        if done or crashed:
            print("Done?", done, "Collision?", state.get('collision', True))
            ep_results = env.get_progress()
            results.append(ep_results)
            obs = env.reset()
            episodes += 1
        if episodes == 5:
            env.close()
            break
        # env.render()
    dists = []
    centerline_dists = []
    ep_count, frames_adjusted = 0, 0
    for r in results:
        dists.append(r['dist_travelled'])
        centerline_dists.extend(r['dist_from_centerline'])
        ep_count += r["total_steps"]
        frames_adjusted += r["frames_adjusted_count"]
    print(f"AVERAGE OVER {episodes} RUNS:"
          f"\n\tdist travelled:{(sum(dists) / len(dists)):1f}"
          f"\n\tdist from ctrline:{(sum(centerline_dists) / len(centerline_dists)):3f}"
          f"\n\tintervention rate:{frames_adjusted / ep_count:3f}")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()