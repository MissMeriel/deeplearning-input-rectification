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
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

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
    newdir = f"DIDIFIXIT-RLtrain-DDPGhuman-halfres-everystep-singleinput-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/DDPGHumanenv4.py", newdir)
    PATH = f"{newdir}/evalrew"

    from GAILenv4 import CarEnv
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), model="GAILMLPSingle", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents', beamnginstance="BeamNG.researchINSTANCE4",
                 # port=64156, scenario="west_coast_usa", road_id="12146", reverse=False, # outskirts with bus stop
                 # port=64356, scenario="automation_test_track", road_id="8185", reverse=False,
                 # port=64356, scenario="west_coast_usa", road_id="10784", reverse=False,
                 port=64756, scenario="hirochi_raceway", road_id="9039", reverse=False,
                 # port=64156, scenario="west_coast_usa", road_id="10673", reverse=False,
                 # port=64156, scenario="west_coast_usa", road_id="12930", reverse=False,
                 base_model=model_filename, test_model=False)

    env = CarEnv(image_shape=(3, 135, 240), model="GAILMLPSingle", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents', beamnginstance="BeamNG.researchINSTANCE5",
                 # port=64156, scenario="west_coast_usa", road_id="12146", reverse=False, # outskirts with bus stop
                 # port=64356, scenario="automation_test_track", road_id="8185", reverse=False,
                 # port=64356, scenario="west_coast_usa", road_id="10784", reverse=False,
                 port=64956, scenario="hirochi_raceway", road_id="9039", reverse=False,
                 # port=64156, scenario="west_coast_usa", road_id="10673", reverse=False,
                 # port=64156, scenario="west_coast_usa", road_id="12930", reverse=False,
                 base_model=model_filename, test_model=False)

    start_time = time.time()

    # IMPLEMENT GAIL
    # ref: https://imitation.readthedocs.io/en/latest/algorithms/gail.html
    # ref: https://imitation.readthedocs.io/en/latest/tutorials/3_train_gail.html
    rng = np.random.default_rng(0)

    expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
    expert.learn(1000)

    rollouts = rollout.rollout(
        expert,
        make_vec_env(
            "seals/CartPole-v0",
            n_envs=5,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            rng=rng,
        ),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
    )
    learner = PPO(env=env, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        env.observation_space,
        env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    gail_trainer.train(20000)
    rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Rewards:", rewards)

    # Create an evaluation callback with the same env, called every 10000 iterations
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=50, verbose=1)
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        callback_after_eval=stop_train_callback,
        n_eval_episodes=5,
        best_model_save_path=f"./{newdir}",
        log_path=f"./{newdir}",
        eval_freq=10000,
        verbose=1
    )
    callbacks.append(eval_callback)

    kwargs = {}
    kwargs["callback"] = callbacks
    # Train for a certain number of timesteps
    model.learn(total_timesteps=5e10, tb_log_name="DDPG-sb3_car_run_" + str(time.time()), **kwargs)

    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()