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

from stable_baselines3.common.callbacks import BaseCallback


class PiXCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    Docs: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, verbose=0):
        super(PiXCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        # self.env = env CONTAINS: logger, model DQN, n_calls, num_timesteps, training_env=DummyVecEnv

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print(f"TRAINING START")
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # evaluation = None
        # for monitor in self.training_env.envs:
        #     # print(f"{monitor.env.state.keys()=}")
        #     evaluation = monitor.env.evaluator()
        # if evaluation == 0:
        #     return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print(f"POLICY UPDATE TRIGGERED")
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

def main():
    global base_filename, default_color, default_scenario, road_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"RLtrain-DDPGhuman-fov70-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/DDPGHumanenv.py", newdir)
    PATH = f"{newdir}/rew2to4minusdist"

    from DDPGHumanenv import CarEnv
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), model="DQN", filepathroot=PATH, beamngpath='C:/Users/Meriel/Documents', beamnginstance="BeamNG.research",
                 # port=64156, scenario="west_coast_usa", road_id="12146", reverse=False, # outskirts with bus stop
                 # port=64356, scenario="automation_test_track", road_id="8185", reverse=False,
                 # port=64356, scenario="west_coast_usa", road_id="10784", reverse=False,
                 port=64356, scenario="hirochi_raceway", road_id="9039", reverse=False,
                 # port=64156, scenario="west_coast_usa", road_id="10673", reverse=False,
                 # port=64156, scenario="west_coast_usa", road_id="12930", reverse=False,
                 base_model=model_filename)
    from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    start_time = time.time()
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=0.00025,
        verbose=1,
        batch_size=32,
        train_freq=4,
        learning_starts=2000000,
        buffer_size=5000,
        device="cuda",
        tensorboard_log=f"./{newdir}/tb_logs_DDPG/",
    )
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
    custom_callback = PiXCallback(
        env, verbose=1
    )
    callbacks.append(custom_callback)

    kwargs = {}
    kwargs["callback"] = callbacks
    # Train for a certain number of timesteps
    model.learn(total_timesteps=5e5, tb_log_name="DDPG-sb3_car_run_" + str(time.time()), **kwargs)

    model.save(f"{PATH}-DDPG_sb3_car_policy.pt")
    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()