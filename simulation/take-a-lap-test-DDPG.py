import shutil
import os
import string
import random
import time
import torch
import logging
import sys
# from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
# from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
#
# import scipy.misc
# import copy
# import torch
# import torchvision.transforms as transforms
# import statistics, math
# from scipy.spatial.transform import Rotation as R
# from scipy import interpolate
import PIL
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
# from stable_baselines3 import DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback
# import gym
# from beamng_env import CarEnv
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
    import argparse
    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument("-t", '--transformation', type=str,
                        help='transformation type (resinc, resdec, fisheye, or depth)')
    parser.add_argument('-s', '--scenario', type=str,
                        help='road segment to learn (Rturn, Lturn, straight, or winding)')
    parser.add_argument('-d', '--path2src', type=str, default="C:/Users/Meriel/Documents",
                        help='road segment to learn (Rturn, Lturn, straight, or winding)')
    parser.add_argument('-b', '--beamnginstance', type=str, default="BeamNG.research",
                        help='parent directory of BeamNG instance')
    parser.add_argument('-p', '--port', type=int, default=64156,
                        help='port to communicate with BeamNG simulator (try 64156, 64356, 64556...)')
    parser.add_argument('-ee', '--evaleps', type=float, default=0.05,
                        help='Evaluation epsilon above which the expert intervenes')
    parser.add_argument('-te', '--testeps', type=float, default=0.01,
                        help='Test epsilon above which the RL agent intervenes')
    args = parser.parse_args()
    print(args.transformation, args.scenario, args.path2src, args.beamnginstance, args.port, args.evaleps, args.testeps)
    return args

args = parse_args()

sys.path.append(f'{args.path2src}/GitHub/DAVE2-Keras')
sys.path.append(f'{args.path2src}/GitHub/deeplearning-input-rectification/models/')
sys.path.append(f'{args.path2src}/GitHub/deeplearning-input-rectification/simulation/')
sys.path.append(f'{args.path2src}/GitHub/BeamNGpy')
sys.path.append(f'{args.path2src}/GitHub/BeamNGpy/src/')
sys.path.append(f'{args.path2src}/GitHub/DPT/')

def main(obs_shape=(3, 135, 240), scenario="hirochi_raceway", road_id="9039", seg=None, label="Rturn", transf="fisheye",
                beamnginstance="BeamNG.research", port=64156, eval_eps=0.05):
    # global base_filename, default_color, default_scenario, road_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device=}")
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    testdir = "RLtrain-MlpPolicy-0.558-onpolicyall-winding-fisheye-max200-0.05eval-2_24-20_22-JALLIT"
    newdir = f"RLtest-{testdir}-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"{os.getcwd()}/TestEnv.py", newdir)

    if "windy" in testdir:
        scenario = "west_coast_usa"; road_id = "10988"; seg = 1
    elif "straight" in testdir:
        scenario = "automation_test_track"; road_id = "8185"; seg = None
    elif "Rturn" in testdir:
        scenario="hirochi_raceway"; road_id="9039"; seg=0
    elif "Lturn" in testdir:
        scenario = "west_coast_usa"; road_id = "12930"; seg = None

    if "fisheye" in testdir:
        obs_shape = (3, 135, 240); transf = "fisheye"
    elif "resdec" in testdir:
        obs_shape = (3, 54, 96); transf = "resdec"
    elif "resinc" in testdir:
        obs_shape = (3, 270, 480); transf = "resinc"
    else:
        obs_shape = (3, 135, 240); transf = "None"

    from TestEnv import CarEnv
    policytype = "MlpPolicy"
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model=f"DDPG{policytype}", filepathroot=newdir, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance=beamnginstance, port=port, scenario=scenario, road_id=road_id, reverse=False,
                 base_model=model_filename, test_model=True, seg=seg, transf=transf, topo=label, eval_eps=eval_eps)
    start_time = time.time()
    from stable_baselines3 import DDPG

    model = DDPG.load(f"F:/RRL-results/{testdir}/best_model", print_system_info=True)
    # model = DDPG.load(f"C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/{testdir}/best_model", print_system_info=True)
    # print(model.policy.actor_target.mu)
    # print("\n\n", model.policy.actor.mu)
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
            ep_results = env.get_progress()
            results.append(ep_results)
            obs = env.reset()
            episodes += 1
        if episodes == 5:
            break

    dists, centerline_dists = [], []
    ep_count, frames_adjusted = 0, 0
    for r in results:
        dists.append(r['dist_travelled'])
        centerline_dists.extend(r['dist_from_centerline'])
        ep_count += r["total_steps"]
        frames_adjusted += r["frames_adjusted_count"]
    print(f"AVERAGE OVER {episodes} RUNS:"
          f"\n\tdist travelled:{(sum(dists) / len(dists)):.1f}"
          f"\n\tdist from ctrline:{(sum(centerline_dists) / len(centerline_dists)):.3f}"
          f"\n\tintervention rate:{frames_adjusted / ep_count:.3f}")
    env.close()

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if args.transformation == "resinc":
        obs_shape = (3, 270, 480)
    elif args.transformation == "resdec":
        obs_shape = (3, 54, 96)
    elif args.transformation == "fisheye" or args.transformation == "depth":
        obs_shape = (3, 135, 240)

    if args.scenario == "Rturn":
        main(obs_shape=obs_shape, scenario="hirochi_raceway", road_id="9039", seg=0, label="Rturn", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "Lturn":
        main(obs_shape=obs_shape, scenario="west_coast_usa", road_id="12930", seg=None, label="Lturn", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "straight":
        main(obs_shape=obs_shape, scenario="automation_test_track", road_id="8185", seg=None, label="straight", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "winding":
        main(obs_shape=obs_shape, scenario="west_coast_usa", road_id="10988", seg=1, label="winding", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
