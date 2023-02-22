import shutil
import os
import string
import random
import numpy as np
import logging
import time
import sys
# import torch
# import cv2
# import matplotlib

# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
# from stable_baselines3 import DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback

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
    parser.add_argument('-te', '--testeps', type=float, default=0.05,
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

def run_RLtrain(obs_shape=(3, 135, 240), scenario="hirochi_raceway", road_id="9039", seg=None, label="Rturn", transf="fisheye",
                beamnginstance="BeamNG.research", port=64156, eval_eps=0.05):
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"F:/RRL-results/RLtrain-onpolicyeval-{label}-{transf}-max200-{eval_eps}eval-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/DDPGHumanenv4writedata.py", newdir)
    newdir_eval = f"F:/RRL-results/RLtrain-onpolicyeval-{label}-{transf}-max200-{eval_eps}eval-eval-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir_eval)

    from DDPGHumanenv4writedata import CarEnv
    policytype = "MlpPolicy"
    noise_sigma = 0.000025
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model=f"DDPG{policytype}", filepathroot=newdir, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance=beamnginstance, port=port, scenario=scenario, road_id=road_id, reverse=False,
                 base_model=model_filename, test_model=False, seg=seg, transf=transf, topo=label, eval_eps=eval_eps)

    eval_env = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model=f"DDPG{policytype}", filepathroot=newdir_eval, beamngpath='C:/Users/Meriel/Documents',
                 beamnginstance='BeamNG.researchINSTANCE4', port=64956, scenario=scenario, road_id=road_id, reverse=False,
                 base_model=model_filename, test_model=True, seg=seg, transf=transf, topo=label, eval_eps=eval_eps)

    start_time = time.time()
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))
    model = DDPG(
        policytype,
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
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=500, verbose=1)
    callbacks = []
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        best_model_save_path=f"{newdir}",
        log_path=f"{newdir}",
        eval_freq=10000,
        verbose=1
    )
    # everyN_callback = EveryNTimesteps(n_steps=50, callback=callback_max_episodes)
    # callbacks.append(everyN_callback)
    callbacks.append(eval_callback)
    callbacks.append(callback_max_episodes)
    kwargs = {}
    kwargs["callback"] = callbacks
    model.learn(total_timesteps=5e10, tb_log_name="DDPG-sb3_car_run_" + str(time.time()), **kwargs)
    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    args = parse_args()

    if args.transformation == "resinc":
        obs_shape = (3, 270, 480)
    elif args.transformation == "resdec":
        obs_shape = (3, 54, 96)
    elif args.transformation == "fisheye" or args.transformation == "depth":
        obs_shape = (3, 135, 240)

    if args.scenario == "Rturn":
        run_RLtrain(obs_shape=obs_shape, scenario="hirochi_raceway", road_id="9039", seg=0, label="Rturn", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "Lturn":
        run_RLtrain(obs_shape=obs_shape, scenario="west_coast_usa", road_id="12930", seg=None, label="Lturn", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "straight":
        run_RLtrain(obs_shape=obs_shape, scenario="automation_test_track", road_id="8185", seg=None, label="straight", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "winding":
        run_RLtrain(obs_shape=obs_shape, scenario="west_coast_usa", road_id="10988", seg=1, label="winding", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
