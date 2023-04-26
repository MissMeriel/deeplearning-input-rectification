import shutil
import string
import random
import matplotlib
import logging

import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes

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
    parser.add_argument("-e", "--evaluate", type=bool, default=False,
                        help="Evaluate mode (True for evaluate, False for train)")
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

import os
from stable_baselines3.common.monitor import Monitor


def run_RLtrain(obs_shape=(3, 135, 240), scenario="hirochi_raceway", road_id="9039", seg=None, label="Rturn", transf="fisheye",
                beamnginstance="BeamNG.research", port=64156, eval_eps=0.05):
    maxepisodes = 1000
    policytype = "CnnPolicy"
    if label == "winding":
        noise_sigma = 0.558
    else:
        noise_sigma = 0.000025
    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"

    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    newdir = f"F:/RRL-results/RLtrain-BWimg-SB31.6.2-{policytype}-{noise_sigma}-onpolicy-1kpunish-{label}-{transf}-max{maxepisodes}-{eval_eps}eval-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        shutil.copy(f"{__file__}", newdir)
        shutil.copy(f"C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/beamng_env_revisit.py", newdir)
    newdir_eval = f"F:/RRL-results/RLtrain-BWimg-SB31.6.2-{policytype}-{noise_sigma}-onpolicy-1kpunish-{label}-{transf}-max{maxepisodes}-{eval_eps}eval-EVAL-{timestr}-{randstr}"
    if not os.path.exists(newdir_eval):
        os.mkdir(newdir_eval)

    from beamng_env_revisit import CarEnv
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model="DQN", filepathroot=newdir, beamngpath='C:/Users/Meriel/Documents',
                     beamnginstance=beamnginstance, port=port, scenario=scenario, road_id=road_id, reverse=False,
                     base_model=model_filename, test_model=True, seg=seg, transf=transf, topo=label, eval_eps=eval_eps)
    env = Monitor(env, newdir)
    env_eval = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model="DQN", filepathroot=newdir_eval, beamngpath='C:/Users/Meriel/Documents',
                      beamnginstance='BeamNG.researchINSTANCE4', port=64956, scenario=scenario, road_id=road_id, reverse=False,
                      base_model=model_filename, test_model=True, seg=seg, transf=transf, topo=label, eval_eps=eval_eps)

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
        buffer_size=1000,
        max_grad_norm=10,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        device="cuda",
        tensorboard_log="./tb_logs_DQNrevisit/",
    )

    # Create an evaluation callback with the same env, called every 10000 iterations
    callbacks = []
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=maxepisodes, verbose=1)


    eval_callback = EvalCallback(
        env_eval,
        n_eval_episodes=1,

        best_model_save_path=newdir,
        log_path=newdir_eval,
        eval_freq=1000,
        verbose=1
    )
    callbacks.append(eval_callback)
    callbacks.append(callback_max_episodes)

    kwargs = {}
    kwargs["callback"] = callbacks
    print(f"{model.policy=}")
    # Train for a certain number of timesteps
    model.learn(
        total_timesteps=5e5, tb_log_name="dqn_v101_sb3_car_run_" + str(time.time()), **kwargs
    )
    print(f"Time to train: {(time.time()-start_time)/60:.1f} minutes")


def run_RL_test(obs_shape=(3, 135, 240), scenario="hirochi_raceway", road_id="9039", seg=None, label="Rturn", transf="fisheye",
                beamnginstance="BeamNG.research", port=64156, eval_eps=0.05):
    testdir = "RLtrain-SB31.6.2-CnnPolicy-0.558-onpolicy-1kpunish-winding-resdec-max1000-0.05eval-4_4-14_42-VC4ZEP"
    # randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    # localtime = time.localtime()
    # timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    testdir = f"F:/RRL-results/{testdir}-TESTDIR"
    if not os.path.exists(testdir):
        os.mkdir(testdir)
        shutil.copy(f"{__file__}", testdir)
        shutil.copy(f"C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/beamng_env_revisit.py", testdir)

    model_filename = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"

    from beamng_env_revisit import CarEnv
    env = CarEnv(image_shape=(3, 135, 240), obs_shape=obs_shape, model="DQN", filepathroot=testdir, beamngpath='C:/Users/Meriel/Documents',
                     beamnginstance=beamnginstance, port=port, scenario=scenario, road_id=road_id, reverse=False,
                     base_model=model_filename, test_model=True, seg=seg, transf=transf, topo=label, eval_eps=eval_eps)
    model = DQN.load(f"F:/RRL-results/{testdir.replace('-TESTDIR', '').replace('F:/RRL-results/', '')}/best_model", print_system_info=True)
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
        # crashed = state.get('collision', True)
        crashed = env.car_state["damage"]["damage"] > 1
        if done or crashed:
            ep_results = env.get_progress()
            results.append(ep_results)
            obs = env.reset()
            episodes += 1
        if episodes == 5:
            break

    dists, centerline_dists, rewards = [], [], []
    ep_count, frames_adjusted = 0, 0
    for r in results:
        dists.append(r['dist_travelled'])
        centerline_dists.extend(r['dist_from_centerline'])
        ep_count += r["total_steps"]
        frames_adjusted += r["frames_adjusted_count"]
        rewards.append(sum(r["rewards"]))
    print(f"AVERAGE OVER {episodes} RUNS:"
          f"\n\tdist travelled:{(sum(dists) / len(dists)):.1f}"
          f"\n\tdist from ctrline:{(sum(centerline_dists) / len(centerline_dists)):.3f}"
          f"\n\tintervention rate:{frames_adjusted / ep_count:.3f}"
          f"\n\tavg reward:{sum(rewards) / len(rewards):.3f}")
    env.close()

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if args.transformation == "resinc":
        obs_shape = (3, 270, 480)
    elif args.transformation == "resdec":
        # obs_shape = (3, 54, 96)
        obs_shape = (1, 81, 144)
    elif args.transformation == "fisheye" or args.transformation == "depth":
        obs_shape = (3, 135, 240)
    elif args.transformation == "None":
        obs_shape = (3, 135, 240)

    if args.scenario == "Rturn":
        run_RLtrain(obs_shape=obs_shape, scenario="hirochi_raceway", road_id="9039", seg=0, label="Rturn", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "Lturn":
        run_RLtrain(obs_shape=obs_shape, scenario="west_coast_usa", road_id="12930", seg=None, label="Lturn", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "straight":
        run_RLtrain(obs_shape=obs_shape, scenario="automation_test_track", road_id="8185", seg=None, label="straight", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
    elif args.scenario == "winding":
        if args.evaluate:
            run_RL_test(obs_shape=obs_shape, scenario="west_coast_usa", road_id="10988", seg=1, label="winding", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)
        else:
            run_RLtrain(obs_shape=obs_shape, scenario="west_coast_usa", road_id="10988", seg=1, label="winding", transf=args.transformation, beamnginstance=args.beamnginstance, port=args.port, eval_eps=args.evaleps)