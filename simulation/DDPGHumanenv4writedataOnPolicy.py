import os
import cv2
import random
import numpy as np
import string
from beamngpy import BeamNGpy, Scenario, Vehicle, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
import PIL
# import scipy.misc
# import copy
import time
import logging
import torchvision.transforms as transforms
import math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import time
import torch
import torchvision.transforms as T
from wand.image import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

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

import gym
from gym import spaces

# noinspection PyUnreachableCode
class CarEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape=(3, 135, 240), obs_shape=(3, 135, 240), model="DDPG", filepathroot=".", beamngpath="C:/Users/Meriel/Documents",
                 beamnginstance="BeamNG.research", port=64356, scenario="west_coast_usa", road_id="12146", reverse=False,
                 base_model=None, test_model=False, seg=None, transf=None, topo="straight", eval_eps=0.05):
        super(CarEnv, self).__init__()
        self.start_time = time.time()
        self.transf = transf
        self.topo=topo
        self.eval_eps=eval_eps
        self.test_eps=eval_eps
        self.f = None
        self.ep_dir = None
        self.seg = seg
        self.test_model = True
        self.action_space = spaces.Box(low=-2, high=2, shape=(1, 1), dtype=np.float32)
        self.transform = T.Compose([T.ToTensor()])
        self.episode_steps, self.frames_adjusted = 0, 0
        self.steer_integral, self.steer_prev_error = 0., 0.
        self.beamngpath = beamngpath
        if base_model is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.base_model = torch.load(base_model, map_location=device).eval() #torch.load(base_model).to(device)
        self.default_scenario = scenario
        self.road_id = road_id
        self.reverse = reverse
        self.integral, self.prev_error = 0.0, 0.0
        self.steer_integral, self.steer_prev_error = 0.0, 0.0
        self.overall_throttle_setpoint = 40
        self.setpoint = self.overall_throttle_setpoint
        self.lanewidth = 3.75  # 2.25
        self.track_cam_pos = None
        self.centerline = []
        self.centerline_interpolated = []
        self.roadleft = []
        self.roadright = []
        self.trajectories, self.current_trajectory = [], []
        self.all_rewards, self.current_rewards, self.all_intervention_rates = [], [], []
        self.actions, self.base_model_inf = [], []
        self.image_shape = image_shape
        self.steps_per_sec = 15
        self.runtime = 0.0
        self.episode = -1
        random.seed(1703)
        self.model = model
        self.filepathroot = filepathroot
        beamng = BeamNGpy('localhost', port=port, home=f'{self.beamngpath}/BeamNG.research.v1.7.0.1', user=f'{self.beamngpath}/{beamnginstance}')

        self.scenario = Scenario(self.default_scenario, 'RL_Agent-train')
        self.vehicle = Vehicle('ego_vehicle', model="hopper", licence='EGO', color="green")
        self.obs_shape = obs_shape

        self.setup_sensors()
        self.spawn = self.spawn_point()

        self.scenario.add_vehicle(self.vehicle, pos=self.spawn['pos'], rot=None, rot_quat=self.spawn['rot_quat'])

        self.scenario.make(beamng)
        self.bng = beamng.open(launch=True)
        self.bng.set_deterministic()
        self.bng.set_steps_per_second(self.steps_per_sec)
        self.bng.load_scenario(self.scenario)
        self.create_ai_line_from_road_with_interpolation()
        self.bng.start_scenario()
        self.bng.pause()
        assert self.vehicle.skt
        ###########################################################################
        self.observation_space = spaces.Box(0, 255, shape=self.obs_shape, dtype=np.uint8)
        self.state = {
            # "image": np.zeros(self.obs_shape[1:]),
            # "prev_image": np.zeros(self.obs_shape[1:]),
            "pose": np.zeros(3),
            "prev_pose": np.zeros(3),
            "collision": False,
        }
        self.image = None
        self.car_state = None


    def _get_obs(self):
        self.state["prev_pose"] = self.vehicle.state["pos"]
        self.vehicle.update_vehicle()
        sensors = self.bng.poll_sensors(self.vehicle)
        self.current_trajectory.append(self.vehicle.state["pos"])
        self.car_state = sensors
        self.state["pose"] = self.vehicle.state["pos"]
        self.state["collision"] = sensors["damage"]["damage"] > 1
        image = np.array(sensors['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
        #if self.obs_shape != self.image_shape:
        #    image = np.array(sensors['quarterres_cam']['colour'].convert('RGB'), dtype=np.uint8)
        image = image.reshape(self.obs_shape)

        # self.state["prev_image"] = self.state["image"]
        # self.state["image"] = image
        if self.episode_steps == 0:
            return np.zeros(self.obs_shape)
        else:
            return image

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.array(gray, dtype=np.uint8)

    def fisheye(self, image):
        with Image.from_array(image) as img:
            img.virtual_pixel = 'transparent'
            img.distort('barrel', (0.1, 0.0, -0.05, 1.0))
            return np.array(img)

    def fisheye_inv(self, image):
        with Image.from_array(image) as img:
            img.virtual_pixel = 'transparent'
            img.distort('barrel_inverse', (0.0, 0.0, -0.5, 1.5))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

    def step(self, action):
        if self.default_scenario == "hirochi_raceway" and self.road_id == "9039" and self.seg == 0:
            cutoff_point = [368.466, -206.154, 43.8237]
        elif self.default_scenario == "automation_test_track" and self.road_id == "8185":
            cutoff_point = [56.017, -272.890, 120.567]  # 100m
        elif self.default_scenario == "west_coast_usa" and self.road_id == "12930":
            cutoff_point = [-296.560, -730.234, 136.713]
        elif self.default_scenario == "west_coast_usa" and self.road_id == "10988" and self.seg == 1:
            # cutoff_point = [843.507, 6.244, 147.019] # middle
            cutoff_point = [843.611, 6.588, 147.018]  # late
        else:
            cutoff_point = [601.547, 53.4482, 43.29]
        if self.test_model:
            obs = self._get_obs()
            kph = self.ms_to_kph(self.car_state['electrics']['wheelspeed'])
            dt = (self.car_state['timer']['time'] - self.start_time) - self.runtime
            # curr_steering = self.car_state['electrics']['steering_input']
            self.runtime = self.car_state['timer']['time'] - self.start_time
            image = np.array(self.car_state['front_cam']['colour'].convert('RGB'), dtype=np.uint8)

            if self.transf == "fisheye":
                image = self.fisheye_inv(image)
            image = cv2.resize(image, (self.image_shape[2], self.image_shape[1]))
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("true image", img)
            cv2.waitKey(1)

            features = self.transform(image)[None]
            base_model_inf = self.base_model(features).item()
            self.base_model_inf.append(base_model_inf)

            # if abs(action.item()) < self.test_eps:
            #     steer = float(base_model_inf)
            #     blackedout = np.zeros(image.shape)  # BLACK
            #     cv2.imshow("action image", blackedout)
            #     cv2.waitKey(1)
            # else:
            steer = float(base_model_inf + action.item())
            self.frames_adjusted += 1
            blackedout = np.ones(image.shape)
            blackedout[:, :, :2] = blackedout[:, :, :2] * 0  # RED
            cv2.imshow("action image", blackedout)
            cv2.waitKey(1)

            # print(f"DDPG action={action.item():.3f}, base_model={base_model_inf:.3f}, steer={steer:.3f}")
            if abs(steer) > 0.15:
                self.setpoint = 35
            else:
                self.setpoint = 40
            throttle = self.throttle_PID(kph, dt)

            self.vehicle.control(throttle=throttle, steering=steer, brake=0.0)
            self.bng.step(1, wait=True)

            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()

            # expert_action, cartocl_theta_deg = self.get_expert_action()
            # reward = self.calc_reward(base_model_inf + action.item(), expert_action)
            reward = self.calc_reward_onpolicy(outside_track, distance_from_center)

            self.episode_steps += 1
            # print(outside_track, self.state["collision"], (self.distance(self.state["pose"][:2], cutoff_point[:2]) < 12))
            done = outside_track or self.state["collision"] or (
                        self.distance(self.state["pose"][:2], cutoff_point[:2]) < 12)
            self.current_rewards.append(reward)
            return obs, reward, done, self.state

        else:
            obs = self._get_obs()
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
            kph = self.ms_to_kph(self.car_state['electrics']['wheelspeed'])
            dt = (self.car_state['timer']['time'] - self.start_time) - self.runtime
            curr_steering = self.car_state['electrics']['steering_input']
            self.runtime = self.car_state['timer']['time'] - self.start_time
            image = np.array(self.car_state['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
            if self.transf == "fisheye":
                image = self.fisheye_inv(image)
            elif "resdec" in self.transf or "resinc" in self.transf:
                #image = image.resize((240, 135))
                #image = np.array(image)
                image = cv2.resize(np.array(image), (240,135))
            elif "depth" in self.transf:
                image_seg = self.car_state['front_cam']['annotation'].convert('RGB')
            # image = cv2.resize(image, (self.image_shape[2], self.image_shape[1]))
            cv2.imshow("true image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            features = self.transform(image)[None]
            base_model_inf = self.base_model(features).item()
            self.base_model_inf.append(base_model_inf)
            # expert_action, cartocl_theta_deg = self.get_expert_action()
            expert_action = -leftrightcenter * (distance_from_center / 8)
            if self.topo == "Rturn" or self.topo == "Lturn":
                expert_action = -leftrightcenter * (distance_from_center)
            # evaluation = self.evaluator(outside_track, distance_from_center, leftrightcenter, segment_shape, base_model_inf + action.item())
            evaluation = abs(expert_action - (base_model_inf + action.item())) < self.eval_eps

            if evaluation:
                taken_action = base_model_inf + action.item()
                blackedout = np.ones(image.shape)
                blackedout[:, :, :2] = blackedout[:, :, :2] * 0
                cv2.imshow("action image", blackedout)  # red
                cv2.waitKey(1)
                self.steer_prev_error, self.steer_integral = 0., 0.
            else:
                taken_action = self.steering_PID(curr_steering, expert_action, dt)
                cv2.imshow("action image", np.zeros(image.shape))  # black
                cv2.waitKey(1)
                self.frames_adjusted += 1

            self.steer_prev_error = expert_action - taken_action

            if abs(taken_action) > 0.15:
                self.setpoint = 35
            else:
                self.setpoint = 40
            throttle = self.throttle_PID(kph, dt)
            self.vehicle.control(throttle=throttle, steering=taken_action, brake=0.0)
            self.bng.step(1, wait=True)
            self.vehicle.update_vehicle()
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
            reward = self.calc_reward(base_model_inf + action.item(), expert_action)

            position = str(self.state['pose']).replace(",", " ")
            orientation = str(self.vehicle.state['dir']).replace(",", " ")
            img_filename = f"{self.ep_dir}/sample-ep{self.episode:03d}-{self.episode_steps:05d}.jpg"
            self.car_state['front_cam']['colour'].convert('RGB').save(img_filename, "JPEG")
            self.f.write(f"{img_filename.split('/')[-1]},{taken_action},{position},{orientation},{kph},{self.car_state['electrics']['steering']},{throttle}\n")

            self.episode_steps += 1
            done = outside_track or self.state["collision"] or (self.distance(self.state["pose"][:2], cutoff_point[:2]) < 12)
            self.current_rewards.append(reward)
            # print(f"STEP() \n\troad_seg {theta_deg=:.1f}\t car-to-CL theta_deg={cartocl_theta_deg:.1f}\n\texpert_action={expert_action:.3f}\t\t{base_model_inf=:.3f}\n\t{reward=:.1f}\n\t{done=}\t{outside_track=}\tcollision={self.state['collision']}")
            return obs, reward, done, self.state


    def calc_reward(self, agent_action, expert_action):
        if self.state["collision"]:
            return -5000
        if abs(expert_action - agent_action) < self.eval_eps:
            return 1
        else:
            return -abs(expert_action - agent_action)

    def calc_reward_onpolicy(self, outside_track, distance_from_center):
        if outside_track or self.state["collision"]:
            return -5000
        else:
            return math.pow(2, 4.0-distance_from_center)

    def close(self):
        self.f.close()
        self.bng.close()

    def get_progress(self):
        dist = self.get_distance_traveled(self.current_trajectory)
        dist_from_centerline = []
        for i in self.current_trajectory:
            distance_from_centerline = self.dist_from_line(self.centerline_interpolated, i)
            dist_from_centerline.append(min(distance_from_centerline))

        summary = {
            "episode": self.episode,
            "rewards": self.current_rewards,
            "trajectory": self.current_trajectory,
            "total_steps": self.episode_steps,
            "frames_adjusted_count": self.frames_adjusted,
            "dist_from_centerline": dist_from_centerline,
            "image_shape": self.image_shape,
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "dist_travelled": dist
        }
        return summary

    def reset(self):
        print(f"\n\n\nRESET()")
        if self.episode > -1 and len(self.current_trajectory) > 0 and self.episode_steps > 0:
            print(f"SUMMARY OF EPISODE #{self.episode}")
            self.trajectories.append(self.current_trajectory)
            self.all_rewards.append(sum(self.current_rewards))
            self.all_intervention_rates.append(self.frames_adjusted / self.episode_steps)
            dist = self.get_distance_traveled(self.current_trajectory)
            dist_from_centerline = []
            for i in self.current_trajectory:
                distance_from_centerline = self.dist_from_line(self.centerline_interpolated, i)
                dist_from_centerline.append(min(distance_from_centerline))

            summary = {
                "episode" : self.episode,
                "rewards": self.current_rewards,
                "trajectory": self.current_trajectory,
                "total_steps": self.episode_steps,
                "frames_adjusted_count": self.frames_adjusted,
                "dist_from_centerline": dist_from_centerline,
                "image_shape":self.image_shape,
                "obs_shape": self.obs_shape,
                "action_space": self.action_space,
                "wall_clock_time": time.time() - self.start_time,
                "sim_time": self.runtime
            }
            picklefile = open(f"{self.filepathroot}/summary-epi{self.episode:03d}.pickle", 'wb')
            pickle.dump(summary, picklefile)
            picklefile.close()
            try:
                print(f"\ttotal distance travelled: {dist:.1f}"
                      f"\n\ttotal episode reward: {sum(self.current_rewards):.1f}"
                      f"\n\tavg dist from ctrline: {sum(dist_from_centerline) / len(dist_from_centerline):.3f}"
                      f"\n\tpercent frames adjusted: {self.frames_adjusted / self.episode_steps:.3f}"
                      f"\n\ttotal steps: {self.episode_steps}"
                      f"\n\trew max/min/avg/stdev: {max(self.current_rewards):.3f} / {min(self.current_rewards):.3f} / {sum(self.current_rewards)/len(self.current_rewards):.3f} / {np.std(self.current_rewards):.3f}")
                self.plot_deviation(f"{self.model} {dist=:.1f} ep={self.episode}", self.filepathroot+str(f" avg dist ctr:{sum(dist_from_centerline) / len(dist_from_centerline):.3f} frames adj:{self.frames_adjusted / self.episode_steps:.3f}  rew={sum(self.current_rewards):.1f}"),
                                    save_path=f"{self.filepathroot}/trajs-ep{self.episode}.jpg", start_viz=False)
                self.plot_durations(self.all_rewards, save=True, title=self.filepathroot, savetitle="rewards_training_performance")
                self.plot_durations(self.all_intervention_rates, save=True, title=self.filepathroot, savetitle="interventions_training_performance")
            except Exception as e:
                print(e)

            self.f.close()
        self.episode += 1
        self.ep_dir = f"{self.filepathroot}/ep{self.episode:03d}"
        if not os.path.exists(self.ep_dir):
            os.makedirs(self.ep_dir)
        self.f = open(f"{self.ep_dir}/data.csv", "w")
        self.f.write(f"filename,steering_input,pos,dir,kph,steering,throttle_input\n")
        self.episode_steps, self.frames_adjusted = 0, 0
        self.current_trajectory = []
        self.current_rewards = []
        self.integral, self.prev_error = 0.0, 0.0
        self.steer_integral, self.steer_prev_error = 0.0, 0.0
        self.steer_integral, self.steer_prev_error = 0., 0.
        self.bng.restart_scenario()
        kph = 0
        while kph < 35:
            self.vehicle.update_vehicle()
            sensors = self.bng.poll_sensors(self.vehicle)
            kph = self.ms_to_kph(sensors['electrics']['wheelspeed'])
            self.vehicle.control(throttle=1., steering=0., brake=0.0)
            self.bng.step(1, wait=True)
        sensors = self.bng.poll_sensors(self.vehicle)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.prev_error = self.setpoint
        self.overall_damage, self.runtime, self.wheelspeed, self.distance_from_center = 0.0, 0.0, 0.0, 0.0
        self.total_loops, self.total_imgs, self.total_predictions = 0, 0, 0
        self.start_time = sensors['timer']['time']
        obs = self._get_obs()
        self.image = np.array(self.car_state['front_cam']['colour'].convert('RGB'))
        self.image = cv2.resize(self.image, (self.image_shape[2], self.image_shape[1]))
        if self.image_shape[0] == 1:
            self.image = self.rgb2gray(self.image)
        self.image = self.image.reshape(self.image_shape)
        self.outside_track = False
        self.done = False
        self.states, self.actions, self.probs, self.rewards, self.critic_values = [], [], [], [], []
        self.traj = []

        return obs

    # bad = 0, good = 1
    def evaluator(self, outside_track, distance_from_center, leftrightcenter, segment_shape, steer):
        if leftrightcenter == 0 and abs(steer) <= 0.15 and segment_shape == 0:
            # centered, driving straight, straight road
            return 1
        elif leftrightcenter == 0 and steer < -0.15 and segment_shape == 1:
            # centered, driving left, left curve road
            return 1
        elif leftrightcenter == 0 and steer > 0.15 and segment_shape == 2:
            # centered, driving right, right curve road
            return 1
        elif leftrightcenter == 1 and steer > 0.15:
            # left of center, turning right
            return 1
        elif leftrightcenter == 2 and steer < -0.15:
            # right of center, turning left
            return 1
        else:
            return 0

    # def get_expert_action(self, outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg):
    #     distance_from_centerline = self.dist_from_line(self.centerline_interpolated, self.vehicle.state['pos'])
    #     dist = min(distance_from_centerline)
    #     i = np.where(distance_from_centerline == dist)[0][0]
    #     next_point = self.centerline_interpolated[(i + 3) % len(self.centerline_interpolated)]
    #     theta_deg = self.get_angle_between_3_points_atan2(self.vehicle.state['pos'][0:2], next_point[0:2], self.vehicle.state['front'][0:2])
    #     action = theta_deg / 180
    #     return action, theta_deg

    def get_expert_action(self):
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, self.vehicle.state['front'])
        dist = min(distance_from_centerline)
        coming_index = 3
        i = np.where(distance_from_centerline == dist)[0][0]
        next_point = self.centerline_interpolated[(i + coming_index) % len(self.centerline_interpolated)]
        # next_point2 = centerline_interpolated[(i + coming_index*2) % len(centerline_interpolated)]
        theta = self.angle_between(self.vehicle.state, next_point)
        action = theta / (2 * math.pi)
        # fig, ax = plt.subplots()
        # plt.plot([self.vehicle.state["front"][0], self.vehicle.state["pos"][0]],
        #          [self.vehicle.state["front"][1], self.vehicle.state["pos"][1]], label="car")
        # plt.plot(self.vehicle.state["front"][0], self.vehicle.state["front"][1], "ko", label="front")
        # plt.plot([j[0] for j in self.centerline_interpolated[i + coming_index:i + 20]],
        #          [j[1] for j in self.centerline_interpolated[i + coming_index:i + 20]], label="centerline")
        # plt.plot(next_point[0], next_point[1], 'ro', label="next waypoint")
        # plt.legend()
        # plt.title(f"{action=:.3f}  deg={math.degrees(theta):.1f}")
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')
        # fig.canvas.draw()
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("get_expert_action", img)
        # cv2.waitKey(1)
        # plt.close('all')
        return action, theta

    def angle_between2(self, vehicle_state, next_waypoint, next_waypoint2=None):
        vehicle_angle = math.atan2(vehicle_state['pos'][1] - vehicle_state['front'][1],
                                   vehicle_state['pos'][0] - vehicle_state['front'][0])
        if next_waypoint2 is not None:
            waypoint_angle = math.atan2((next_waypoint2[1] - next_waypoint[1]),
                                        (next_waypoint2[0] - next_waypoint[0]))
        else:
            waypoint_angle = math.atan2((next_waypoint[1] - vehicle_state['front'][1]),
                                        (next_waypoint[0] - vehicle_state['front'][0]))
        inner_angle = vehicle_angle - waypoint_angle
        print(f"wp_angle={math.degrees(waypoint_angle):.1f}  \tvehicle_angle={math.degrees(vehicle_angle):.1f}")
        return math.atan2(math.sin(inner_angle), math.cos(inner_angle))

    def angle_between(self, vehicle_state, next_waypoint, next_waypoint2=None):
        vehicle_angle = math.atan2(vehicle_state['front'][1] - vehicle_state['pos'][1],
                                   vehicle_state['front'][0] - vehicle_state['pos'][0])
        if next_waypoint2 is not None:
            waypoint_angle = math.atan2((next_waypoint2[1] - next_waypoint[1]),
                                        (next_waypoint2[0] - next_waypoint[0]))
        else:
            waypoint_angle = math.atan2((next_waypoint[1] - vehicle_state['front'][1]),
                                        (next_waypoint[0] - vehicle_state['front'][0]))
        inner_angle = vehicle_angle - waypoint_angle
        # print(f"wp_angle={math.degrees(waypoint_angle):.1f}  \tvehicle_angle={math.degrees(vehicle_angle):.1f}")
        return math.atan2(math.sin(inner_angle), math.cos(inner_angle))

    ''' track ~12.50m wide; car ~1.85m wide '''
    def has_car_left_track(self):
        self.vehicle.update_vehicle()
        vehicle_pos = self.vehicle.state['front']
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        leftrightcenter = self.get_position_relative_to_centerline(self.vehicle.state['front'], dist, i, centerdist=0.25)
        # print(f"{leftrightcenter=}  \tdist from ctrline={dist:.3f}")
        segment_shape, theta_deg = self.get_current_segment_shape(vehicle_pos)
        return dist > 4.0, dist, leftrightcenter, segment_shape, theta_deg

    # track ~12.50m wide; car ~1.85m wide
    def has_car_left_track_old(self):
        self.vehicle.update_vehicle()
        vehicle_pos = self.vehicle.state['front'] #self.vehicle.state['pos']
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        # leftrightcenter = self.get_position_relative_to_centerline(dist, i, centerdist=1.5)
        # segment_shape, theta_deg = self.get_current_segment_shape(vehicle_pos)
        return dist > 4.0, dist #, leftrightcenter, segment_shape, theta_deg



    def get_current_segment_shape(self, vehicle_pos):
        distance_from_centerline = self.dist_from_line(self.centerline, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        A = np.array(self.centerline[(i + 2) % len(self.centerline)])
        B = np.array(self.centerline[i])
        C = np.array(self.roadright[i])
        theta = math.acos(np.vdot(B-A, B-C) / (np.linalg.norm(B-A) * np.linalg.norm(B-C)))
        theta_deg = math.degrees(theta)
        if theta_deg > 110:
            return 1, theta_deg
        elif theta_deg < 70:
            return 2, theta_deg
        else:
            return 0, theta_deg

    def get_angle_between_3_points_atan2(self, A, B, C):
        result = math.atan2(C[1] - A[1], C[0] - A[0]) - math.atan2(B[1] - A[1], B[0] - A[0])
        result = math.degrees(result)
        if result > 180:
            return result - 360
        elif result < -180:
            return result + 360
        return result

    '''returns centered=0, left of centerline=-1, right of centerline=1'''

    def get_position_relative_to_centerline(self, front, dist, i, centerdist=1):
        A = self.centerline_interpolated[(i + 1) % len(self.centerline_interpolated)]
        B = self.centerline_interpolated[(i + 4) % len(self.centerline_interpolated)]
        P = front
        d = (P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])
        if abs(dist) < centerdist:
            return 0  # on centerline
        elif d < 0:
            return -1  # left of centerline
        elif d > 0:
            return 1  # right of centerline

    def get_position_relative_to_centerline_old(self, dist, i, centerdist=1):
        A = self.centerline_interpolated[i]
        B = self.centerline_interpolated[(i + 2) % len(self.centerline_interpolated)]
        P = self.vehicle.state['front']
        d = (P[0]-A[0])*(B[1]-A[1])-(P[1]-A[1])*(B[0]-A[0])
        if abs(dist) < centerdist:
            return 0
        elif d < 0:
            return 1
        elif d > 0:
            return 2

    ################################# BEAMNG HELPERS #################################

    def steering_PID(self, curr_steering, steer_setpoint, dt):
        if dt == 0:
            return 0
        if "winding" in self.topo:
            kp = 0.425; ki = 0.00; kd = 0.0  # using LRC and dist to ctrline; Average deviation: 1.023
        elif "straight" in self.topo:
            # kp = 0.8125; ki = 0.00; kd = 0.2
            kp = 0.1; ki = 0.00; kd = 0.01  # decent on straight Average deviation: 1.096
        elif "Rturn" in self.topo:
            # kp = 0.8125; ki = 0.00; kd = 0.0 #0.3
            kp = 0.325; ki = 0.00; kd = 0.0 #0.3
        elif "Lturn" in self.topo:
            kp = 0.5; ki = 0.00; kd = 0.3
        else:
            kp = 0.75; ki = 0.01; kd = 0.2  # decent
        error = steer_setpoint - curr_steering
        deriv = (error - self.steer_prev_error) / dt
        self.steer_integral = self.steer_integral + error * dt
        w = kp * error + ki * self.steer_integral + kd * deriv
        # print(f"steering_PID({curr_steering=:.3f}  \t{steer_setpoint=:.3f}  \t{dt=:.3f})  \t{self.steer_prev_error=:.3f}  "
        #       f"\n{error:.3f} \t{deriv=:.3f} \t{w=:.3f}")
        self.steer_prev_error = error
        return w

    def throttle_PID(self, kph, dt):
        kp, ki, kd = 0.19, 0.0001, 0.008
        error = self.setpoint - kph
        if dt > 0:
            deriv = (error - self.prev_error) / dt
        else:
            deriv = 0
        self.integral = self.integral + error * dt
        w = kp * error + ki * self.integral + kd * deriv
        self.prev_error = error
        return w

    def setup_sensors(self):
        camera_pos = (-0.5, 0.38, 1.3)
        camera_dir = (0, 1.0, 0)
        if self.transf == "fisheye":
            fov = 75
        else:
            fov = 51 # 60 works for full lap #63 breaks on hairpin turn

        width = int(self.obs_shape[2])
        height = int(self.obs_shape[1])
        resolution = (width, height)
        if self.transf == "depth":
            front_camera = Camera(camera_pos, camera_dir, fov, resolution,
                                  colour=True, depth=True, annotation=False, near_far=(1, 50))
            far_camera = Camera(camera_pos, camera_dir, fov, (self.obs_shape[2], self.obs_shape[1]),
                                  colour=True, depth=True, annotation=False, near_far=(1, 100))
        else:
            front_camera = Camera(camera_pos, camera_dir, fov, resolution,
                                  colour=True, depth=True, annotation=False)
        gforces = GForces()
        electrics = Electrics()
        damage = Damage()
        timer = Timer()
        self.vehicle.attach_sensor('front_cam', front_camera)
        if self.transf == "depth":
            self.vehicle.attach_sensor('far_camera', far_camera)
        self.vehicle.attach_sensor('gforces', gforces)
        self.vehicle.attach_sensor('electrics', electrics)
        self.vehicle.attach_sensor('damage', damage)
        self.vehicle.attach_sensor('timer', timer)

    def spawn_point(self):
        if self.default_scenario == 'cliff':
            #return {'pos':(-124.806, 142.554, 465.489), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
        elif self.default_scenario == 'west_coast_usa':
            if self.road_id == 'midhighway': # mid highway scenario (past shadowy parts of road)
                return {'pos': (-145.775, 211.862, 115.55), 'rot': None, 'rot_quat': (0.0032586499582976, -0.0018308814615011, 0.92652350664139, -0.37621837854385)}
            # actually past shadowy parts of road?
            #return {'pos': (95.1332, 409.858, 117.435), 'rot': None, 'rot_quat': (0.0077012465335429, 0.0036200874019414, 0.90092438459396, -0.43389266729355)}
            # surface road (crashes early af)
            elif self.road_id == "13242":
                return {'pos': (-733.7, -923.8, 163.9), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, 0.805, 0.592), -20)}
            elif self.road_id == "8650":
                return {'pos': (-365.24, -854.45, 136.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 90)}
            elif self.road_id == "12667":
                return {'pos': (-892.4, -793.4, 114.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 70)}
            elif self.road_id == "8432":
                if self.reverse:
                    return {'pos': (-871.9, -803.2, 115.3), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 165 - 180)}
                return {'pos': (-390.4,-799.1,139.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 165)}
            elif self.road_id == "8518":
                if self.reverse:
                    return {'pos': (-913.2, -829.6, 118.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 60)}
                return {'pos': (-390.5, -896.6, 138.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 20)}
            elif self.road_id == "8417":
                return {'pos': (-402.7,-780.2,141.3), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == "8703":
                if self.reverse:
                    return {'pos': (-312.4, -856.8, 135.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
                return {'pos': (-307.8,-784.9,137.6), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 80)}
            elif self.road_id == "12641":
                if self.reverse:
                    return {'pos': (-964.2, 882.8, 75.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
                return {'pos': (-366.1753845214844, 632.2236938476562, 75.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 180)}
            elif self.road_id == "13091":
                if self.reverse:
                    return {'pos': (-903.6078491210938, -586.33154296875, 106.6), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 80)}
                return {'pos': (-331.0728759765625,-697.2451782226562,133.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "11602":
            #     return {'pos': (-366.4, -858.8, 136.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == "12146":
                if self.reverse:
                    return {'pos': (995.7, -855.0, 167.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -15)}
                return {'pos': (-391.0,-798.8, 139.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -15)}
            elif self.road_id == "13228":
                return {'pos': (-591.5175170898438,-453.1298828125,114.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == "13155": #middle laneline on highway #12492 11930, 10368 is an edge
                return {'pos': (-390.7796936035156, -36.612098693847656, 109.9), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -90)}
            elif self.road_id == "10784": # 13228  suburb edge, 12939 10371 12098 edge
                return {'pos': (57.04786682128906, -150.53302001953125, 125.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -115)}
            elif self.road_id == "10673":
                return {'pos': (-21.712169647216797, -826.2122802734375, 133.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -90)}
            elif self.road_id == "12930":  # 13492 dirt road
                # return {'pos': (-347.2, -824.7, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                return {'pos': (-353.731, -830.905, 137.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            elif self.road_id == "10988":  # track
                # return {'pos': (622.2, -251.1, 147.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
                # return {'pos': (660.388,-247.67,147.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -160)}
                if self.seg == 0:  # straight portion
                    return {'pos': (687.5048828125, -185.7435302734375, 146.9), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                elif self.seg == 1:  # approaching winding portion
                    #  crashes around [846.0238647460938, 127.84288787841797, 150.64915466308594]
                    # return {'pos': (768.1991577148438, -108.50184631347656, 146.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                    return {'pos': (781.2423095703125, -95.72360229492188, 147.4), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                    return {'pos': (790.599, -86.7973, 147.3), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -100)}  # slightly better?
                elif self.seg == 2:
                    return {'pos': (854.4083862304688, 136.79324340820312, 152.7), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                else:
                    return {'pos': (599.341, -252.333, 147.6), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -60)}
                # other end of the gas station-airfield yellow road
                # return {'pos': (622.163330078125,-251.1154022216797,146.99), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -90)}
            elif self.road_id == "13306":
                return {'pos': (-310,-790.044921875,137.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -30)}
            elif self.road_id == "13341":
                return {'pos': (-393.4385986328125,-34.0107536315918,109.64727020263672), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 90)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == '12669':
                return {'pos': (456.85526276, -183.39646912,  145.54124832), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922), 90)}
            elif self.road_id == 'surfaceroad1':
                return {'pos': (945.285, 886.716, 132.061), 'rot': None, 'rot_quat': (-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922)}
            # surface road 2
            elif self.road_id == 'surfaceroad2':
                return {'pos': (900.016, 959.477, 127.227), 'rot': None, 'rot_quat': (-0.046136282384396, 0.018260028213263, 0.94000166654587, 0.3375423848629)}
            # surface road 3 (start at top of hill)
            elif self.road_id == 'surfaceroad3':
                return {'pos':(873.494, 984.636, 125.398), 'rot': None, 'rot_quat':(-0.043183419853449, 2.3034785044729e-05, 0.86842048168182, 0.4939444065094)}
            # surface road 4 (right turn onto surface road) (HAS ACCOMPANYING AI DIRECTION AS ORACLE)
            elif self.road_id == 'surfaceroad4':
                return {'pos': (956.013, 838.735, 134.014), 'rot': None, 'rot_quat': (0.020984912291169, 0.037122081965208, -0.31912142038345, 0.94675397872925)}
            # surface road 5 (ramp past shady el)
            elif self.road_id == 'surfaceroad5':
                return {'pos':(166.287, 812.774, 102.328), 'rot': None, 'rot_quat':(0.0038638345431536, -0.00049926445353776, 0.60924011468887, 0.79297626018524)}
            # entry ramp going opposite way
            elif self.road_id == 'entryrampopp':
                return {'pos': (850.136, 946.166, 123.827), 'rot': None, 'rot_quat': (-0.030755277723074, 0.016458060592413, 0.37487033009529, 0.92642092704773)}
            # racetrack
            elif self.road_id == 'racetrack':
                return {'pos': (395.125, -247.713, 145.67), 'rot': None, 'rot_quat': (0, 0, 0.700608, 0.713546)}
        elif self.default_scenario == 'smallgrid':
            return {'pos':(0.0, 0.0, 0.0), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            # right after toll
            return {'pos': (-852.024, -517.391 + lanewidth, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
            # return {'pos':(-717.121, 101, 118.675), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            return {'pos': (-717.121, 101, 118.675), 'rot': None, 'rot_quat': (0, 0, 0.918812, -0.394696)}
        elif self.default_scenario == 'automation_test_track':
            if self.road_id == 'startingline':
                # starting line
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
            elif self.road_id == "7991":
                return {'pos': (57.229, 360.560, 128.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            elif self.road_id == "8293":
                return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
            elif self.road_id == '8185': # highway (open, farm-like)
                return {'pos': (174.92, -289.7, 120.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
                # return {'pos': (174.9,-289.7,120.8), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.704635, 0.70957), 180)}
                # return {'pos': (36.742,-269.105,120.461), 'rot': None, 'rot_quat': (-0.0070,0.0082,0.7754,0.6314)}
                # return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.704635, 0.70957), 180)}
            elif self.road_id == 'starting line 30m down':
                # 30m down track from starting line
                return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
            elif self.road_id == 'handlingcircuit':
                # handling circuit
                return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
            elif self.road_id == 'handlingcircuit2':
                return {'pos': (-280.704, -25.4946, 118.794), 'rot': None, 'rot_quat': (-0.00862686, 0.0063203, 0.98271, 0.184842)}
            elif self.road_id == 'handlingcircuit3':
                return {'pos': (-214.929, 61.2237, 118.593), 'rot': None, 'rot_quat': (-0.00947676, -0.00484788, -0.486675, 0.873518)}
            elif self.road_id == 'handlingcircuit4':
                # return {'pos': (-180.663, 117.091, 117.654), 'rot': None, 'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
                # return {'pos': (-171.183,147.699,117.438), 'rot': None, 'rot_quat': (0.001710215350613,-0.039731655269861,0.99312973022461,-0.11005393415689)}
                return {'pos': (-173.009,137.433,116.701), 'rot': None,'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
                return {'pos': (-166.679, 146.758, 117.68), 'rot': None,'rot_quat': (0.075107827782631, -0.050610285252333, 0.99587279558182, 0.0058960365131497)}
            elif self.road_id == 'rally track':
                # rally track
                return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
            elif self.road_id == 'highway':
                # highway (open, farm-like)
                return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
            elif self.road_id == 'highwayopp':
                # highway (open, farm-like)
                return {'pos': (-542.719,-251.721,117.083), 'rot': None, 'rot_quat': (0.0098941307514906,0.0096141006797552,0.72146373987198,0.69231480360031)}
            elif self.road_id == 'default':
                # default
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif self.default_scenario == 'industrial':
            if self.road_id == 'west':
                # western industrial area -- didnt work with AI Driver
                return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
            # open industrial area -- didnt work with AI Driver
            # drift course (dirt and paved)
            elif self.road_id == 'driftcourse':
                return {'pos': (20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
            # rallycross course/default
            elif self.road_id == 'rallycross':
                return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
            # racetrack
            elif self.road_id == 'racetrackright':
                return {'pos': (184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat': (-0.005, 0.001, 0.299, 0.954)}
            elif self.road_id == 'racetrackleft':
                return {'pos': (216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat': (-0.0051, -0.003147, -0.67135, 0.74112)}
            elif self.road_id == 'racetrackstartinggate':
                return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
            elif self.road_id == "racetrackstraightaway":
                return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.010505940765142, 0.029969356954098, -0.44812294840813, 0.89340770244598)}
            elif self.road_id == "racetrackcurves":
                return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.029027424752712,0.022241719067097,0.98601061105728,0.16262225806713)}
        elif self.default_scenario == "hirochi_raceway":
            # if self.road_id == "9039" or self.road_id == "9040": # good candidate for input rect.
            #     if self.seg == 0:
            #         return {'pos': (289.327, -281.458, 46.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            #     else:
            #         # return {'pos': (292.405,-271.64,46.75), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}
            #         return {'pos': (290.558, -277.280, 46.0), 'rot': None,
            #                 'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}

            if self.road_id == "9039":  # good candidate for input rect.
                if self.seg == 0:  # start of track, right turn; 183m; cutoff at (412.079,-191.549,38.2418)
                    return {'pos': (289.327, -281.458, 46.0), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
                elif self.seg == 1:  # straight road
                    return {'pos': (330.3320007324219, -217.5743408203125, 45.7054443359375), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
                elif self.seg == 2:  # left turn
                    # return {'pos': (439.0, -178.4, 35.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                    # return {'pos': (448.1, -174.6, 34.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                    return {'pos': (496.2, -150.6, 35.6), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                elif self.seg == 3:
                    return {'pos': (538.2, -124.3, 40.5), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -110)}
                elif self.seg == 4:  # straight; cutoff at vec3(596.333,18.7362,45.6584)
                    return {'pos': (561.7396240234375, -76.91995239257812, 44.7), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
                elif self.seg == 5:  # left turn; cutoff at (547.15234375, 115.24089050292969, 35.97171401977539)
                    return {'pos': (598.3154907226562, 40.60638427734375, 43.9), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -147)}
                elif self.seg == 6:
                    return {'pos': (547.15234375, 115.24089050292969, 36.3), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
                elif self.seg == 7:
                    return {'pos': (449.7561340332031, 114.96491241455078, 25.801856994628906), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
                elif self.seg == 8:  # mostly straight, good behavior; cutoff at  vec3(305.115,304.196,38.4392)
                    return {'pos': (405.81732177734375, 121.84907531738281, 25.04170036315918), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
                elif self.seg == 9:
                    return {'pos': (291.171875, 321.78662109375, 38.6), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
                elif self.seg == 10:
                    return {'pos': (216.40045166015625, 367.1772155761719, 35.99), 'rot': None,
                            'rot_quat': (-0.037829957902431, 0.0035844487138093, 0.87171512842178, 0.48853760957718)}
                else:
                    return {'pos': (290.558, -277.280, 46.0), 'rot': None,
                            'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif self.road_id == "9205":
                return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            elif self.road_id == "9156":
                return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            else:
                return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.2777, 0.9607)}
        elif self.default_scenario == "small_island":
            if self.road_id == "int_a_small_island":
                return {"pos": (280.397, 210.259, 35.023), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 110)}
            elif self.road_id == "ai_1":
                return {"pos": (314.573, 105.519, 37.5), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 155)}
            else:
                return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542)}
        elif self.default_scenario == "jungle_rock_island":
            return {'pos': (-9.99082, 580.726, 156.72), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}

    def create_ai_line_from_road_with_interpolation(self):
        points, point_colors, spheres, sphere_colors = [], [], [], []
        self.centerline_interpolated = []
        self.road_analysis()
        # interpolate centerline at 1m distance
        for i, p in enumerate(self.centerline[:-1]):
            if self.distance(p, self.centerline[i+1]) > 1:
                y_interp = interpolate.interp1d([p[0], self.centerline[i+1][0]], [p[1], self.centerline[i+1][1]])
                num = int(self.distance(p, self.centerline[i+1]))
                xs = np.linspace(p[0], self.centerline[i+1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                for x,y in zip(xs, ys):
                    self.centerline_interpolated.append([x, y])
            else:
                self.centerline_interpolated.append([p[0],p[1]])
        # set up debug line
        for p in self.centerline[:-1]:
            points.append([p[0], p[1], p[2]])
            point_colors.append([0, 1, 0, 0.1])
            spheres.append([p[0], p[1], p[2], 0.25])
            sphere_colors.append([1, 0, 0, 0.8])
        self.bng.add_debug_line(points, point_colors, spheres=spheres, sphere_colors=sphere_colors, cling=True, offset=0.1)

    def ms_to_kph(self, wheelspeed):
        return wheelspeed * 3.6

    '''return distance between two N-dimensional points'''
    def distance(self, a, b):
        sqr = sum([math.pow(ai-bi, 2) for ai, bi in zip(a,b)])
        return math.sqrt(sqr)

    def road_analysis(self):
        print("Performing road analysis...")
        # self.get_nearby_racetrack_roads(point_of_in=(-391.0,-798.8, 139.7))
        # self.plot_racetrack_roads()
        print(f"Getting road {self.road_id}...")
        edges = self.bng.get_road_edges(self.road_id)
        if self.reverse:
            edges.reverse()
            print(f"new spawn={edges[0]['middle']}")
        else:
            print(f"reversed spawn={edges[-1]['middle']}")
        self.centerline = [edge['middle'] for edge in edges]
        # self.roadleft = [edge['left'] for edge in edges]
        # self.roadright = [edge['right'] for edge in edges]
        if self.road_id == "8185":
            edges = self.bng.get_road_edges("8096")
            self.roadleft = [edge['middle'] for edge in edges]
            edges = self.bng.get_road_edges("7878")  # 7820, 7878, 7805
            self.roadright = [edge['middle'] for edge in edges]
        else:
            self.roadleft = [edge['left'] for edge in edges]
            self.roadright = [edge['right'] for edge in edges]

    def plot_racetrack_roads(self):
        roads = self.bng.get_roads()
        sp = self.spawn_point()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
        for road in roads:
            road_edges = self.bng.get_road_edges(road)
            if len(road_edges) < 100:
                continue
            if self.distance(road_edges['left'][0], road_edges['left'][0]) < 5:
                continue
            x_temp, y_temp = [], []
            xy_def = [edge['middle'][:2] for edge in road_edges]
            dists = [self.distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1])]
            s = sum(dists)
            if (s < 500) or s > 800:
                continue
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
            print(f"{road}\tdist={s:.1f}\t{len(road_edges)=}\tstart={road_edges[0]['middle']}")
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
        plt.plot([sp['pos'][0]], [sp['pos'][1]], "bo")
        plt.title(self.default_scenario)
        plt.legend(ncol=10)
        plt.show()
        plt.pause(0.001)

    def get_nearby_racetrack_roads(self, point_of_in):
        print(f"Plotting nearby roads to point={point_of_in}")
        roads = self.bng.get_roads()
        print("retrieved roads")
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
        for road in roads:
            road_edges = self.bng.get_road_edges(road)
            x_temp, y_temp = [], []
            if len(road_edges) < 100:
                continue
            xy_def = [edge['middle'][:2] for edge in road_edges]
            # dists = [self.distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1:5])]
            # road_len = sum(dists)
            dists = [self.distance(i, point_of_in) for i in xy_def]
            s = min(dists)
            if (s > 100): # or road_len < 200:
                continue
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
            print(f"{road=}\tstart=({x_temp[0]},{y_temp[0]},{road_edges[0]['middle'][2]})\t{road_edges[0]['middle']}")
        plt.plot([point_of_in[0]], [point_of_in[1]], "bo")
        plt.title(f"{self.default_scenario} poi={point_of_in}")
        plt.legend(ncol=10)
        plt.show()
        plt.pause(0.001)

    def turn_X_degrees(self, rot_quat, degrees=90):
        r = R.from_quat(list(rot_quat))
        r = r.as_euler('xyz', degrees=True)
        r[2] = r[2] + degrees
        r = R.from_euler('xyz', r, degrees=True)
        return tuple(r.as_quat())

    def dist_from_line(self, centerline, point):
        a = [x[0:2] for x in centerline[:-1]]
        b = [x[0:2] for x in centerline[1:]]
        return self.lineseg_dists(point[0:2], np.array(a), np.array(b))

    def lineseg_dists(self, p, a, b):
        """Cartesian distance from point to line segment
        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892
        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

        return np.hypot(h, c)

    def plot_deviation(self, model, deflation_pattern, save_path=".", start_viz=False):
        plt.figure(20, dpi=180)
        plt.clf()
        x, y = [], []
        for point in self.centerline:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "b-")
        x, y = [], []
        for point in self.roadleft:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "b-")
        x, y = [], []
        for point in self.roadright:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "b-", label="Road")
        for i,t in enumerate(self.trajectories):
            x, y = [], []
            for point in t:
                x.append(point[0])
                y.append(point[1])
            plt.plot(x, y) #, label="Run {}".format(i))
        if self.topo == "winding":
            plt.xlim([700, 900])
            plt.ylim([-200, 125])
        elif self.topo == "Rturn":
            plt.xlim([200, 400])
            plt.ylim([-300, -100])
        elif self.topo == "straight":
            plt.xlim([25, 200])
            plt.ylim([-300, -265])
        details = deflation_pattern.split("/")[-1]
        if "avg" in details:
            details = details.replace("avg", "\navg")
        plt.title(f'Trajectories with {model}\n{details}', fontdict={'fontsize': 8})
        plt.legend(fontsize=8)
        plt.draw()
        plt.savefig(save_path)
        plt.clf()

    def get_distance_traveled(self, traj):
        dist = 0.0
        for i in range(len(traj[:-1])):
            dist += math.sqrt(
                math.pow(traj[i][0] - traj[i + 1][0], 2) + math.pow(traj[i][1] - traj[i + 1][1], 2) + math.pow(
                    traj[i][2] - traj[i + 1][2], 2))
        return dist

    def plot_durations(self, rewards, save=False, title="temp", savetitle="rewards_training_performance"):
        # plt.figure(2)
        plt.figure(4, figsize=(10,10), dpi=100)
        plt.clf()
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        # durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        # plt.ylabel('Duration')
        plt.plot(rewards_t.numpy(), label=savetitle)
        # plt.plot(durations_t.numpy(), "--")
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 20:
            means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
            means = torch.cat((torch.ones(19) * means[0], means))
            plt.plot(means.numpy(), '--')
        plt.legend()
        if save:
            plt.savefig(f"{title}/{savetitle}.jpg")