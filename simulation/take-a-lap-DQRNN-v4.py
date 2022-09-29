import os.path
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
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
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


'''
actor-critic model based on Microsoft example:
https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/22-DeepRL/CartPole-RL-TF.ipynb
'''

# globals
default_color = 'green'
default_scenario = 'hirochi_raceway'
road_id = "9039"
integral = 0.0
prev_error = 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []
steps_per_sec = 30

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super().__init__()
        self.input_shape = (h,w)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 5, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 5, stride=1)
        self.dropout = nn.Dropout()

        s1 = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        size = np.product(s1(torch.zeros(1, 3, *self.input_shape)).shape)

        self.lin1 = nn.Linear(in_features=size, out_features=100, bias=True)
        self.lin2 = nn.Linear(in_features=100, out_features=50, bias=True)
        self.lin3 = nn.Linear(in_features=50, out_features=10, bias=True)
        self.lin4 = nn.Linear(in_features=10, out_features=outputs, bias=True)

        self.lin5 = nn.Linear(in_features=size, out_features=100, bias=True)
        self.lin6 = nn.Linear(in_features=100, out_features=outputs, bias=True)


    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = self.conv5(x)
        x = F.elu(x)
        x = x.flatten(1)
        a = self.lin1(x)
        a = self.dropout(a)
        a = F.elu(a)
        a = self.lin2(a)
        a = self.dropout(a)
        a = F.elu(a)
        a = self.lin3(a)
        a = self.dropout(a)
        a = F.elu(a)
        a = self.lin4(a)
        # x = torch.tanh(x)
        # x = torch.softmax(x, dim=0)
        # x = 2 * torch.atan(x)
        a = torch.abs(a)

        b = self.lin5(x)
        b = self.dropout(b)
        b = F.relu(b)
        b = self.lin6(b)
        b = self.dropout(b)
        b = F.relu(b)
        return a, b


def spawn_point():
    global lanewidth, road_id, default_scenario
    if default_scenario == 'cliff':
        #return {'pos':(-124.806, 142.554, 465.489), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
    elif default_scenario == 'west_coast_usa':
        if road_id == 'midhighway':
            # mid highway scenario (past shadowy parts of road)
            return {'pos': (-145.775, 211.862, 115.55), 'rot': None, 'rot_quat': (0.0032586499582976, -0.0018308814615011, 0.92652350664139, -0.37621837854385)}
        # actually past shadowy parts of road?
        #return {'pos': (95.1332, 409.858, 117.435), 'rot': None, 'rot_quat': (0.0077012465335429, 0.0036200874019414, 0.90092438459396, -0.43389266729355)}
        # surface road (crashes early af)
        elif road_id == '12669':
            return {'pos': (456.85526276, -183.39646912,  145.54124832), 'rot': None, 'rot_quat': turn_X_degrees((-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922), 90)}
        elif road_id == 'surfaceroad1':
            return {'pos': (945.285, 886.716, 132.061), 'rot': None, 'rot_quat': (-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922)}
        # surface road 2
        elif road_id == 'surfaceroad2':
            return {'pos': (900.016, 959.477, 127.227), 'rot': None, 'rot_quat': (-0.046136282384396, 0.018260028213263, 0.94000166654587, 0.3375423848629)}
        # surface road 3 (start at top of hill)
        elif road_id == 'surfaceroad3':
            return {'pos':(873.494, 984.636, 125.398), 'rot': None, 'rot_quat':(-0.043183419853449, 2.3034785044729e-05, 0.86842048168182, 0.4939444065094)}
        # surface road 4 (right turn onto surface road) (HAS ACCOMPANYING AI DIRECTION AS ORACLE)
        elif road_id == 'surfaceroad4':
            return {'pos': (956.013, 838.735, 134.014), 'rot': None, 'rot_quat': (0.020984912291169, 0.037122081965208, -0.31912142038345, 0.94675397872925)}
        # surface road 5 (ramp past shady el)
        elif road_id == 'surfaceroad5':
            return {'pos':(166.287, 812.774, 102.328), 'rot': None, 'rot_quat':(0.0038638345431536, -0.00049926445353776, 0.60924011468887, 0.79297626018524)}
        # entry ramp going opposite way
        elif road_id == 'entryrampopp':
            return {'pos': (850.136, 946.166, 123.827), 'rot': None, 'rot_quat': (-0.030755277723074, 0.016458060592413, 0.37487033009529, 0.92642092704773)}
        # racetrack
        elif road_id == 'racetrack':
            return {'pos': (395.125, -247.713, 145.67), 'rot': None, 'rot_quat': (0, 0, 0.700608, 0.713546)}
    elif default_scenario == 'smallgrid':
        return {'pos':(0.0, 0.0, 0.0), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        # right after toll
        return {'pos': (-852.024, -517.391 + lanewidth, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
        # return {'pos':(-717.121, 101, 118.675), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        return {'pos': (-717.121, 101, 118.675), 'rot': None, 'rot_quat': (0, 0, 0.918812, -0.394696)}
    elif default_scenario == 'automation_test_track':
        if road_id == 'startingline':
            # starting line
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif road_id == "7991":
            return {'pos': (57.229, 360.560, 128.203), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif road_id == "8293":
            return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
        elif road_id == 'starting line 30m down':
            # 30m down track from starting line
            return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif road_id == 'handlingcircuit':
            # handling circuit
            return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
        elif road_id == 'handlingcircuit2':
            return {'pos': (-280.704, -25.4946, 118.794), 'rot': None, 'rot_quat': (-0.00862686, 0.0063203, 0.98271, 0.184842)}
        elif road_id == 'handlingcircuit3':
            return {'pos': (-214.929, 61.2237, 118.593), 'rot': None, 'rot_quat': (-0.00947676, -0.00484788, -0.486675, 0.873518)}
        elif road_id == 'handlingcircuit4':
            # return {'pos': (-180.663, 117.091, 117.654), 'rot': None, 'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
            # return {'pos': (-171.183,147.699,117.438), 'rot': None, 'rot_quat': (0.001710215350613,-0.039731655269861,0.99312973022461,-0.11005393415689)}
            return {'pos': (-173.009,137.433,116.701), 'rot': None,'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
            return {'pos': (-166.679, 146.758, 117.68), 'rot': None,'rot_quat': (0.075107827782631, -0.050610285252333, 0.99587279558182, 0.0058960365131497)}
        elif road_id == 'rally track':
            # rally track
            return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
        elif road_id == 'highway':
            # highway (open, farm-like)
            return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
        elif road_id == 'highwayopp':
            # highway (open, farm-like)
            return {'pos': (-542.719,-251.721,117.083), 'rot': None, 'rot_quat': (0.0098941307514906,0.0096141006797552,0.72146373987198,0.69231480360031)}
        elif road_id == 'default':
            # default
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
    elif default_scenario == 'industrial':
        if road_id == 'west':
            # western industrial area -- didnt work with AI Driver
            return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
        # open industrial area -- didnt work with AI Driver
        # drift course (dirt and paved)
        elif road_id == 'driftcourse':
            return {'pos': (20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
        # rallycross course/default
        elif road_id == 'rallycross':
            return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
        # racetrack
        elif road_id == 'racetrackright':
            return {'pos': (184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat': (-0.005, 0.001, 0.299, 0.954)}
        elif road_id == 'racetrackleft':
            return {'pos': (216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat': (-0.0051, -0.003147, -0.67135, 0.74112)}
        elif road_id == 'racetrackstartinggate':
            return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
        elif road_id == "racetrackstraightaway":
            return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.010505940765142, 0.029969356954098, -0.44812294840813, 0.89340770244598)}
        elif road_id == "racetrackcurves":
            return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.029027424752712,0.022241719067097,0.98601061105728,0.16262225806713)}
    elif default_scenario == "hirochi_raceway":
        if road_id == "9039": # good candidate for input rect.
            return {'pos': (290.558, -277.280, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}
        elif road_id == "9205":
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif road_id == "9156":
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        else:
            return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.2777, 0.9607)}
    elif default_scenario == "small_island":
        if road_id == "int_a_small_island":
            return {"pos": (280.397, 210.259, 35.023), 'rot': None, 'rot_quat': turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 110)}
        elif road_id == "ai_1":
            return {"pos": (314.573, 105.519, 37.5), 'rot': None, 'rot_quat': turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 155)}
        else:
            return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542)}
    elif default_scenario == "jungle_rock_island":
        return {'pos': (-9.99082, 580.726, 156.72), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}


def setup_sensors(vehicle):
    camera_pos = (-0.5, 0.38, 1.3)
    camera_dir = (0, 1.0, 0)
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    front_camera = Camera(camera_pos, camera_dir, fov, resolution,
                          colour=True, depth=True, annotation=True)
    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()

    # Attach them
    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle


def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6


def throttle_PID(kph, dt):
    global integral, prev_error, setpoint
    kp = 0.19; ki = 0.0001; kd = 0.008
    error = setpoint - kph
    if dt > 0:
        deriv = (error - prev_error) / dt
    else:
        deriv = 0
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w


def plot_deviation(trajectories, model, deflation_pattern):
    global centerline, roadleft, roadright
    plt.figure(20, dpi=180)
    plt.clf()
    i = 0; x = []; y = []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "b-")
    x, y = [], []
    for point in roadleft:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "b-")
    x, y = [], []
    for point in roadright:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "b-", label="Road")
    for t in trajectories:
        x = []; y = []
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, label="Run {}".format(i))
        i += 1
    plt.title(f'Trajectories with {model} {deflation_pattern}')
    plt.legend(fontsize=8)
    plt.xlim([200,400])
    plt.ylim([-300,-100])
    plt.draw()
    plt.savefig(f"{model}-{deflation_pattern}-trajs-so-far.jpg")
    plt.clf()

def lineseg_dists(p, a, b):
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


#return distance between two any-dimenisonal points
def distance(a, b):
    sqr = sum([math.pow(ai-bi, 2) for ai, bi in zip(a,b)])
    # return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    return math.sqrt(sqr)


def dist_from_line(centerline, point):
    a = [[x[0],x[1]] for x in centerline[:-1]]
    b = [[x[0],x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist


def calc_deviation_from_center(centerline, traj):
    dists = []
    for point in traj:
        dist = dist_from_line(centerline, point)
        dists.append(min(dist))
    stddev = statistics.stdev(dists)
    return stddev


def road_analysis(bng):
    global centerline, roadleft, roadright
    global road_id, default_scenario, road_id
    # plot_racetrack_roads(bng.get_roads(), bng)
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    actual_middle = [edge['middle'] for edge in edges]
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [edge['middle'] for edge in edges] #[np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/4.0 for edge in edges]
    centerline = actual_middle
    return actual_middle, adjusted_middle


def plot_trajectory(traj, title="Trajectory", label1="car traj."):
    global centerline, roadleft, roadright
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r-')
    plt.plot([t[0] for t in roadleft], [t[1] for t in roadleft], 'r-')
    plt.plot([t[0] for t in roadright], [t[1] for t in roadright], 'r-')
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'b--', label=label1)
    # plt.gca().set_aspect('equal')
    # plt.axis('square')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.savefig(f"{title}.jpg")
    plt.show()
    plt.pause(0.1)


def plot_input(timestamps, input, input_type, run_number=0):
    plt.plot(timestamps, input)
    plt.xlabel('Timestamps')
    plt.ylabel('{} input'.format(input_type))
    plt.title("{} over time".format(input_type))
    plt.savefig("Run-{}-{}.png".format(run_number, input_type))
    plt.show()
    plt.pause(0.1)


def create_ai_line_from_road_with_interpolation(spawn, bng):
    global centerline, remaining_centerline, centerline_interpolated, road_id
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    print("Performing road analysis...")
    actual_middle, adjusted_middle = road_analysis(bng)
    print(f"{actual_middle[0]=}")
    print(f"{actual_middle[-1]=}")
    middle_end = adjusted_middle[:3]
    middle = adjusted_middle[3:]
    temp = [list(spawn['pos'])]; temp.extend(middle); middle = temp
    middle.extend(middle_end)
    remaining_centerline = copy.deepcopy(middle)
    timestep = 0.1; elapsed_time = 0; count = 0
    # set up adjusted centerline
    for i,p in enumerate(middle[:-1]):
        # interpolate at 1m distance
        if distance(p, middle[i+1]) > 1:
            y_interp = interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
            num = int(distance(p, middle[i+1]))
            xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x,y in zip(xs,ys):
                traj.append([x,y])
        else:
            elapsed_time += distance(p, middle[i+1]) / 12
            traj.append([p[0],p[1]])
            linedict = {"x": p[0], "y": p[1], "z": p[2], "t": elapsed_time}
            line.append(linedict)
            count += 1
    # set up debug line
    for i,p in enumerate(actual_middle[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
        count += 1
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    # plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    # centerline = copy.deepcopy(traj)
    remaining_centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    # for i in range(4):
    #     centerline.extend(copy.deepcopy(centerline))
    #     remaining_centerline.extend(copy.deepcopy(remaining_centerline))
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng


# track ~12.50m wide; car ~1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    dist = min(distance_from_centerline)
    return dist > 4.0, dist


def setup_beamng(vehicle_model='etk800', camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0):
    global base_filename, default_color, default_scenario, road_id, steps_per_sec, road_id

    random.seed(1703)
    setup_logging()

    beamng = BeamNGpy('localhost', 64356, home='C:/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE2')

    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model,
                      licence='EGO', color=default_color)

    vehicle = setup_sensors(vehicle)
    spawn = spawn_point()

    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat'])
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    bng.set_deterministic()
    bng.set_steps_per_second(steps_per_sec)
    bng.load_scenario(scenario)
    print("Starting scenario....")
    bng.start_scenario()
    print("Interpolating centerline...")
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn, bng)
    print("Pausing BeamNG...")
    bng.pause()
    assert vehicle.skt
    return vehicle, bng, scenario


def run_scenario(vehicle, bng, scenario, model=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    global base_filename, default_color, default_scenario, road_id, steps_per_sec
    global integral, prev_error, setpoint
    integral, prev_error = 0.0, 0.0
    bng.restart_scenario()
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    endpoint = [-346.5, 431.0, 30.8] #centerline[-1]
    transform = transforms.Compose([transforms.ToTensor()])
    prev_error = setpoint
    overall_damage, runtime, wheelspeed, distance_from_center = 0.0, 0.0, 0.0, 0.0
    total_loops, total_imgs, total_predictions = 0, 0, 0
    start_time = sensors['timer']['time']
    state = np.array(sensors['front_cam']['colour'].convert('RGB'))
    outside_track = False
    done = False
    action_inputs = [-1, 0, 1]
    action_indices = [0,1,2]
    states, actions, probs, rewards, critic_values = [], [], [], [], []
    traj = []
    while overall_damage <= 0 or not outside_track or not done:
        cv2.imshow('car view', state[:, :, ::-1])
        cv2.waitKey(1)
        total_imgs += 1
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - start_time) - runtime
        runtime = sensors['timer']['time'] - start_time
        traj.append(vehicle.state['pos'])

        if kph < 30:
            vehicle.control(throttle=1.0, steering=0.0, brake=0.0)
        else:
            # Select and perform an action
            action_probs, est_rew = model(torch.tensor(state.reshape((1, 3, 135, 240)), dtype=torch.float32, device=device))
            action_probs = action_probs.cpu().detach().numpy()
            est_rew = est_rew.cpu().detach().numpy()
            action_index = np.random.choice(action_indices, p=np.squeeze(action_probs) / np.sum(action_probs))
            # action_index = np.where(np.max(action_probs[0]))[0][0]
            action = action_inputs[action_index]
            print(f"cycle={total_loops:04}\t{distance_from_center=:.1f}\t{action_probs=}\t{action_index=}\t{action=}")

            if abs(action) > 0.2:
                setpoint = 30
            else:
                setpoint = 40

            throttle = throttle_PID(kph, dt)
            vehicle.control(throttle=throttle, steering=float(action), brake=0.0)
            total_loops += 1

        overall_damage = sensors['damage']["damage"]
        outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)

        if overall_damage > 0.0:
            print(f"Damage={overall_damage:2f}, exiting...")
            done = True
            break
        if outside_track:
            print("Left track, exiting...")
            done = True
            break
        if distance(vehicle.state['pos'], endpoint) < 20:
            print("Reached endpoint, exiting...")
            done = True
            break
        bng.step(1, wait=True)
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)

        if kph > 30:
            # Observe next state
            next_state = np.array(sensors['front_cam']['colour'].convert('RGB'))
            outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)
            reward = math.pow(2, 4.0 - distance_from_center)
            states.append(state.reshape(1, 3, 135, 240))
            actions.append(action_index)
            probs.append(np.log(action_probs[0, action_index]))
            rewards.append(reward)
            critic_values.append(est_rew[0,0])
        state = np.array(sensors['front_cam']['colour'].convert('RGB'))
    print(f"{total_loops=}")
    cv2.destroyAllWindows()
    # return states, actions, probs, rewards, critic, total_loops
    return states, actions, probs, rewards, critic_values, total_loops, traj

def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(math.pow(traj[i][0] - traj[i+1][0],2) + math.pow(traj[i][1] - traj[i+1][1],2) + math.pow(traj[i][2] - traj[i+1][2],2))
    return dist

def turn_X_degrees(rot_quat, degrees=90):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + degrees
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())


eps = 0.0001

def discounted_rewards(rewards,gamma=0.99,normalize=True):
    ret = []
    s = 0
    for i, r in enumerate(rewards[::-1]):
        s = r * math.pow(gamma, (len(rewards)-i))
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    return ret


def plot_durations(rewards, episode_durations, save=False, title="temp"):
    # plt.figure(2)
    plt.figure(4, figsize=(3,3), dpi=100)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    # plt.ylabel('Duration')
    plt.plot(rewards_t.numpy(), label="Rewards")
    # plt.plot(durations_t.numpy(), "k", label="Duration")
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 20:
        means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        plt.plot(means.numpy(), '--')
    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    if save:
        plt.savefig(f"training_performance-0{title}.jpg")


def train_one_epoch(model, training_set, epoch_index, device=torch.device("cpu")):
    running_loss = 0.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss() # nn.CrossEntropyLoss()
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True, num_workers=1)
    for p in range(50):
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True, device=device)
            inputs = torch.clamp(inputs + torch.randn(*inputs.shape).to(device)/10, 0, 1)
            labels = torch.tensor(labels, dtype=torch.float32, requires_grad=False, device=device)
            # print(f"{labels=}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if p % 5 == 0:
            print(f"ITER {p:03}\tloss={loss.item()}")

def train_via_actor_critic(model, action_probs, critic_values, dr, device=torch.device("cpu")):
    # Calculating loss values to update our network
    actor_losses = []
    critic_losses = []
    loss_fn = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for log_prob, value, rew in zip(action_probs, critic_values, dr):
        # When we took the action with probability `log_prob`, we received discounted reward of `rew`,
        # while critic predicted it to be `value`
        # First we calculate actor loss, to make actor predict actions that lead to higher rewards
        diff = rew - value
        actor_losses.append(-log_prob * diff)

        # The critic loss is to minimize the difference between predicted reward `value` and actual discounted reward `rew`
        print(f"{value=}\t{rew=}")
        # critic_losses.append(loss_fn(value.expand(0), rew.expand(0)))
        critic_losses.append(loss_fn(torch.Tensor([np.expand_dims(value,0)]), torch.Tensor([np.expand_dims(rew, 0)])))

    # Backpropagation
    # loss_value = sum(actor_losses) + sum(critic_losses)
    # grads = tape.gradient(loss_value, model.trainable_variables)
    # optimizer.apply_gradients(zip(grads, model.trainable_variables))
    actor_loss_sum = sum(actor_losses)
    critic_loss_sum = sum(critic_losses).numpy().item()
    loss = torch.tensor([actor_loss_sum + critic_loss_sum], dtype=torch.float32, requires_grad=True, device=device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    global base_filename, default_color, default_scenario, road_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vehicle, bng, scenario = setup_beamng(vehicle_model='hopper')
    num_episodes = 1000
    start_time = time.time()
    alpha = 1e-4
    history, durations = [], []
    trajectories = []
    screen_height, screen_width = 135, 240
    n_actions = 3
    running_reward = 0.0
    from resnet import ResNet152
    # model = ResNet152(n_actions, channels=3).to(device)
    model = DQN(screen_height, screen_width, n_actions).to(device)
    PATH = "DQRNN-v4-actorcritic"
    try:
        model.load_state_dict(torch.load(PATH))
        print(f"Loaded weights from ./{PATH}.pt")
    except:
        print("No existing weights available")
    for episode in range(num_episodes):
        states, actions, action_probs, rewards, critic_values, duration, traj = run_scenario(vehicle, bng, scenario, model=model)
        trajectories.append(traj)
        episode_reward = np.sum(rewards)
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        dr = discounted_rewards(rewards)
        print(f"Episode {episode} -> Episode Reward:{episode_reward:.2f}\tRunning reward:{running_reward:.2f}")
        train_via_actor_critic(model, action_probs, critic_values, dr, device=device)
        history.append(np.sum(rewards))
        durations.append(duration)
        plot_durations(history, durations, save=(episode % 10 == 0), title=PATH)
        plot_deviation(trajectories, "DQN", "Actor-Critic")
        torch.save(model.state_dict(), f"{PATH}.pt")
        # Log details
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode))

        if running_reward > 3000:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode))
            break
    print(f"Complete; {(time.time() - start_time)/60.0:2f} minutes elapsed")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()