import tree
import os.path
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import logging
import scipy.misc
import copy
import tensorflow as tf
import torch
import statistics, math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import csv
from ast import literal_eval
import PIL
import sys
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/DAVE2-Keras')
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
# import VAEsteer, VAE, VAEbasic
# from VAEsteer import *
from vaes import VQVAE
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/superdeepbillboard')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy/src/')
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from beamngpy import ProceduralCube
# sys.path.append(f'{args.path2src}/GitHub/superdeepbillboard')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy/src/')

# globals
default_color = 'green' #'Red'
default_scenario = 'automation_test_track' #'hirochi_raceway' #'industrial' #'automation_test_track'
road_id = "8205"
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint #50.0 #53.3 #https://en.wikipedia.org/wiki/Speed_limits_by_country
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []
steps_per_sec = 30 #100 # 36
training_file = 'metas/training_runs_{}-{}1-deletelater.txt'.format(default_scenario, road_id)

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
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif road_id == "7991": # immediately crashes into guardrail
            return {'pos': (57.229, 360.560, 128.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        elif road_id == "7846": # immediately leaves track
            return {'pos': (-456.0, -100.3, 117.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        elif road_id == "7811":
            return  {'pos': (-146.2, -255.5, 119.95), 'rot': None, 'rot_quat': turn_X_degrees((-0.021, -0.009, 0.740, 0.672), 180)}
        elif road_id == "8185": # good for saliency testing
            # return {'pos': (174.92, -289.7, 120.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            # return {'pos': (-180.4, -253.0, 120.7), 'rot': None, 'rot_quat': (-0.008, 0.004, 0.779, 0.63)}
            return {'pos': (-58.2675, -255.216, 120.175), 'rot': None, 'rot_quat': (-0.021, -0.009, 0.740, 0.672)}
        elif road_id == "8293": # immediately leaves track
            return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
        elif road_id == "8341": # dirt road
            return {'pos': (775.5, -2.2, 132.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 90)}
        # elif road_id == "8287": # actually the side of the road
        #     return {'pos': (-198.8, -251.0, 119.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)}
        # elif road_id == "7998": # actually the side of the road
        #     return {'pos': (-162.6, 108.8, 122.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 60)}
        elif road_id == "8357": # mountain road, immediately crashes into guardrail
            return {'pos': (-450.45, 679.2, 249.45), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 150)}
        elif road_id == "7770": # good candidate but actually the side of the road
            # return {'pos': (-453.42, 61.7, 117.32), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
            return {'pos': (-443.42, 61.7, 118), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8000": # immediately leaves track
            return {'pos': (-49.1, 223, 127), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -145)}
        elif road_id == "7905": # dirt road, immediately leaves track
            return {'pos': (768.2, 452.04, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -110)}
        elif road_id == "8205":
            return {'pos': (501.4, 178.6, 131.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8353":
            return {'pos': (887.2, 359.8, 159.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -20)}
        elif road_id == "7882":
            return {'pos': (-546.8, 568.0, 199.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 155)}
        elif road_id == "8179":
            return {'pos': (-738.1, 257.3, 133.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 150)}
        # elif road_id == "8248": # actually the side of the road
        #     return {'pos': (-298.8, 13.6, 118.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 180)}
        # elif road_id == "7768": # actually the side of the road
        #     return {'pos': (-298.8, 13.6, 118.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 180)}
        # elif road_id == "7807": # actually the side of the road
        #     return {'pos': (-251.2, -260.0, 119.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "8049":  # actually the side of the road
        #     return {'pos': (-405.0, -26.6, 117.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -110)}
        elif road_id == 'starting line 30m down':
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
            return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
        elif road_id == 'highway': #(open, farm-like)
            return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
        elif road_id == 'highwayopp': # (open, farm-like)
            return {'pos': (-542.719,-251.721,117.083), 'rot': None, 'rot_quat': (0.0098941307514906,0.0096141006797552,0.72146373987198,0.69231480360031)}
        elif road_id == 'default':
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
            # return {'pos': (-3, 230.0, 26.2), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        # elif road_id == "9156":
        #     return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif road_id == "9119":
            return {"pos": (-452.972, 16.0, 29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 170)}
        elif road_id == "9167":
            return {'pos': (105.3, -96.4, 25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 40)}
        elif road_id == "9156":
            return {'pos': (-376.25, 200.8, 25.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 45)}
            # return {'pos': (-379.184,208.735,25.4121), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
        elif road_id == "9189":
            return {'pos': (-383.498, 436.979, 32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
        elif road_id == "9202": # lanelines
            return {'pos': (-315.2, 80.94, 32.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
        elif road_id == "9062":
            return {'pos': (-315.2, 80.94, 32.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
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
        return {'pos': (-10.0, 580.73, 156.8), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}

def setup_sensors(vehicle):
    # Set up sensors
    # pos = (-0.3, 1, 1.0) # default
    # pos = (-0.5, 2, 1.0) #center edge of hood
    # pos = (-0.5, 1, 1.0)  # center middle of hood
    # pos = (-0.5, 0.4, 1.0)  # dashboard
    # pos = (-0.5, 0.38, 1.5) # roof
    # pos = (-0.5, 0.38, 1.3) # windshield
    # pos = (-0.5, 0.38, 1.7) # above windshield
    # direction = (0, 1.0, -0.275)
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    # camera_zs = [1.3] #[0.87, 1.07, 1.3, 1.51, 1.73]
    # with open(training_file, 'w') as f:
    #     f.write("CAMERA_DIR,CAMERA_PITCH_EULER,CAMERA_POS,RUNTIME,DISTANCE,AVG_STDEV,RUN_SCORE\n")
    #     for z in camera_zs:
    pos = (-0.5, 0.38, 1.3)
    direction = (0, 1.0, 0)
    front_camera = Camera(pos, direction, fov, resolution,
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
    # kp = 0.001; ki = 0.00001; kd = 0.0001
    # kp = .3; ki = 0.01; kd = 0.1
    # kp = 0.15; ki = 0.0001; kd = 0.008 # worked well but only got to 39kph
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

def diff_damage(damage, damage_prev):
    new_damage = 0
    if damage is None or damage_prev is None:
        return 0
    new_damage = damage['damage'] - damage_prev['damage']
    return new_damage

# takes in 3D array of sequential [x,y]
# produces plot
def plot_deviation(trajectories, model, deflation_pattern, centerline):
    i = 0; x = []; y = []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, label="Centerline")
    for t in trajectories:
        x = []; y = []
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, label="Run {}".format(i))
        i += 1
    # plt.xlabel('x - axis')
    # plt.ylabel('y - axis')
    # Set a title of the current axes.
    plt.title('Trajectories with {} {}'.format(model, deflation_pattern))
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    plt.pause(0.1)
    return

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

def plot_racetrack_roads(roads, bng):
    global default_scenario, road_id
    print("Plotting scenario roads...")
    sp = spawn_point()
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    selectedroads = []
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        add = True
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        s = sum(dists)
        if (s < 200):
            continue
        for edge in road_edges:
            # if edge['middle'][0] < -250 or edge['middle'][0] > 50 or edge['middle'][1] < 0 or edge['middle'][1] > 300:
            if edge['middle'][1] < -50 or edge['middle'][1] > 250:
                add = False
                break
            if add:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
        if add:
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
            selectedroads.append(road)
    for r in selectedroads: # ["8179", "8248", "8357", "8185", "7770", "7905", "8205", "8353", "8287", "7800", "8341", "7998"]:
        a = bng.get_road_edges(r)
        print(r, a[0]['middle'])
    plt.plot([sp['pos'][0]], [sp['pos'][1]], "bo")
    plt.legend()
    plt.show()
    plt.pause(0.001)


def road_analysis(bng):
    global centerline, roadleft, roadright
    global road_id, default_scenario, road_id
    # plot_racetrack_roads(bng.get_roads(), bng)
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    actual_middle = [edge['middle'] for edge in edges]
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/4.0 for edge in edges]
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
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.savefig(f"{title}.jpg")
    plt.show()
    plt.pause(0.1)

def create_ai_line_from_road(spawn, bng, road_id="7982"):
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []
    middle = road_analysis(bng)
    middle_end = middle[:3]
    middle = middle[3:]
    middle.extend(middle_end)
    traj = []
    with open("centerline_lap_data.txt", 'w') as f:
        for i,p in enumerate(middle[:-1]):
            f.write("{}\n".format(p))
            # interpolate at 1m distance
            if distance(p, middle[i+1]) > 1:
                y_interp = scipy.interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
                num = abs(int(middle[i+1][0] - p[0]))
                xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                for x,y in zip(xs,ys):
                    traj.append([x,y])
                    line.append({"x":x, "y":y, "z":p[2], "t":i * 10})
                    points.append([x, y, p[2]])
                    point_colors.append([0, 1, 0, 0.1])
                    spheres.append([x, y, p[2], 0.25])
                    sphere_colors.append([1, 0, 0, 0.8])
            else:
                traj.append([p[0],p[1]])
                line.append({"x": p[0], "y": p[1], "z": p[2], "t": i * 10})
                points.append([p[0], p[1], p[2]])
                point_colors.append([0, 1, 0, 0.1])
                spheres.append([p[0], p[1], p[2], 0.25])
                sphere_colors.append([1, 0, 0, 0.8])
    #         plot_trajectory(traj, "Points on Script So Far")
    # plot_trajectory(traj, "Planned traj.")
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng

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
    for i in range(4):
        centerline.extend(copy.deepcopy(centerline))
        remaining_centerline.extend(copy.deepcopy(remaining_centerline))
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng

# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    dist = min(distance_from_centerline)
    return dist > 9.0, dist

def add_qr_cubes(scenario):
    global qr_positions
    qr_positions = []
    with open(f'posefiles/qr_box_locations-{road_id}-swerve0.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "platform=" in line:
                line = line.replace("platform=", "")
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                size= float(line[2])
                cube = ProceduralCube(name='cube_platform',
                                      pos=pos,
                                      rot=None,
                                      rot_quat=rot_quat,
                                      size=(2, size, 0.5))
                scenario.add_procedural_mesh(cube)
            else:
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
                box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
                                      rot_quat=rot_quat, scale=(1,1,1), JBeam = 'qrbox2', datablock="default_vehicle")
                scenario.add_object(box)

def setup_beamng(vehicle_model='etk800'):
    global base_filename, default_color, default_scenario, road_id, steps_per_sec, road_id

    random.seed(1703)
    setup_logging()

    beamng = BeamNGpy('localhost', 64256, home='C:/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user='C:/Users/Meriel/Documents/BeamNG.research')
    # beamng = BeamNGpy('localhost', 64256, home='C:/Users/Meriel/Documents/BeamNG.tech.v0.21.3.0', user='C:/Users/Meriel/Documents/BeamNG.tech')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model,
                      licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle)
    spawn = spawn_point()
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat']) #, partConfig=parts_config)
    add_barriers(scenario)
    # add_qr_cubes(scenario)
    # eagles_eye_cam = Camera((221.854, -128.443, 165.5),
    #                         (0.013892743289471, -0.015607489272952, -1.39813470840454, 0.91656774282455),
    #                         fov=90, resolution=(1500,1500),
    #                       colour=True, depth=True, annotation=True)
    # scenario.add_camera(eagles_eye_cam, "eagles_eye_cam")
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    bng.set_deterministic()
    bng.set_steps_per_second(steps_per_sec)
    bng.load_scenario(scenario)
    bng.start_scenario()
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn, bng)
    bng.pause()
    assert vehicle.skt
    # bng.resume()
    return vehicle, bng, scenario

def run_scenario_with_VAE(vehicle, bng, scenario, model, vehicle_model='etk800', run_number=0,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), vae=None, vae_type="torch"):
    global base_filename, default_color, default_scenario, road_id, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()
    # collect overhead view of setup
    # freecams = scenario.render_cameras()
    # plt.title("freecam")
    # plt.imshow(freecams['eagles_eye_cam']["colour"].convert('RGB'))
    # freecams['eagles_eye_cam']["colour"].convert('RGB').save("eagles-eye-view.jpg", "JPEG")
    plt.pause(0.01)

    # perturb vehicle
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_cam']['colour'].convert('RGB')
    pitch = vehicle.state['pitch'][0]
    roll = vehicle.state['roll'][0]
    z = vehicle.state['pos'][2]
    spawn = spawn_point()

    wheelspeed = 0.0; throttle = 0.0; prev_error = setpoint; damage_prev = None; runtime = 0.0
    kphs = []; traj = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = None; overall_damage = 0.0
    final_img = None
    total_loops = 0; total_imgs = 0; total_predictions = 0
    start_time = sensors['timer']['time']
    outside_track = False
    distance_from_center = 0
    writedir = f"{default_scenario}-{road_id}-lap-test"
    if not os.path.isdir(writedir):
        os.mkdir(writedir)
    with open(f"{writedir}/data.txt", "w") as f:
        f.write(f"IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT\n")

        while overall_damage <= 0:
            # collect images
            vehicle.update_vehicle()
            sensors = bng.poll_sensors(vehicle)
            image = sensors['front_cam']['colour'].convert('RGB')
            image_seg = sensors['front_cam']['annotation'].convert('RGB')
            cv2.imshow('car view', np.array(image)[:, :, ::-1])
            cv2.waitKey(1)
            # cv2.imshow('segmentation', np.array(image_seg)[:, :, ::-1])
            # cv2.waitKey(1)
            total_imgs += 1
            kph = ms_to_kph(sensors['electrics']['wheelspeed'])
            dt = (sensors['timer']['time'] - start_time) - runtime
            processed_img = model.process_image(image).to(device)
            prediction = model(processed_img)
            steering = float(prediction.item())

            if False: # distance_from_center > 4.0:
                if vae_type == "torch":
                    rectified_image = vae(processed_img)
                    rectimg = rectified_image[0].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                else:
                    rectimg = np.asarray(image)[None]
                    rectimg = tf.image.resize(rectimg, [136, 240])
                    rectimg = VQVAE.cast_and_normalise_images(rectimg)
                    rectimg = vae(rectimg, is_training=False)['x_recon']
                    rectimg = rectimg + 0.5
                    rectimg = rectimg.numpy()[0]

                cv2.imshow('rectified_img_downres', rectimg[:, :, ::-1])
                cv2.waitKey(1)
                print(f"orig. prediction={prediction.item():3f}")
                processed_img = model.process_image(rectimg).to(device)
                prediction_vae = model(processed_img)
                print(f"vae prediction={prediction_vae.item():3f}")
                steering = float(prediction_vae.item())

            runtime = sensors['timer']['time'] - start_time

            # control params
            total_predictions += 1
            position = str(vehicle.state['pos']).replace(",", " ")
            orientation = str(vehicle.state['dir']).replace(",", " ")

            image.save(f"{writedir}/sample-{total_imgs:05d}.jpg", "JPEG")
            image_seg.save(f"{writedir}/sample-segmented-{total_imgs:05d}.jpg", "JPEG")
            f.write(f"sample-{total_imgs:05d}.jpg,{prediction.item()},{position},{orientation},{kph},{sensors['electrics']['steering']}\n")
            if abs(steering) > 0.2:
                setpoint = 30
            else:
                setpoint = 40
            throttle = throttle_PID(kph, dt)
            vehicle.control(throttle=throttle, steering=steering, brake=0.0)
            # 17.2,233.2,26.2 0.0049366145394742,-0.0014738570898771,-0.72177958488464,0.69210386276245
            steering_inputs.append(steering)
            throttle_inputs.append(throttle)
            timestamps.append(runtime)

            damage = sensors['damage']
            overall_damage = damage["damage"]
            new_damage = diff_damage(damage, damage_prev)
            damage_prev = damage
            vehicle.update_vehicle()
            traj.append(vehicle.state['pos'])

            kphs.append(ms_to_kph(wheelspeed))
            total_loops += 1
            final_img = image

            if new_damage > 0.0:
                print("New damage={}, exiting...".format(new_damage))
                break
            bng.step(1, wait=True)

            if distance(spawn['pos'], vehicle.state['pos']) < 5 and runtime > 10:
                print("Completed one lap, exiting...")
                reached_start = True
                break

            outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)
            print(f"{distance_from_center=:.1f}")
            if outside_track:
                print("Left track, exiting...")
                break

    cv2.destroyAllWindows()

    print("Total predictions: {} \nexpected predictions ={}*{}={}".format(total_predictions, round(runtime,3), steps_per_sec, round(runtime*steps_per_sec,3)))
    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj, 'pitch': round(pitch,3),
               'roll':round(roll,3), "z":round(z,3), 'final_img':final_img, 'deviation':deviation
               }
    return results

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

def add_barriers(scenario):
    barrier_locations = []
    with open('posefiles/hirochi_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            rot_quat = turn_X_degrees(rot_quat, degrees=-115)
            # barrier_locations.append({'pos':pos, 'rot_quat':rot_quat})
            ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            scenario.add_object(ramp)

def main():
    global base_filename, default_color, default_scenario, road_id
    vae_type = "tensorflow"
    # model_name = "/mnt/c/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/dave2/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    model_name = "dave2/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    model = torch.load(model_name, map_location=torch.device('cpu')).eval()
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if vae_type == "torch":
        vae_name = "/mnt/c/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/vae/VAEsteer-4thattempt-nobatchnorm.pt"
        vae = torch.load(vae_name, map_location=torch.device('cpu')).eval()
        vae = vae.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        vae = VQVAE.instantiate_VQVAE()
        print(vae)
    vehicle, bng, scenario = setup_beamng(vehicle_model='hopper')

    for i in range(1):
        results = run_scenario_with_VAE(vehicle, bng, scenario, model, vehicle_model='hopper', run_number=i, vae=vae, vae_type=vae_type)
        results['distance'] = get_distance_traveled(results['traj'])
        plot_trajectory(results['traj'], f"{default_scenario}-{model._get_name()}-{road_id}-runtime{results['runtime']:.2f}-dist{results['distance']:.2f}")
    bng.close()


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()