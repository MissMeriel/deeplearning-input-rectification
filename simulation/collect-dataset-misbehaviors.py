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
from PIL import Image
import sys

sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/DAVE2-Keras')
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
from models.CrashNet import Model
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy/src/')
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

# globals
default_color = 'green' #'Red'
default_scenario = 'automation_test_track'
road_id = "8185"
# default_scenario, road_id = "italy", "24132"
integral = 0.0
prev_error = 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75
centerline, centerline_interpolated = [], []
roadleft, roadright = [], []
steps_per_sec = 30

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
        if road_id == '8127': # dirt road
            return {'pos': (121.2, -314.8, 123.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 20)}
        elif road_id == "8304": # big bridge/tunnel road, concrete walls
            # return {'pos': (357.2, 741.5, 132.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 60)}
            return {'pos': (324.593,655.58,132.642), 'rot': None, 'rot_quat': (-0.007, 0.005, 0.111, 0.994)}
        elif road_id == "8301": # country road next to river
            return {'pos': (262.0, -289.3, 121), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
        elif road_id == "8394": # long straight tree-lined country road near gas stn.
            return {'pos': (-582.0, -249.2, 117.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
        elif road_id == "7991": # long road, spawns atop overpass bridge
            return {'pos': (57.229, 360.560, 128.203), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        elif road_id == "8293": # winding mountain road, lanelines
            return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
        elif road_id == "8205": # starting line, same as default
            # return {'pos': (501.36,178.62,131.69), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)} # orig
            return {'pos': (517.08, 178.84, 132.2), 'rot': None, 'rot_quat': (-0.0076747848652303, -0.0023407069966197, -0.70286595821381, 0.71127712726593)}  # closer
        elif road_id == "8185": # bridge, lanelines
            return {'pos': (174.92,-289.67,120.67), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)}
        # elif road_id == "": #
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)}
        else: # default
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
            return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036, 0.0065, 0.9234, -0.3837)}
        elif road_id == "racetrackstraightaway":
            return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.0105, 0.02997, -0.4481, 0.8934)}
        elif road_id == "racetrackcurves":
            return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.0290,0.0222,0.9860,0.1626)}
    elif default_scenario == "hirochi_raceway":
        if road_id == "9039": # good candidate for input rect.
            # return {'pos': (290.558, -277.280, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -130)} #start
            # return {'pos': (490.073, -154.12, 35.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -90)}  # 2nd quarter
            # centerline[ 614 ]: [557.3650512695312, -90.57571411132812, 43.21120071411133]
            # return {'pos': (557.4, -90.6, 43.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 35)}  # 2nd quarter
            # centerline[ 761 ]: [9.719991683959961, 342.0410461425781, 31.564104080200195]
            # return {'pos': (9.72, 342.0, 31.75), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 130)}  # 3rd quarter
            #centerline[ 1842 ]: [220.56719970703125, 365.5675048828125, 35.992027282714844]
            # return {'pos': (220.6, 365.6, 36.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 130)}  # 3rd quarter
            # centerline[1900]: [-32.84414291381836, 386.8899841308594, 36.25067901611328]
            # return {'pos': (-32.8, 386.9, 36.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 15)}  # 3rd quarter
            # centerline[1909]: [-45.08585739135742, 414.32073974609375, 38.64292526245117]
            # return {'pos': (-45.1, 414.3, 38.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 0)}  # 3rd quarter
            # centerline[2000]: [523.5741577148438, -135.5963134765625, 38.25138473510742]
            return {'pos': (523.6, -135.6, 38.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 90)}  # 3rd quarter
            # # centerline[4479]: [-346.49896240234375, 431.0029296875, 30.750564575195312]
            # return {'pos': (-346.5, 431.0, 30.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 0)}  # 3rd quarter
        elif road_id == "9205":
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif road_id == "9119":
            # return {"pos": (-452.972, 16.0, 29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 200)} # before tollbooth
            # return {"pos": (-452.972, 64.0, 30.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 200)} # past tollbooth
            # centerline[ 150 ]: [-548.0482177734375, 485.4112243652344, 32.8107795715332]
            return {"pos": (-548.0, 485.4, 33.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -100)}
            # centerline[ 300 ]: [-255.45079040527344, 454.82879638671875, 28.71044921875]
            return {"pos": (-255.5, 454.8, 28.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            # centerline[ 500 ]: [-317.8174743652344, 459.4542236328125, 31.227020263671875]
            return {"pos": (-317.8, 459.5, 31.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -50)}
            # centerline[ 1000 ]: [-421.6916809082031, 508.9856872558594, 36.54324722290039]
            return {"pos": (-421.7, 509, 36.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            # centerline[ 1647 ]: [-236.64036560058594, 428.26605224609375, 29.6795597076416]
            return {"pos": (-236.6, 428.3, 29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9167":
            # veers away onto dirt road
            # return {'pos': (105.3, -96.4, 25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 40)}
            # centerline[ 150 ]: [187.20333862304688, -146.42086791992188, 25.565567016601562]
            # return {'pos': (187.2, -146.4, 25.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 160)}
            # centerline[ 300 ]: [152.4763641357422, -257.7218933105469, 29.21633529663086]
            return {'pos': (152.5, -257.7, 29.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
            # centerline[ 1983 ]: [279.4185791015625, -261.2400817871094, 47.39253234863281]
            return {'pos': (279.4, -261.2, 47.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9156":  # starting line
            # return {'pos': (-376.25, 200.8, 25.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 45)} # crashes into siderail
            # return {'pos': (-374, 90.2, 28.75), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 15)}  # past siderail
            return {'pos': (-301.314,-28.4299,32.9049), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 55)}  # past siderail
            # return {'pos': (-379.184,208.735,25.4121), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif road_id == "9189":
            # return {'pos': (-383.498, 436.979, 32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
            return {'pos': (-447.6, 468.22, 32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
        elif road_id == "9202": # lanelines
            # return {'pos': (-315.2, 80.94, 32.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
            # centerline[150]: [233.89662170410156, 88.48623657226562, 25.24519157409668]
            # return {'pos': (233.9, 88.5, 25.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)}
            # centerline[300]: [-244.8330078125, -9.06863784790039, 25.405052185058594]
            return {'pos': (-244.8, -9.1, 25.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)}
            # centerline[ 4239 ]: [-9.485580444335938, 17.10186004638672, 25.56268310546875]
        elif road_id == "9062":
            # return {'pos': (24.32, 231, 26.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)} # start
            # return {'pos': (155.3, 119.1, 25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 70)} # middle
            return {'pos': (-82.7, 10.7, 25.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 110)} # end
        elif road_id == "9431": # actually a road edge
            return {'pos': (204.34,-164.94,25.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9033": # road around a tan parking lot
            return {'pos': (-293.84,225.67,25.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
        elif road_id == "9055": # roadside edge
            return {'pos': (469.74,122.98,27.45), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9064": # narrow, good for input rect, similar to orig track
            return {'pos': (-117.6, 201.2, 25.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 110)}
            return {'pos': (-93.30,208.40,25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 110)} # start, crashes immediately
        elif road_id == "9069": # narrow, past dirt patch, lots of deviations
            return {'pos': (-77.27,-135.96,29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9095": # good for input rect, well formed, long runtime w/o crash
            return {'pos': (-150.15,174.55,32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 180)}
        elif road_id == "9129": # good for input rect, long runtime w/o crash
            return {'pos': (410.96,-85.45,30.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
        elif road_id == "9189": # narrow, edged by fence, early crash
            return {'pos': (-383.50,436.98,32.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9204": # lanelines, near tan parking lot as 9033, turns into roadside edge
            return {'pos': (-279.37,155.85,30.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -110)}
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
    elif default_scenario == 'italy':
        if road_id == "default":
            return {'pos': (729.631,763.917,177.997), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}
        elif road_id == "22486": # looks good for input rect but crashes immediately, narrow road
            return {'pos': (-723.4, 631.6, 266.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)} # slightly ahead of start,crashes immediately
            return {'pos': (-694.47,658.12,267.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)} # start, crashes immediately
        elif road_id == "22700": # narrow
            return {'pos': (-1763.44,-1467.49,160.46), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)} # start, crashes immediately
        elif road_id == "22738": # narrow, crashes almost immediately into left low wall
            return {'pos': (1747.30,-75.16,150.66), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 60)}
        elif road_id == "22752":
            return {'pos': (1733.98,-1263.44,170.05), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22762":
            return {'pos': (1182.45,-1797.33,171.59), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22763":
            return {'pos': (-1367.21,759.51,307.28), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22815":
            return {'pos': (-388.07,-558.00,407.62), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22831":
            return {'pos': (313.95,-1945.95,144.11), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22889":
            return {'pos': (-1349.51,760.28,309.26), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22920":
            return {'pos': (1182.45,-1797.33,171.59), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23106":
            return {'pos': (1182.94,-1801.46,171.71), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23113":
            return {'pos': (-1478.87,25.52,302.22), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23132":
            return {'pos': (1513.50,814.87,148.05), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23193":
            return {'pos': (-1542.79,730.60,149.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23199":
            return {'pos': (1224.43,132.39,255.65), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23209":
            return {'pos': (-578.08,1215.80,162.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 100)}
        elif road_id == "23217":
            return {'pos': (-1253.26,471.33,322.71), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23325":
            return {'pos': (1348.08,834.64,191.01), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23398":
            return {'pos': (-1449.63,-302.06,284.02), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23447":
            return {'pos': (-326.59,-838.56,354.59), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23514":
            return {'pos': (-584.13,1209.96,162.43), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23548":
            return {'pos': (927.49,19.38,213.93), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23552":
            return {'pos': (546.33,-871.00,264.50), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23629":
            return {'pos': (100.73,-588.97,331.22), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23665":
            return {'pos': (1265.40,932.52,160.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 150)}
        elif road_id == "23667":
            return {'pos': (-1769.06,-1128.61,142.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23689":
            return {'pos': (276.16,-1435.78,406.12), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23821":
            return {'pos': (-1013.98,402.76,290.71), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23833":
            return {'pos': (-322.82,-1752.61,157.55), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23858":
            return {'pos': (-1896.97,-66.76,148.80), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23898":
            return {'pos': (1718.98,1225.72,217.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -135)}
        elif road_id == "23914":
            return {'pos': (1525.70,813.31,148.26), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23966":
            return {'pos': (1501.21,805.18,147.48), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23978":
            return {'pos': (-1902.02,1567.42,152.42), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23987":
            return {'pos': (-491.94,-770.24,489.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 100)}
        elif road_id == "24007":
            return {'pos': (-68.48,-935.31,463.87), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24029":
            return {'pos': (-585.00,1860.96,152.41), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24043":
            return {'pos': (559.85,884.34,172.88), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 210)}
        # italy roads 400<=length<600
        elif road_id == "21587":
            return {'pos': (265.43,-971.06,270.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "21592":
            return {'pos': (265.78,-890.68,247.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -90)}
        elif road_id == "22518":
            return {'pos': (1754.05,-1284.92,170.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22645":
            return {'pos': (-260.08,395.71,179.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 40)}
        elif road_id == "22654":
            return {'pos': (704.64,212.57,178.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22674":
            return {'pos': (1444.18,-1560.81,164.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22713":
            return {'pos': (1754.70,-1288.44,170.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22744":
            return {'pos': (-456.62,-1355.66,197.83), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "22853":
            return {'pos': (-1753.30,1295.95,139.12), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607),45)}
        elif road_id == "22927":
            return {'pos': (626.50,-1489.26,332.41), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -50)}
        elif road_id == "23037":
            return {'pos': (1105.51,1371.28,139.49), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23103":
            return {'pos': (-1431.80,-253.16,287.63), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23155":
            return {'pos': (-152.73,5.62,259.82), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23189":
            return {'pos': (1754.70,-1288.44,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23197":
            return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23210":
            return {'pos': (-1248.98,-1096.07,587.78), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 80)}
        elif road_id == "23289":
            return {'pos': (1444.18,-1560.81,164.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23308":
            return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23327":
            return {'pos': (720.06,-886.29,216.04), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23346":
            return {'pos': (1252.73,622.00,201.92), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23497":
            return {'pos': (-948.37,879.28,385.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -150)}
        elif road_id == "23670":
            return {'pos': (102.15,518.06,179.00), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
            # centerline[ 23 ]: [78.65234375, 515.7720947265625, 178.9859619140625]
        elif road_id == "23706":
            return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23740":
            return {'pos': (8.14,-557.07,326.49), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 150)}
        elif road_id == "23751":
            return {'pos': (1444.18,-1560.81,164.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23827":
            return {'pos': (-129.24,655.54,192.31), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23949":
            return {'pos': (-1454.84,42.73,303.24), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "23997":
            return {'pos': (431.93,1.94,180.92), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24050":
            return {'pos': (-310.99,-1865.30,160.74), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -98)}
        elif road_id == "24132":
            return {'pos': (-221.87,-1927.10,156.89), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -70)}
        elif road_id == "24217":
            return {'pos': (529.43,616.03,178.00), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24271":
            return {'pos': (1754.70,-1288.44,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24296":
            return {'pos': (-1328.52,1501.30,164.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -65)}
        elif road_id == "24323":
            return {'pos': (64.70,174.23,202.21), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24327":
            return {'pos': (68.60,178.10,202.52), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24332":
            return {'pos': (-1560.53,-255.65,230.70), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24347":
            return {'pos': (1211.34,390.34,235.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24429":
            return {'pos': (1439.79,834.08,208.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24538":
            return {'pos': (884.74,751.11,180.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24597":
            return {'pos': (1324.57,838.79,189.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24619":
            return {'pos': (-336.69,50.83,210.65), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24624":
            return {'pos': (577.46,140.94,183.53), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24642":
            return {'pos': (-1560.53,-255.65,230.70), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24722":
            return {'pos': (-1177.30,-721.74,406.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24878":
            return {'pos': (839.88,1281.43,147.02), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "24931":
            return {'pos': (1440.93,-1559.53,165.00), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 65)}
        elif road_id == "25085":
            return {'pos': (264.33,-586.83,345.14), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25219":
            return {'pos': (464.91,361.13,172.24), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25430":
            return {'pos': (-1114.47,-1533.54,164.19), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25434":
            return {'pos': (571.5, 1234.11, 176.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -215)}
            # 505.523,1254.41,173.275, (-0.02110549248755,0.022285981103778,0.83397573232651,0.55094695091248)
            return {'pos': (569.54,1220.11,176.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -140)}
        elif road_id == "25444":
            return {'pos': (493.77,50.73,187.70), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 50)}
        elif road_id == "25505":
            return {'pos': (344.05,-1454.35,428.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -75)}
        elif road_id == "25509":
            return {'pos': (1259.58,918.19,160.25), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 140)}
        elif road_id == "25511":
            # return {'pos': (567.4, 895.0, 172.83), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 180)}
            return {'pos': (554.4,914.2,170), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 180)}
        elif road_id == "25547":
            return {'pos': (1688.07,-1075.33,152.91), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25555":
            return {'pos': (-1867.60,-130.75,149.08), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25573":
            return {'pos': (-1326.46,1513.28,164.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -50)}
        elif road_id == "25622":
            return {'pos': (-1719.26,106.23,148.42), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25721":
            return {'pos': (-831.32,-1364.53,142.14), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25741":
            return {'pos': (819.14,195.64,186.95), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25893":
            return {'pos': (878.10,745.78,180.86), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "25923":
            return {'pos': (1257.99,1006.8, 175.01), 'rot': None,'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
        elif road_id == "25944":
            return {'pos': (310.03,1802.97,207.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 45)}
        elif road_id == "26124":
            return {'pos': (1182.59,1239.12,148.01), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26127":
            return {'pos': (-1177.30,-721.74,406.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26139":
            return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26214":
            return {'pos': (-162.74,-155.10,233.43), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26311":
            return {'pos': (-169.32,-159.67,234.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26360":
            return {'pos': (-1177.30,-721.74,406.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26404":
            return {'pos': (1211.34,390.34,234.94), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -100)}
        elif road_id == "26425":
            return {'pos': (15.93,-1726.24,332.41), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26464":
            return {'pos': (-479.20,-1624.81,143.42), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -180)}
        elif road_id == "26592":
            return {'pos': (1256.22,915.62,160.28), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "26599":
            return {'pos': (1467.54,-1546.11,163.82), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "":
            return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}


def setup_sensors(vehicle):
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    # camera_zs = [1.3] #[0.87, 1.07, 1.3, 1.51, 1.73]
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
    plt.title('Trajectories with {} {}'.format(model, deflation_pattern))
    plt.legend()
    plt.show()
    plt.pause(0.1)

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
    sp = spawn_point()
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    print(f"{len(roads)=}")
    tt = []
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp, y_temp = [], []
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        s = sum(dists)
        if s >= 300: # and s < 600:
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
            print(f"road_id={road} x,y,z={x_temp[0]:.2f},{y_temp[0]:.2f},{road_edges[0]['middle'][2]:.2f}")
            tt.append(f"road_id={road} x,y,z={x_temp[0]:.2f},{y_temp[0]:.2f},{road_edges[0]['middle'][2]:.2f}")
    import re
    p = re.compile("road_id=[0-9]+")
    tt.sort(key=lambda s: int(p.match(s).group().replace("road_id=", "")))
    print("\n\n\n")
    for i in tt:
        print(i)
    plt.plot([sp['pos'][0]], [sp['pos'][1]], "bo")
    plt.legend(ncol=2)
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
    i = np.where(distance_from_centerline == dist)
    # print("vehicle_pos", vehicle_pos)
    # print("centerline[",i[0][0],"]:", centerline[i[0][0]])
    # print("centerline[", 150, "]:", centerline[150])
    # print("centerline[", 300, "]:", centerline[300])
    print("centerline[", len(centerline)-1, "]:", centerline[-1])
    return dist > 9.0, dist


def add_qr_cubes(scenario):
    global qr_positions
    qr_positions = []
    with open('posefiles/qr_box_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
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
    # beamng = BeamNGpy('localhost', 64256, home='/mnt/c/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user='/mnt/c/Users/Meriel/Documents/BeamNG.research')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model,
                      licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle)
    spawn = spawn_point()
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat'])
    add_barriers(scenario)
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

def run_scenario(vehicle, bng, scenario, model, vehicle_model='etk800', run_number=0,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), det_model=None):
    global base_filename, default_color, default_scenario, road_id, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()
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
    writedir = f"../CrashNet-traces2/{default_scenario}-{road_id}-lap-CrashNet"
    if not os.path.isdir(writedir):
        os.mkdir(writedir)
    with open(f"{writedir}/data.txt", "w") as f:
        tracked_secs = 1
        # f.write(f"IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT,DIST_1STEP,DIST_3STEP,DIST_5STEP,DIST_1SEC,DIST_3SEC\n")
        f.write(f"IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT,DISTS\n")
        past_dists = np.zeros((tracked_secs * steps_per_sec))
        past_imgs = np.zeros((tracked_secs * steps_per_sec, 135, 240, 3), dtype=np.uint8)
        while overall_damage <= 0:
            vehicle.update_vehicle()
            sensors = bng.poll_sensors(vehicle)
            runtime = sensors['timer']['time'] - start_time
            kph = ms_to_kph(sensors['electrics']['wheelspeed'])
            dt = (sensors['timer']['time'] - start_time) - runtime
            image = sensors['front_cam']['colour'].convert('RGB')
            image_seg = sensors['front_cam']['annotation'].convert('RGB')
            cv2.imshow('car view', np.array(image)[:, :, ::-1])
            cv2.waitKey(1)
            # cv2.imshow('segmentation', np.array(image_seg)[:, :, ::-1])
            # cv2.waitKey(1)
            # control params
            if kph < 35:
                steering = 0
            else:
                total_imgs += 1

                processed_img = model.process_image(image).to(device)
                prediction = model(processed_img)
                print(f"{prediction.item()=:.3f}")
                steering = float(prediction.item())
                total_predictions += 1
            position = str(vehicle.state['pos']).replace(",", " ")
            orientation = str(vehicle.state['dir']).replace(",", " ")
            if det_model is not None:
                processed_det_img = det_model.process_image(image).to(device)
                det_prediction = det_model(processed_det_img)
                print(f"{det_prediction[0][0]=}\n{det_prediction[0][-1]=}")
            if runtime > tracked_secs:
                pil_image = Image.fromarray(past_imgs[-1])
                # img_filename = f"sample-{total_imgs-tracked_secs*steps_per_sec:05d}.jpg"
                # print(f"Image saved to: sample-{total_imgs-tracked_secs*steps_per_sec:05d}.jpg")
                # pil_image.save(f"{writedir}/{img_filename}", "JPEG")
                # dists_str = str(past_dists).replace("\n", " ")
                # f.write(f"{img_filename},{prediction.item()},{position},{orientation},{kph},{sensors['electrics']['steering']},"
                #         f"{dists_str}\n")
            if abs(steering) > 0.2:
                setpoint = 30
            else:
                setpoint = 40
            throttle = throttle_PID(kph, dt)
            vehicle.control(throttle=throttle, steering=steering, brake=0.0)
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
            past_dists = np.roll(past_dists, 1)
            past_dists[0] = distance_from_center
            past_imgs = np.roll(past_imgs, 1, axis=0)
            past_imgs[0] = np.array(image)
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
    with open('../simulation/posefiles/hirochi_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            rot_quat = turn_X_degrees(rot_quat, degrees=-115)
            # barrier_locations.append({'pos':pos, 'rot_quat':rot_quat})
            # add barrier to scenario
            ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            scenario.add_object(ramp)

def main():
    global base_filename, default_color, default_scenario, road_id
    model_name = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()
    model = model.to(device)
    # detection model
    det_model = torch.load("../models/weights/CrashNet/CrashNet-final-archadj.pt", map_location=device).eval()
    vehicle, bng, scenario = setup_beamng(vehicle_model='hopper')
    for i in range(1):
        results = run_scenario(vehicle, bng, scenario, model, vehicle_model='hopper', run_number=i, det_model=det_model)
        results['distance'] = get_distance_traveled(results['traj'])
        plot_trajectory(results['traj'], f"{model._get_name()}-{default_scenario}-{road_id}-40kph-runtime{results['runtime']:.2f}-distance{results['distance']:.2f}")
    bng.close()


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()