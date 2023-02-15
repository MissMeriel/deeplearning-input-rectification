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
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/superdeepbillboard')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy/src/')
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from beamngpy import ProceduralCube
# sys.path.append(f'{args.path2src}/GitHub/superdeepbillboard')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy/src/')
from wand.image import Image as WandImage

# globals
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint #50.0 #53.3 #https://en.wikipedia.org/wiki/Speed_limits_by_country
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []
training_file = "" #"'metas/training_runs_{}-{}1-deletelater.txt'.format(default_scenario, road_id)


# positive angle is to the right / clockwise
def spawn_point(default_scenario, road_id, reverse=False, seg=1):
    global lanewidth
    if default_scenario == 'cliff':
        #return {'pos':(-124.806, 142.554, 465.489), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
    elif default_scenario == 'west_coast_usa':
        # surface road (crashes early af)
        if road_id == "13242":
            return {'pos': (-733.7, -923.8, 163.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, 0.805, 0.592), -20)}
        elif road_id == "8650":
            if reverse:
                return {'pos': (-358.719,-846.965,136.99), 'rot': None, 'rot_quat': turn_X_degrees((-0.0075049293227494,-0.014394424855709,0.32906526327133,0.9441676735878), 7)}
            # yellow lanelines, left curve, driving on left
            return {'pos': (-365.24, -854.45, 136.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 90)}
        elif road_id == "12667":
            # lanelines, uphill left curve
            return {'pos': (-892.4, -793.4, 114.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 70)}
        elif road_id == "8432":
            # lanelines, wide 2direction highway, long left curve
            if reverse:
                return {'pos': (-871.9, -803.2, 115.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 165 - 180)}
            return {'pos': (-390.4, -799.1, 139.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 165)}
        elif road_id == "8518":
            if reverse:
                return {'pos': (-913.2, -829.6, 118.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 60)}
            # starts on right turn, rock walls surrounding road, lanelines
            return {'pos': (-390.5, -896.6, 138.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 20)}
        elif road_id == "8417":
            if reverse:
                # road surrounding suburb, big intersection
                # return {"pos": (-282.82, -875.592, 134.9), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 155)}
                # past intersection
                # return {"pos": (-306.01910400390625, -862.2595825195312, 135.1), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 155)}
                # return {"pos": (-320.4444885253906, -853.4628295898438, 135.6), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 155)}
                return {"pos": (-348.3262939453125, -831.0289916992188, 137.0), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 165)}
            else:
                # road surrounding suburb, starts on left side of road
                return {'pos': (-402.7, -780.2, 141.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        # elif road_id == "8703":
        #     if reverse:
        #         # return {'pos': (-312.4, -856.8, 135.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 155)}
        #     # 055.646|E|libbeamng.lua.V.updateGFX|Object position: vec3(-327.926,-846.741,136.099)
        #     # 055.646|E|libbeamng.lua.V.updateGFX|Object rotation: quat(0.012735072523355,-0.0081959860399365,0.87504893541336,0.48379769921303)
        #         return {'pos': (-327.926,-846.741,136.099), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 155)}
        #     return {'pos': (-307.8, -784.9, 137.6), 'rot': None,
        #             'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 80)}
        elif road_id == "12641":
            # if reverse:
            #     return {'pos': (-964.2, 882.8, 75.1), 'rot': None,
            #             'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            return {'pos': (-366.1753845214844, 632.2236938476562, 75.1), 'rot': None,
                    'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 180)}
        elif road_id == "13091":
            if reverse:
                return {'pos': (-903.6078491210938, -586.33154296875, 106.6), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 80)}
            return {'pos': (-331.0728759765625, -697.2451782226562, 133.0), 'rot': None,
                    'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        # elif road_id == "11602":
        #     return {'pos': (-366.4, -858.8, 136.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "12146":
            if reverse:
                return {'pos': (995.7, -855.0, 167.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
            return {'pos': (-391.0, -798.8, 139.7), 'rot': None,
                    'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
        elif road_id == "13228":
            return {'pos': (-591.5175170898438, -453.1298828125, 114.0), 'rot': None,
                    'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "13155":  # middle laneline on highway #12492 11930, 10368 is an edge
            return {'pos': (-390.7796936035156, -36.612098693847656, 109.9), 'rot': None,
                    'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "10784":  # track # 13228  suburb edge, 12939 10371 12098 edge
            if reverse:
                # (800.905,350.394,156.297)
                return {'pos': (800.905,350.394,156.297), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -135)}
            # return {'pos': (57.05, -150.53, 125.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -115)}
            # return {'pos': (86.1454,-118.969,127.519), 'rot': None, 'rot_quat': (-0.03234875574708,0.022467797622085,-0.8322811126709,0.55295306444168)}
            return {'pos': (144.962, -96.1268, 128.935), 'rot': None, 'rot_quat': (-0.032797202467918, 0.024726673960686, -0.80050182342529, 0.59792125225067)}
        elif road_id == "10673":
            return {'pos': (-21.7, -826.2, 133.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "12930":  # 13492 dirt road
            # return {'pos': (-347.2, -824.7, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            return {'pos': (-353.731,-830.905,137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
        elif road_id == "10988": # track
            # return {'pos': (622.2, -251.1, 147.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
            # return {'pos': (660.388,-247.67,147.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -160)}
            if seg == 0: # straight portion
                return {'pos': (687.5048828125, -185.7435302734375, 146.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            elif seg == 1: # approaching winding portion
                #  crashes around [846.0238647460938, 127.84288787841797, 150.64915466308594]
                # return {'pos': (768.1991577148438, -108.50184631347656, 146.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                return {'pos': (781.2423095703125, -95.72360229492188, 147.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                return {'pos': (790.599,-86.7973,147.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)} # slightly better?
            elif seg == 2:
                return {'pos': (854.4083862304688, 136.79324340820312, 152.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            else:
                return {'pos': (599.341, -252.333, 147.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
        elif road_id == "13306":
            return {'pos': (-310, -790.044921875, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -30)}
        elif road_id == "13341":
            return {'pos': (-393.4, -34.0, 109.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 90)}
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
            return {'pos': (174.92, -289.7, 120.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            # return {'pos': (-180.4, -253.0, 120.7), 'rot': None, 'rot_quat': (-0.008, 0.004, 0.779, 0.63)}
            return {'pos': (-58.2675, -255.216, 120.175), 'rot': None, 'rot_quat': (-0.021, -0.009, 0.740, 0.672)}
        elif road_id == "8293": # immediately leaves track
            return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.7115), 120)}
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
            if seg == 0: # start of track, right turn; 183m; cutoff at (412.079,-191.549,38.2418)
                return {'pos': (289.327,-281.458, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 1: # straight road
                return {'pos': (330.3320007324219, -217.5743408203125, 45.7054443359375), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 2: # left turn
                # return {'pos': (439.0, -178.4, 35.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                # return {'pos': (448.1, -174.6, 34.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                return {'pos': (496.2, -150.6, 35.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
            elif seg == 3:
                return {'pos': (538.2, -124.3, 40.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -110)}
            elif seg == 4: # straight; cutoff at vec3(596.333,18.7362,45.6584)
                return {'pos': (561.7396240234375, -76.91995239257812, 44.7), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 5: # left turn; cutoff at (547.15234375, 115.24089050292969, 35.97171401977539)
                return {'pos': (598.3154907226562, 40.60638427734375, 43.9), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -147)}
            elif seg == 6:
                return {'pos': (547.15234375, 115.24089050292969, 36.3), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
            elif seg == 7:
                return {'pos': (449.7561340332031, 114.96491241455078, 25.801856994628906), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
            elif seg == 8: # mostly straight, good behavior; cutoff at  vec3(305.115,304.196,38.4392)
                return {'pos': (405.81732177734375, 121.84907531738281, 25.04170036315918), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
            elif seg == 9:
                return {'pos': (291.171875, 321.78662109375, 38.6), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
            elif seg == 10:
                return {'pos': (216.40045166015625, 367.1772155761719, 35.99), 'rot': None,
                        'rot_quat': (-0.037829957902431,0.0035844487138093,0.87171512842178,0.48853760957718)}
            else:
                return {'pos': (290.558, -277.280, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
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

def setup_sensors(vehicle, img_dims, fov=51):
    # Set up sensors
    fov = fov # 60 works for full lap #63 breaks on hairpin turn
    resolution = img_dims #(240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    pos = (-0.5, 0.38, 1.3)
    direction = (0, 1.0, 0)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)

    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()

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

def diff_damage(damage, damage_prev):
    if damage is None or damage_prev is None:
        return 0
    else:
        return damage['damage'] - damage_prev['damage']

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
    avg = sum(dists) / len(dists)
    return {"stddev":stddev, "mean":avg}

def plot_racetrack_roads(roads, bng, default_scenario, road_id, reverse=False):
    print("Plotting scenario roads...")
    sp = spawn_point(default_scenario, road_id, reverse=reverse)
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


def road_analysis(bng, road_id):
    global centerline, roadleft, roadright
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
    middle = road_analysis(bng, road_id)
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

def create_ai_line_from_road_with_interpolation(spawn, bng, road_id):
    global centerline, remaining_centerline, centerline_interpolated
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    print("Performing road analysis...")
    actual_middle, adjusted_middle = road_analysis(bng, road_id)
    # plt.plot([i[0] for i in actual_middle], [i[1] for i in actual_middle])
    # plt.show()
    print(f"{actual_middle[0]=}, {actual_middle[-1]=}")
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
    return dist > 5.0, dist

def setup_beamng(default_scenario, road_id, reverse=False, seg=1, img_dims=(240,135), fov=51, vehicle_model='etk800', default_color="green", steps_per_sec=15,
                 beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE4', port=64956):
    global base_filename

    random.seed(1703)
    setup_logging()
    print(road_id)
    beamng = BeamNGpy('localhost', port, home='C:/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user=beamnginstance)
    # beamng = BeamNGpy('localhost', 64256, home='C:/Users/Meriel/Documents/BeamNG.tech.v0.21.3.0', user='C:/Users/Meriel/Documents/BeamNG.tech')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, img_dims, fov=fov)
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)
    print(default_scenario, road_id, seg, spawn)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat']) #, partConfig=parts_config)
    print(road_id)
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    bng.set_deterministic()
    bng.set_steps_per_second(steps_per_sec)
    bng.load_scenario(scenario)
    bng.start_scenario()
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn, bng, road_id)
    bng.pause()
    assert vehicle.skt
    # bng.resume()
    return vehicle, bng, scenario

def run_scenario(vehicle, bng, scenario, model, default_scenario, road_id, reverse=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), seg=None):
    global base_filename
    global integral, prev_error, setpoint
    if default_scenario == "hirochi_raceway" and road_id == "9039" and seg == 0:
        cutoff_point = [368.466, -206.154, 43.8237]
    elif default_scenario == "automation_test_track" and road_id == "8185":
        # cutoff_point = [-44.8323, -256.138, 120.186] # 200m
        cutoff_point = [56.01722717285156, -272.89007568359375, 120.56710052490234] # 100m
    elif default_scenario == "west_coast_usa" and road_id == "12930":
        cutoff_point = [-296.55999755859375, -730.234130859375, 136.71339416503906]
    elif default_scenario == "west_coast_usa" and road_id == "10988" and seg == 1:
        # cutoff_point = [843.5116577148438, 5.83634090423584, 147.02598571777344] # a little too early
        # cutoff_point = [843.5073852539062, 6.2438249588012695, 147.01889038085938] # middle
        cutoff_point = [843.6112670898438, 6.58771276473999, 147.01829528808594] # late
    else:
        cutoff_point = [601.547, 53.4482, 43.29]
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
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)

    wheelspeed = 0.0; kph = 0.; throttle = 0.0; prev_error = setpoint; damage_prev = None; runtime = 0.0
    kphs = []; traj = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = 0.0
    final_img = None
    total_loops = 0; total_imgs = 0; total_predictions = 0
    start_time = sensors['timer']['time']
    outside_track = False
    distance_from_center = 0
    writedir = f"{default_scenario}-{road_id}-lap-test"
    if not os.path.isdir(writedir):
        os.mkdir(writedir)
    # with open(f"{writedir}/data.txt", "w") as f:
    # f.write(f"IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT\n")
    while kph < 35:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        vehicle.control(throttle=1., steering=0., brake=0.0)
        bng.step(1, wait=True)
    while damage <= 1:
        # collect images
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        image_seg = sensors['front_cam']['annotation'].convert('RGB')
        # image = image.resize((240,135))
        # image = cv2.resize(np.array(image), (135,240))
        # image = fisheye_inv(image)
        cv2.imshow('car view', np.array(image)[:, :, ::-1])
        cv2.waitKey(1)
        total_imgs += 1
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - start_time) - runtime
        processed_img = model.process_image(image).to(device)
        prediction = model(processed_img)
        steering = float(prediction.item())
        runtime = sensors['timer']['time'] - start_time

        total_predictions += 1
        # position = str(vehicle.state['pos']).replace(",", " ")
        # orientation = str(vehicle.state['dir']).replace(",", " ")
        # image.save(f"{writedir}/sample-{total_imgs:05d}.jpg", "JPEG")
        # image_seg.save(f"{writedir}/sample-segmented-{total_imgs:05d}.jpg", "JPEG")
        # f.write(f"sample-{total_imgs:05d}.jpg,{prediction.item()},{position},{orientation},{kph},{sensors['electrics']['steering']}\n")
        if abs(steering) > 0.15:
            setpoint = 30
        else:
            setpoint = 40
        throttle = throttle_PID(kph, dt)
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)
        steering_inputs.append(steering)
        throttle_inputs.append(throttle)
        timestamps.append(runtime)

        damage = sensors['damage']["damage"]
        vehicle.update_vehicle()
        traj.append(vehicle.state['pos'])

        kphs.append(ms_to_kph(wheelspeed))
        total_loops += 1
        final_img = image
        dists = dist_from_line(centerline, vehicle.state['pos'])
        m = np.where(dists==min(dists))[0][0]
        # print(f"Try next spawn: {centerline[m + 5]}")
        # print(f"{vehicle.state['pos']=}\ndistance travelled: {get_distance_traveled(traj):3f}")
        if damage > 1.0:
            print(f"Damage={damage:.3f}, exiting...")
            print(f"Try next spawn: {centerline[m+5]}")
            break
        bng.step(1, wait=True)

        # dist_to_cutoff = distance2D(vehicle.state["pos"], cutoff_point)
        # print(f"{dist_to_cutoff=:3f}")
        if distance2D(vehicle.state["pos"], cutoff_point) < 12:
            print("Reached cutoff point, exiting...")
            break

        outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)
        if outside_track:
            print("Left track, exiting...")
            break

    cv2.destroyAllWindows()

    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj, 'pitch': round(pitch,3),
               'roll':round(roll,3), "z":round(z,3), 'final_img':final_img, 'deviation':deviation
               }
    return results

def distance2D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

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
    with open('posefiles/hirochi_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            rot_quat = turn_X_degrees(rot_quat, degrees=-115)
            ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            scenario.add_object(ramp)

def fisheye_wand(image, filename=None):
    with WandImage.from_array(image) as img:
        img.virtual_pixel = 'transparent'
        img.distort('barrel', (0.1, 0.0, -0.05, 1.0))
        img.alpha_channel = False
        img = np.array(img, dtype='uint8')
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def fisheye_inv(image):
    with WandImage.from_array(image) as img:
        img.virtual_pixel = 'transparent'
        img.distort('barrel_inverse', (0.0, 0.0, -0.5, 1.5))
        img = np.array(img, dtype='uint8')
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def get_topo(topo_id):
    if "windy" in topo_id:
        default_scenario = "west_coast_usa"; road_id = "10988"; seg = 1
    elif "straight" in topo_id:
        default_scenario = "automation_test_track"; road_id = "8185"; seg = None
    elif "Rturn" in topo_id:
        default_scenario="hirochi_raceway"; road_id="9039"; seg=0
    elif "Lturn" in topo_id:
        default_scenario = "west_coast_usa"; road_id = "12930"; seg = None
    return default_scenario, road_id, seg

def get_transf(transf_id):
    if transf_id is None:
        img_dims = (240,135); fov = 51; transf = "None"
    elif "fisheye" in transf_id:
        img_dims = (240,135); fov=75; transf = "fisheye"
    elif "resdec" in transf_id:
        img_dims = (96, 54); fov = 51; transf = "resdec"
    elif "resinc" in transf_id:
        img_dims = (480,270); fov = 51; transf = "resinc"
    elif "depth" in transf_id:
        img_dims = (240, 135); fov = 51; transf = "depth"
    return img_dims, fov, transf

def main():
    global base_filename
    model_name = "../models/weights/dave2-weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt" # orig model
    model_name = "../models/retrained-lighterblur-noflip-fixednoise/model-fixnoise-DAVE2v3-135x240-lr1e4-100epoch-64batch-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-best.pt"
    # model_name = "C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/models/weights/fixed-base-model/model-DAVE2v3-135x240-lr1e4-100epoch-64batch-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()

    topo_id= "Rturn"
    transf_id = None
    default_scenario, road_id, seg = get_topo(topo_id)
    img_dims, fov, transf = get_transf(transf_id)

    vehicle, bng, scenario = setup_beamng(default_scenario=default_scenario, road_id=road_id, seg=seg, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE3', port=64556)
    distances, deviations = [], []
    runs = 50
    for i in range(runs):
        results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, seg=seg)
        results['distance'] = get_distance_traveled(results['traj'])
        # plot_trajectory(results['traj'], f"{default_scenario}-{model._get_name()}-{road_id}-runtime{results['runtime']:.2f}-dist{results['distance']:.2f}")
        print(f"\nBASE MODEL USING IMG DIMS {img_dims} RUN {i}:"
              f"\n\tdistance={results['distance']}"
              f"\n\tavg dist from center={results['deviation']['mean']}")
        distances.append(results['distance'])
        deviations.append(results['deviation']['mean'])
    print(f"OUT OF {runs} RUNS:\n\tAverage distance: {(sum(distances)/len(distances)):1f}"
          f"\n\tAverage deviation: {(sum(deviations) / len(deviations)):3f}"
          f"\n\t{distances=}"
          f"\n\t{deviations:}")
    bng.close()


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()