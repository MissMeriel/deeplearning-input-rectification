import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array(gray, dtype=np.uint8)


def spawn_point(default_scenario, road_id, seg, reverse):
    if default_scenario == 'cliff':
        #return {'pos':(-124.806, 142.554, 465.489), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
    elif default_scenario == 'west_coast_usa':
        if road_id == 'midhighway': # mid highway scenario (past shadowy parts of road)
            return {'pos': (-145.775, 211.862, 115.55), 'rot': None, 'rot_quat': (0.0032586499582976, -0.0018308814615011, 0.92652350664139, -0.37621837854385)}
        # actually past shadowy parts of road?
        #return {'pos': (95.1332, 409.858, 117.435), 'rot': None, 'rot_quat': (0.0077012465335429, 0.0036200874019414, 0.90092438459396, -0.43389266729355)}
        # surface road (crashes early af)
        elif road_id == "13242":
            return {'pos': (-733.7, -923.8, 163.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, 0.805, 0.592), -20)}
        elif road_id == "8650":
            return {'pos': (-365.24, -854.45, 136.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 90)}
        elif road_id == "12667":
            return {'pos': (-892.4, -793.4, 114.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 70)}
        elif road_id == "8432":
            if reverse:
                return {'pos': (-871.9, -803.2, 115.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 165 - 180)}
            return {'pos': (-390.4,-799.1,139.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 165)}
        elif road_id == "8518":
            if reverse:
                return {'pos': (-913.2, -829.6, 118.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 60)}
            return {'pos': (-390.5, -896.6, 138.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 20)}
        elif road_id == "8417":
            return {'pos': (-402.7,-780.2,141.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "8703":
            if reverse:
                return {'pos': (-312.4, -856.8, 135.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            return {'pos': (-307.8,-784.9,137.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 80)}
        elif road_id == "12641":
            if reverse:
                return {'pos': (-964.2, 882.8, 75.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            return {'pos': (-366.1753845214844, 632.2236938476562, 75.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 180)}
        elif road_id == "13091":
            if reverse:
                return {'pos': (-903.6078491210938, -586.33154296875, 106.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 80)}
            return {'pos': (-331.0728759765625,-697.2451782226562,133.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        # elif road_id == "11602":
        #     return {'pos': (-366.4, -858.8, 136.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "12146":
            if reverse:
                return {'pos': (995.7, -855.0, 167.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
            return {'pos': (-391.0,-798.8, 139.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
        elif road_id == "13228":
            return {'pos': (-591.5175170898438,-453.1298828125,114.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "13155": #middle laneline on highway #12492 11930, 10368 is an edge
            return {'pos': (-390.7796936035156, -36.612098693847656, 109.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "10784": # 13228  suburb edge, 12939 10371 12098 edge
            return {'pos': (57.04786682128906, -150.53302001953125, 125.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -115)}
        elif road_id == "10673":
            return {'pos': (-21.712169647216797, -826.2122802734375, 133.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "12930":  # 13492 dirt road
            # return {'pos': (-347.2, -824.7, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            return {'pos': (-353.731, -830.905, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
        elif road_id == "10988":  # track
            # return {'pos': (622.2, -251.1, 147.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
            # return {'pos': (660.388,-247.67,147.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -160)}
            if seg == 0:  # straight portion
                return {'pos': (687.5048828125, -185.7435302734375, 146.9), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            elif seg == 1:  # approaching winding portion
                #  crashes around [846.0238647460938, 127.84288787841797, 150.64915466308594]
                # return {'pos': (768.1991577148438, -108.50184631347656, 146.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                return {'pos': (781.2423095703125, -95.72360229492188, 147.4), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                return {'pos': (790.599, -86.7973, 147.3), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}  # slightly better?
            elif seg == 2:
                return {'pos': (854.4083862304688, 136.79324340820312, 152.7), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            else:
                return {'pos': (599.341, -252.333, 147.6), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
            # other end of the gas station-airfield yellow road
            # return {'pos': (622.163330078125,-251.1154022216797,146.99), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "13306":
            return {'pos': (-310,-790.044921875,137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -30)}
        elif road_id == "13341":
            return {'pos': (-393.4385986328125,-34.0107536315918,109.64727020263672), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 90)}
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
            return {'pos': (57.229, 360.560, 128.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        elif road_id == "8293":
            return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
        elif road_id == '8185': # highway (open, farm-like)
            return {'pos': (174.92, -289.7, 120.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            # return {'pos': (174.9,-289.7,120.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.704635, 0.70957), 180)}
            # return {'pos': (36.742,-269.105,120.461), 'rot': None, 'rot_quat': (-0.0070,0.0082,0.7754,0.6314)}
            # return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.704635, 0.70957), 180)}
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
        # if road_id == "9039" or road_id == "9040": # good candidate for input rect.
        #     if seg == 0:
        #         return {'pos': (289.327, -281.458, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
        #     else:
        #         # return {'pos': (292.405,-271.64,46.75), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}
        #         return {'pos': (290.558, -277.280, 46.0), 'rot': None,
        #                 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}

        if road_id == "9039":  # good candidate for input rect.
            if seg == 0:  # start of track, right turn; 183m; cutoff at (412.079,-191.549,38.2418)
                return {'pos': (289.327, -281.458, 46.0), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 1:  # straight road
                return {'pos': (330.3320007324219, -217.5743408203125, 45.7054443359375), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 2:  # left turn
                # return {'pos': (439.0, -178.4, 35.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                # return {'pos': (448.1, -174.6, 34.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                return {'pos': (496.2, -150.6, 35.6), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
            elif seg == 3:
                return {'pos': (538.2, -124.3, 40.5), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -110)}
            elif seg == 4:  # straight; cutoff at vec3(596.333,18.7362,45.6584)
                return {'pos': (561.7396240234375, -76.91995239257812, 44.7), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 5:  # left turn; cutoff at (547.15234375, 115.24089050292969, 35.97171401977539)
                return {'pos': (598.3154907226562, 40.60638427734375, 43.9), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -147)}
            elif seg == 6:
                return {'pos': (547.15234375, 115.24089050292969, 36.3), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
            elif seg == 7:
                return {'pos': (449.7561340332031, 114.96491241455078, 25.801856994628906), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
            elif seg == 8:  # mostly straight, good behavior; cutoff at  vec3(305.115,304.196,38.4392)
                return {'pos': (405.81732177734375, 121.84907531738281, 25.04170036315918), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
            elif seg == 9:
                return {'pos': (291.171875, 321.78662109375, 38.6), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
            elif seg == 10:
                return {'pos': (216.40045166015625, 367.1772155761719, 35.99), 'rot': None,
                        'rot_quat': (-0.037829957902431, 0.0035844487138093, 0.87171512842178, 0.48853760957718)}
            else:
                return {'pos': (290.558, -277.280, 46.0), 'rot': None,
                        'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
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


def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6

'''return distance between two N-dimensional points'''
def distance(a, b):
    sqr = sum([math.pow(ai-bi, 2) for ai, bi in zip(a,b)])
    return math.sqrt(sqr)

def turn_X_degrees(rot_quat, degrees=90):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + degrees
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def dist_from_line(centerline, point):
    a = [x[0:2] for x in centerline[:-1]]
    b = [x[0:2] for x in centerline[1:]]
    return lineseg_dists(point[0:2], np.array(a), np.array(b))

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

def get_cutoff_point(default_scenario, road_id, seg):
    if default_scenario == "hirochi_raceway" and road_id == "9039" and seg == 0:
        cutoff_point = [368.466, -206.154, 43.8237]
    elif default_scenario == "automation_test_track" and road_id == "8185":
        cutoff_point = [56.017, -272.890, 120.567]  # 100m
    elif default_scenario == "west_coast_usa" and road_id == "12930":
        cutoff_point = [-296.560, -730.234, 136.713]
    elif default_scenario == "west_coast_usa" and road_id == "10988" and seg == 1:
        # cutoff_point = [843.507, 6.244, 147.019] # middle
        cutoff_point = [843.611, 6.588, 147.018]  # late
    else:
        cutoff_point = [601.547, 53.4482, 43.29]
    return cutoff_point

def setup_sensors(transf, obs_shape, vehicle):
    camera_pos = (-0.5, 0.38, 1.3)
    camera_dir = (0, 1.0, 0)
    if transf == "fisheye":
        fov = 75
    else:
        fov = 51 # 60 works for full lap #63 breaks on hairpin turn

    width = int(obs_shape[2])
    height = int(obs_shape[1])
    resolution = (width, height)
    if transf == "depth":
        front_camera = Camera(camera_pos, camera_dir, fov, resolution,
                              colour=True, depth=True, annotation=False, near_far=(1, 50))
        far_camera = Camera(camera_pos, camera_dir, fov, (obs_shape[2], obs_shape[1]),
                              colour=True, depth=True, annotation=False, near_far=(1, 100))
    else:
        front_camera = Camera(camera_pos, camera_dir, fov, resolution,
                              colour=True, depth=True, annotation=False)
    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()
    vehicle.attach_sensor('front_cam', front_camera)
    if transf == "depth":
        vehicle.attach_sensor('far_camera', far_camera)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle

def create_ai_line_from_road_with_interpolation(bng, road_id):
    points, point_colors, spheres, sphere_colors = [], [], [], []
    centerline_interpolated = []
    centerline, roadleft, roadright = road_analysis(bng, road_id)
    # interpolate centerline at 1m distance
    for i, p in enumerate(centerline[:-1]):
        if distance(p, centerline[i+1]) > 1:
            y_interp = interpolate.interp1d([p[0], centerline[i+1][0]], [p[1], centerline[i+1][1]])
            num = int(distance(p, centerline[i+1]))
            xs = np.linspace(p[0], centerline[i+1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x,y in zip(xs, ys):
                centerline_interpolated.append([x, y])
        else:
            centerline_interpolated.append([p[0],p[1]])
    # set up debug line
    for p in centerline[:-1]:
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
    bng.add_debug_line(points, point_colors, spheres=spheres, sphere_colors=sphere_colors, cling=True, offset=0.1)
    return bng, centerline_interpolated, centerline, roadleft, roadright

def road_analysis(bng, road_id, reverse=False):
    print("Performing road analysis...")
    # get_nearby_racetrack_roads()
    # plot_racetrack_roads(bng, default_scenario)
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    if reverse:
        edges.reverse()
        print(f"new spawn={edges[0]['middle']}")
    else:
        print(f"reversed spawn={edges[-1]['middle']}")
    centerline = [edge['middle'] for edge in edges]
    if road_id == "8185":
        edges = bng.get_road_edges("8096")
        roadleft = [edge['middle'] for edge in edges]
        edges = bng.get_road_edges("7878")  # 7820, 7878, 7805
        roadright = [edge['middle'] for edge in edges]
    else:
        roadleft = [edge['left'] for edge in edges]
        roadright = [edge['right'] for edge in edges]
    return centerline, roadleft, roadright


def plot_racetrack_roads(bng, default_scenario):
    roads = bng.get_roads()
    sp = spawn_point()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
    for road in roads:
        road_edges = bng.get_road_edges(road)
        if len(road_edges) < 100:
            continue
        if distance(road_edges['left'][0], road_edges['left'][0]) < 5:
            continue
        x_temp, y_temp = [], []
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1])]
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
    plt.title(default_scenario)
    plt.legend(ncol=10)
    plt.show()
    plt.pause(0.001)

def get_nearby_racetrack_roads(bng, default_scenario, point_of_in):
    print(f"Plotting nearby roads to point={point_of_in}")
    roads = bng.get_roads()
    print("retrieved roads")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp, y_temp = [], []
        if len(road_edges) < 100:
            continue
        xy_def = [edge['middle'][:2] for edge in road_edges]
        # dists = [distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1:5])]
        # road_len = sum(dists)
        dists = [distance(i, point_of_in) for i in xy_def]
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
    plt.title(f"{default_scenario} poi={point_of_in}")
    plt.legend(ncol=10)
    plt.show()
    plt.pause(0.001)

def turn_X_degrees(rot_quat, degrees=90):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + degrees
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def plot_deviation(model, deflation_pattern, centerline, roadleft, roadright, trajectories, topo, save_path=".", start_viz=False):
    plt.figure(20, dpi=180)
    plt.clf()
    x, y = [], []
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
    for i,t in enumerate(trajectories):
        x, y = [], []
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y) #, label="Run {}".format(i))
    if topo == "winding":
        plt.xlim([700, 900])
        plt.ylim([-200, 125])
    elif topo == "Rturn":
        plt.xlim([200, 400])
        plt.ylim([-300, -100])
    elif topo == "straight":
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

def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(
            math.pow(traj[i][0] - traj[i + 1][0], 2) + math.pow(traj[i][1] - traj[i + 1][1], 2) + math.pow(
                traj[i][2] - traj[i + 1][2], 2))
    return dist

def plot_durations(rewards, save=False, title="temp", savetitle="rewards_training_performance"):
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