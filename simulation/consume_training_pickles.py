import numpy as np
from matplotlib import pyplot as plt
import logging, random, string, time, copy, os

import statistics, math
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate

import torch
import cv2
from skimage import util
from PIL import Image
from sklearn.metrics import mean_squared_error
import kornia
from torchvision.utils import save_image
import pandas as pd
import pickle



def plot_deviation(model, topo, trajectories, centerline, roadleft, roadright, deflation_pattern, save_path=".", start_viz=False):
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
    for i, t in enumerate(trajectories):
        x, y = [], []
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y)  # , label="Run {}".format(i))
    if topo == "winding":
        plt.xlim([700, 900])
        plt.ylim([-200, 125])
    elif topo == "Rturn":
        plt.xlim([200, 400])
        plt.ylim([-300, -100])
    elif topo == "straight":
        plt.xlim([25, 200])
        plt.ylim([-300, -265])
    plt.title(f'Trajectories with {model}\n{deflation_pattern.split("/")[-1]}', fontdict={'fontsize': 8})
    plt.legend(fontsize=8)
    plt.draw()
    plt.savefig(save_path)
    plt.clf()

def plot_errors(errors, filename="images/errors.png"):
    plt.title("Errors")
    for ei, e in enumerate(errors):
        plt.plot(range(len(e)), e, label=f"Error {ei}")
    plt.savefig(filename)
    plt.show()
    plt.pause(0.1)

    plt.title("Error Distributions per Run")
    avgs = []
    for ei, e in enumerate(errors):
        plt.scatter(np.ones((len(e)))*ei, e, s=5) #, label=f"Error {ei}")
        avgs.append(float(sum(e)) / len(e))
    plt.plot(range(len(avgs)), avgs)
    plt.savefig(filename.replace(".png", "-distribution.png"))
    plt.show()
    plt.pause(0.1)

def write_results(training_file, results, all_trajs, unperturbed_traj,
                                  modelname, technique, direction, lossname, bbsize, its, nl):
    results["all_trajs"] = all_trajs
    results["unperturbed_traj"] = unperturbed_traj
    results["modelname"] = modelname
    results["technique"] = technique
    results["direction"] = direction
    results["lossname"] = lossname
    results["bbsize"] = bbsize
    results["its"] = its
    results["nl"] = nl
    with open(training_file, "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)



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


def dist_from_line(centerline, point):
    a = [[x[0], x[1]] for x in centerline[:-1]]
    b = [[x[0], x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist


def calc_deviation_from_center(centerline, traj):
    dists = []
    for point in traj:
        dist = dist_from_line(centerline, point)
        dists.append(min(dist))
    avg_dist = sum(dists) / len(dists)
    stddev = statistics.stdev(dists)
    return stddev, dists, avg_dist


def intake_lap_file(filename="DAVE2v1-lap-trajectory.txt"):
    # global expected_trajectory
    expected_trajectory = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            # print(line)
            line = literal_eval(line)
            expected_trajectory.append(line)
    return expected_trajectory


def road_analysis(bng):
    # roads = bng.get_roads()
    # get relevant road
    edges = bng.get_road_edges('7983')
    middle = [edge['middle'] for edge in edges]
    left = [edge['left'] for edge in edges]
    right = [edge['right'] for edge in edges]
    return middle, left, right


def plot_input(timestamps, input, input_type, run_number=0):
    plt.plot(timestamps, input)
    plt.xlabel('Timestamps')
    plt.ylabel('{} input'.format(input_type))
    # Set a title of the current axes.
    plt.title("{} over time".format(input_type))
    plt.savefig("images/Run-{}-{}.png".format(run_number, input_type))
    plt.show()
    plt.pause(0.1)


def plot_trajectory(traj, title="Trajectory", run_number=0):
    global centerline
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x, y, 'bo', label="AI behavior")
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r+', label="AI line script")
    plt.title(title)
    plt.legend()
    plt.savefig("images/Run-{}-traj.png".format(run_number))
    plt.show()
    plt.pause(0.1)

def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(
            math.pow(traj[i][0] - traj[i + 1][0], 2) + math.pow(traj[i][1] - traj[i + 1][1], 2) + math.pow(
                traj[i][2] - traj[i + 1][2], 2))
    return dist

def unpickle_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def main():
    global base_filename, default_scenario, default_spawnpoint, setpoint, integral
    global prev_error, centerline, centerline_interpolated, unperturbed_traj
    start = time.time()
    evalResultsDir = "F:/RRL-results/RLtrain-MlpPolicy-0.558-onpolicyall-winding-fisheye-max200-0.05eval-2_24-20_22-JALLIT"
    trainResultsDir = "F:/RRL-results/RLtrain-onpolicyeval-winding-fisheye-max200-0.05eval-eval-2_24-20_22-JALLIT"
    fileExt = r".pickle"
    dirs = [_ for _ in os.listdir(trainResultsDir) if _.endswith(fileExt)]
    from numpy import load

    data = load(f'{trainResultsDir}/evaluations.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])
    # exit(0)
    dists_from_center = []
    dists_travelled = []
    for d in dirs:
        results = unpickle_results("/".join([trainResultsDir, d]))
        print(d)
        print(results.keys())
        dist = get_distance_traveled(results['trajectory'])
        print(f"\ttotal distance travelled: {dist:.1f}"
              f"\n\ttotal episode reward: {sum(results['rewards']):.1f}"
              f"\n\tavg dist from ctrline: {sum(results['dist_from_centerline']) / len(results['dist_from_centerline']):.3f}"
              f"\n\tpercent frames adjusted: {results['frames_adjusted_count'] / results['total_steps']:.3f}"
              f"\n\ttotal steps: {results['total_steps']}"
              f"\n\trew max/min/avg/stdev: {max(results['rewards']):.3f} / {min(results['rewards']):.3f} / {sum(results['rewards']) / len(results['rewards']):.3f} / {np.std(results['rewards']):.3f}"
              f"\n")
        dists_travelled.append(dist)
        dists_from_center.append(sum(results['dist_from_centerline']) / len(results['dist_from_centerline']))
    print(f"Avg dist travelled: {sum(dists_travelled) / len(dists_travelled)}"
          f"\nAvg dist from centerline: {sum(dists_from_center) / len(dists_from_center)}")
    print(f"Finished in {time.time() - start} seconds")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()
