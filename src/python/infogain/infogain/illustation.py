

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb

from os import listdir
from os.path import isfile, join

mpl.use('pdf')

# width as measured in inkscape
width = 8  # 3.487
height = width / 1.5

dt = 0.1
T = 2
V_ego = 10  # m/s
A_max = 4.572  # m/s^2 == 15 ft/s^2
horizon_length = 10  # planning ten steps ahead
V_o1 = 3  # m/s
V_o2 = 8  # m/s


def plot_actor(ax, pos=[0, 0], scale=1.0, orientation=0.0):
    rot = np.array(
        [
            [np.cos(orientation), -np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)],
        ]
    )

    pts = np.array([
        [5, 0],
        [-5, 4],
        [-2, 0],
        [-5, -4],
        [5, 0],
    ]) * scale

    pts = ((rot @ pts.T)).T + pos

    poly = plt.Polygon(pts, closed=True, edgecolor='darkblue', facecolor='cyan', alpha=0.3)
    ax.add_patch(poly)


def main():
    sb.set()
    sb.set_theme(style="whitegrid")

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=12)

    fig, ax = plt.subplots()

    d_s = V_ego**2 / (2 * A_max)
    t_s = V_ego / A_max

    # define a straight line trajectory
    ego_traj = np.array([[dt * i * V_ego, 0] for i in range(int(T*t_s/dt))])

    p_risk_upper = [[0, 0]]
    p_risk_lower = [[0, 0]]
    for step in range(1, int(T * t_s / dt)):
        t = step * dt

        x = t * V_ego
        d_o = t * V_o2
        p_risk_upper.append([x, d_o])
        p_risk_lower.append([x, -d_o])

    p_risk_upper = np.array(p_risk_upper)
    p_risk_lower = np.array(p_risk_lower)

    # create two patches for occlusion
    i_start = int((0.95 * d_s) / (V_ego * dt))
    i_end = int(i_start + 0.5 / dt)
    poly = plt.Polygon(np.array([p_risk_upper[i_start, :], p_risk_lower[i_start, :], p_risk_lower[i_end, :],
                       p_risk_upper[i_end, :]]), closed=True, edgecolor='firebrick', facecolor='salmon', alpha=0.3)
    ax.add_patch(poly)
    i_start = int((2.0 * d_s) / (V_ego * dt))
    i_end = int(i_start + 0.5 / dt)
    poly = plt.Polygon(np.array([p_risk_upper[i_start, :], p_risk_lower[i_start, :], p_risk_lower[i_end, :], p_risk_upper[i_end, :]]),
                       closed=True, edgecolor='lightblue', facecolor='lightslategrey', alpha=0.3)
    ax.add_patch(poly)

    sb.lineplot(x=ego_traj[:, 0], y=ego_traj[:, 1], color='black', alpha=0.4)
    sb.lineplot(x=p_risk_lower[:, 0], y=p_risk_lower[:, 1], color='slategrey')
    sb.lineplot(x=p_risk_upper[:, 0], y=p_risk_upper[:, 1], color='slategrey')

    p_risk_upper = [[0, 0]]
    p_risk_lower = [[0, 0]]
    for step in range(1, int(T * t_s / dt)):
        t = step * dt

        x = t * V_ego
        d_o = t * V_o1
        p_risk_upper.append([x, d_o])
        p_risk_lower.append([x, -d_o])

    p_risk_upper = np.array(p_risk_upper)
    p_risk_lower = np.array(p_risk_lower)

    # create two patches for occlusion
    i_start = int((0.95 * d_s) / (V_ego * dt))
    i_end = int(i_start + 0.5 / dt)
    poly = plt.Polygon(np.array([p_risk_upper[i_start, :], p_risk_lower[i_start, :], p_risk_lower[i_end, :],
                       p_risk_upper[i_end, :]]), closed=True, edgecolor='firebrick', facecolor='gold', alpha=0.3)
    ax.add_patch(poly)
    i_start = int((2.0 * d_s) / (V_ego * dt))
    i_end = int(i_start + 0.5 / dt)
    poly = plt.Polygon(np.array([p_risk_upper[i_start, :], p_risk_lower[i_start, :], p_risk_lower[i_end, :],
                       p_risk_upper[i_end, :]]), closed=True, edgecolor='lightblue', facecolor='dodgerblue', alpha=0.3)
    ax.add_patch(poly)

    sb.lineplot(x=p_risk_lower[:, 0], y=p_risk_lower[:, 1], color='darkblue')
    sb.lineplot(x=p_risk_upper[:, 0], y=p_risk_upper[:, 1], color='darkblue')

    # Add the safe spaces along the trajectory
    for step in range(1, horizon_length):
        t = step * dt

        x = t * V_ego
        c_pos = plt.Circle((x, 0), 0.3, edgecolor='black', facecolor='darkgrey')
        ax.add_patch(c_pos)

        c_safe = plt.Circle((x, 0), d_s, edgecolor='red', fill=False, alpha=0.1)
        ax.add_patch(c_safe)

    plot_actor(ax, scale=0.5, orientation=0)

    ax.axis('equal')
    ax.set_xlabel("X (m)", fontsize=16)
    ax.set_ylabel("Y (m)", fontsize=16)

    fig.savefig('meaning_of_infogain.pdf')

    fig, ax = plt.subplots()

    sb.lineplot(x=ego_traj[:, 0], y=ego_traj[:, 1], color='black', alpha=0.4)

    # create two patches for occlusion
    t = 3  # (3.0 * d_s) / (V_ego)
    i_start = int(t / dt)
    d_o = t * V_o1
    poly = plt.Circle((p_risk_upper[i_start, :]), d_o, edgecolor='firebrick', facecolor='gold', alpha=0.6)
    ax.add_patch(poly)
    poly = plt.Circle((p_risk_upper[i_start, :]), 0.1, edgecolor='black', facecolor='black', alpha=0.8)
    ax.add_patch(poly)

    # sb.lineplot(x=p_risk_lower[:, 0], y=p_risk_lower[:, 1], color='darkblue')
    sb.lineplot(x=p_risk_upper[:, 0], y=p_risk_upper[:, 1], color='darkblue')

    plot_actor(ax, scale=0.5, orientation=0)
    plot_actor(ax, pos=p_risk_upper[i_start], scale=0.5, orientation=-np.pi/2)

    ax.axis('equal')
    ax.set_xlabel("X (m)", fontsize=16)
    ax.set_ylabel("Y (m)", fontsize=16)

    fig.savefig('risk_of_an_agent.pdf')


if __name__ == '__main__':
    main()
