import scipy.special as special
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['pcolor.shading'] = 'nearest'

# Utilities


def plot_amp(xticks, yticks, dst, posSrc=None, cent=None, radi=None):
    """plot amplitude distribution

    Parameters
    ----------------
        xticks: ticks of x-axis
        yticks: ticks of y-axis
        dst: amplitude distribution for plot
        posSrc: postion of source
        cent: center of reproduction area
        radi: radius of reproduction area

    Return
    ----------
        fig: Figure object
        ax: Axes object
    """
    X, Y = np.meshgrid(xticks, yticks)

    fig, ax = plt.subplots()
    quadmesh = ax.pcolormesh(X, Y, dst.real, cmap="pink_r")
    ax.set_aspect('equal')
    quadmesh.set_clim(vmin=-1.2, vmax=1.2)
    ax.grid(False)

    cbar = fig.colorbar(quadmesh, ax=ax, ticks=np.arange(-1, 1+0.001, 0.5))
    cbar.set_label("Amplitude (real part)")
    cbar.ax.tick_params(labelsize=13)

    if posSrc is not None:
        ax.scatter(posSrc[:, 0], posSrc[:, 1], s=60, linewidth=0.8,
                   marker='o', c="blue", label="Source", edgecolor="white")

    if cent is not None and radi is not None:
        obj_zone = patches.Circle(
            xy=cent, radius=radi, ec='#000000', fill=False, ls='dashed')
        ax.add_patch(obj_zone)
        edge = patches.Circle(xy=cent, radius=radi+0.01,
                              ec='#ffffff', fill=False, ls='dashed')
        ax.add_patch(edge)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    fig.tight_layout()

    return fig, ax


def plot_err(xticks, yticks, des, syn, posSrc=None, cent=None, radi=None):
    """plot normalized error distribution
    Parameters
    ----------------
        xticks: ticks of x-axis
        yticks: ticks of y-axis
        des: desired distribution
        syn: synthesized distribution
        posSrc: postion of source
        cent: center of reproduction area
        radi: radius of reproduction area

    Return
    ----------
        fig: Figure object
        ax: Axes object
    """
    X, Y = np.meshgrid(xticks, yticks)
    fig, ax = plt.subplots()
    ne = 10 * np.log10(abs(des - syn)**2/abs(syn)**2)
    quadmesh = ax.pcolormesh(X, Y, ne, cmap="pink_r")
    ax.set_aspect('equal')
    quadmesh.set_clim(vmin=-40, vmax=10)
    ax.grid(False)
    cbar = fig.colorbar(quadmesh, ax=ax)
    cbar.set_label("Normalized error (dB)")
    cbar.ax.tick_params(labelsize=13)

    if posSrc is not None:
        ax.scatter(posSrc[:, 0], posSrc[:, 1], s=60, linewidth=0.8,
                   marker='o', c="blue", label="Source", edgecolor="white")

    if cent is not None and radi is not None:
        obj_zone = patches.Circle(
            xy=cent, radius=radi, ec='#000000', fill=False, ls='dashed')
        ax.add_patch(obj_zone)
        edge = patches.Circle(xy=cent, radius=radi+0.01,
                              ec='#ffffff', fill=False, ls='dashed')
        ax.add_patch(edge)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    fig.tight_layout()

    return fig, ax


def SDR(des, syn):
    """calculate SDR
    Parameters
    ----------------
        des, syn: matrix which size is the same

    Return
    ---------
        SDR value
    """
    des = np.array(des).reshape(-1).real
    syn = np.array(syn).reshape(-1).real
    sdr = 10*np.log10(np.linalg.norm(des, ord=2)**2 /
                      np.linalg.norm(des - syn, ord=2)**2)
    return round(sdr, 2)


def block_bi_recurrence(C, D, E, H, m):
    """block bi-recurrence method ( reference: ブロック三重対角連立一次方程式の並列解析,成富敬,1997 )
    Parameter
    --------------
        C, E: (p-1,q,q), block matrix list
        D: (p,q,q), block matrix list
        H: (p,q), vector list
        m: balancer ( int )
    """
    # initialization
    C = np.array(C)
    E = np.array(E)
    D = np.array(D)
    H = np.array(H)
    if C.shape != E.shape:
        raise ValueError
    p = D.shape[0]
    q = D.shape[1]

    C = np.concatenate([np.zeros((1, q, q)), C])  # 論文中のindexが2から始まっているため

    # block attractive stage
    Lmd = np.zeros((p, q, q), dtype=complex)
    Gam = np.zeros((p, q), dtype=complex)
    D_inv = np.linalg.inv(D[0])
    Lmd[0] = -D_inv@E[0]
    Gam[0] = D_inv@H[0]
    D_inv = np.linalg.inv(D[-1])
    Lmd[p-1] = -D_inv@C[p-1]
    Gam[p-1] = D_inv@H[p-1]
    for i in np.arange(1, m-1+1):
        inv = np.linalg.inv(D[i]+C[i]@Lmd[i-1])
        Lmd[i] = -inv@E[i]
        Gam[i] = inv@(H[i]-C[i]@Gam[i-1])
    for k in np.arange(p-2, m-1, -1):
        inv = np.linalg.inv(D[k]+E[k]@Lmd[k+1])
        Lmd[k] = -inv@C[k]
        Gam[k] = inv@(H[k]-E[k]@Gam[k+1])

    # block interactive stage
    X = np.zeros((p, q), dtype=complex)
    X[m-1] = np.linalg.inv(np.eye(q)-Lmd[m-1]@Lmd[m]
                           )@(Gam[m-1]+Lmd[m-1]@Gam[m])
    X[m] = np.linalg.inv(np.eye(q)-Lmd[m]@Lmd[m-1])@(Gam[m]+Lmd[m]@Gam[m-1])

    # block repulsive stage
    for i in np.arange(m-2, -1, -1):
        X[i] = Lmd[i]@X[i+1]+Gam[i]
    for k in np.arange(m+1, p-1+1):
        X[k] = Lmd[k]@X[k-1]+Gam[k]

    return X


def ILD(h_L, h_R):
    h_L = np.array(h_L)
    h_R = np.array(h_R)
    assert h_L.shape == h_R.shape
    val = 10*np.log10(sum(h_L**2) / sum(h_R**2))

    return val


def ITD(h_L, h_R):
    h_L = np.array(h_L)
    h_R = np.array(h_R)
    assert len(h_L) == len(h_R)

    if len(h_L.shape) == 1:
        T = len(h_L)
        val1 = []
        for tau in range(1, T):
            val1.insert(0, sum(h_R[:T-tau]*h_L[tau:]) /
                        (np.sqrt(sum(h_R[:T-tau]**2 * h_L[tau:]**2)+1e-6)))
        for tau in range(T):
            val1.append(sum(h_L[:T-tau]*h_R[tau:]) /
                        (np.sqrt(sum(h_L[:T-tau]**2 * h_R[tau:]**2)+1e-6)))
        t = -T+1 + np.argmax(val1)
    else:
        raise ValueError
    return t


if __name__ == '__main__':
    pass
