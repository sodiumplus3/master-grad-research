import scipy.special as special
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['pcolor.shading'] = 'nearest'

# Function


def cart2sph(x, y, z):
    r_xy = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(r_xy, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    return r, phi, theta


def sph2cart(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sph_hn1(n, x):
    if np.abs(x) < 1e-5:
        x = 1e-5
    return special.spherical_jn(n, x)+1j*special.spherical_yn(n, x)


def sph_hn2(n, x):
    if np.abs(x) < 1e-5:
        x = 1e-5
    return special.spherical_jn(n, x)-1j*special.spherical_yn(n, x)


def sph_harm(n, m, phi, theta):
    return special.sph_harm(m, n, phi, theta)
    # return (-1)**np.abs(m) * special.sph_harm(m, n, phi, theta)


def sf_basis2d(order, r, phi, k):
    return special.jv(order, k*r) * np.exp(1j*order*phi)


def sf_basis3d(n, m, r, phi, theta, k):
    return special.spherical_jn(n, k*r) * sph_harm(n, m, phi, theta)


def sphericalwave(r, rl, k, amp=1):
    """Green function in 3D.
    Parameter
    --------------
        r : (3,) or (L,3), observation point(s)
        rl : (3,) or (L,3), source point(s)
        k : wave number
        amp : amplitude
    """
    r = np.array(r)
    rl = np.array(rl)
    if len(r.shape) == 1:
        l2 = np.linalg.norm(r-rl, ord=2)
    elif len(r.shape) == 2:
        l2 = np.linalg.norm(r-rl, ord=2, axis=1)
    p = amp*np.exp(-1j*k*l2)/(4*np.pi*l2)
#     p = amp*np.exp(1j*k*l2)/(4*np.pi*l2)
    return p


def cylindricalwave(r, rl, k, amp=1):
    """Cylindricalwave in 2D.
    Parameter
    --------------
        r : (2,) or (L,2), observation point(s)
        rl : (2,) or (L,2), source point(s)
        k : wave number
        amp : amplitude
    Return
    ---------
        pressure
    """
    r = np.array(r)
    rl = np.array(rl)
    if len(r.shape) == 1:
        r = r[None, :]
    if len(rl.shape) == 1:
        rl = rl[None, :]
    l2 = np.linalg.norm(r-rl, ord=2, axis=1)
    l2[l2 < 1e-4] = 1e-4
    p = - amp*1j/4*special.hankel2(0, k*l2)
#     p = amp*1j/4*special.hankel1(0, k*l2)
    return p


def planewave(r, phi, theta, k, amp=1):
    """Plane wave transfer function.
    Parameter
    --------------
        (r, phi, theta) : listening point in polar coordinate
        k : wave number
        amp: amplitude
    Return
    ---------
        pressure
    """
    kx, ky, kz = sph2cart(k, phi, theta)
    p = amp * np.exp(-1j * (kx * r[0] + ky * r[1] + kz * r[2]))
#     p = amp * np.exp(1j * (kx * r[0] + ky * r[1] + kz * r[2]))
    return p


def sphericalwave_coef(r_c, r_s, k, N, amp=1, polar=False):
    """Spherical wave expansion coefficients.
    Parameter
    --------------
        r_c : listening position (3D)
        r_s : source position (3D)
        k : wave number
        N : truncation order of expansion
        amp: amplitude
        polar: if r_c and r_s are given in polar coordinates
    Return
    ---------
        coefficients
    """
    assert type(polar) == bool

    if np.array(r_c).shape != (3,) or np.array(r_s).shape != (3,):
        raise ValueError("Matrix size is wrong.")
    n, m = nmvec(N)

    if polar:
        x_c, y_c, z_c = sph2cart(r_c[0], r_c[1], r_c[2])
        x_s, y_s, z_s = sph2cart(r_s[0], r_s[1], r_s[2])
    else:
        (x_c, y_c, z_c) = r_c
        (x_s, y_s, z_s) = r_s

    r, phi, theta = cart2sph(x_s-x_c, y_s-y_c, z_s-z_c)

    coef = - amp*1j*k*sph_hn2(n, k*r)*sph_harm(n, m, phi, theta).conj()
#     coef = amp*1j*k*sph_hn1(n, k*r)*sph_harm(n, m, phi, theta).conj()
    return coef


def cylindricalwave_coef(r_c, r_s, k, N, amp=1, polar=False):
    """Cylindrical wave expansion coefficients.
    Parameter
    --------------
        r_c : listening position (2D)
        r_s : source position (2D)
        k : wave number
        N : truncation order of expansion
        amp: amplitude
        polar: if r_c and r_s are given in polar coordinates
    Return
    ---------
        coefficients
    """
    if np.array(r_c).shape != (2,) or np.array(r_s).shape != (2,):
        raise ValueError("Matrix size is wrong.")
    n = nvec(N)

    if polar:
        x_c, y_c, _ = sph2cart(r_c[0], r_c[1], np.pi/2)
        x_s, y_s, _ = sph2cart(r_s[0], r_s[1], np.pi/2)
    else:
        (x_c, y_c) = r_c
        (x_s, y_s) = r_s

    r = np.sqrt((x_c-x_s)**2 + (y_c-y_s)**2)
    phi = np.arctan2(y_s-y_c, x_s-x_c)
    coef = - amp*1j/4*special.hankel2(n, k*r)*np.exp(-1j*n*phi)
#     coef = amp*1j/4*special.hankel1(n, k*r)*np.exp(-1j*n*phi)
    return coef


def planewave_coef(r, phi, theta, k, N, amp=1, cylind=False):
    """Spherical / Cylindrical wave expansion coefficients.
    Parameter
    --------------
        r : center of expansion
        (phi, theta) : propagating direction of planewave
        k : wave number
        N : truncation order of expansion
        amp: amplitude
        cylind: if use cylindrical wave
    Return
    ---------
        coefficients
    """
    kx, ky, kz = sph2cart(k, phi, theta)
    if cylind:
        n = nvec(N)
        coef = amp * (-1j)**n * np.exp(-1j*n*phi)
#         coef = amp * 1j**n * np.exp(-1j*n*phi)
    else:
        n, m = nmvec(N)
        coef = amp * 4 * np.pi * (-1j)**n * sph_harm(n, m, phi, theta).conj()
#         coef = amp * 4 * np.pi * 1j**n * sph_harm(n, m, phi, theta).conj()
#     coef *=  np.exp(-1j * (kx*r[0] + ky*r[1] + kz*r[2]))
    return coef.reshape(-1, 1)


def nmvec(order, rep=None):
    n = np.array([0])
    m = np.array([0])
    for nn in np.arange(1, order+1):
        nn_vec = np.tile([nn], 2*nn+1)
        n = np.append(n, nn_vec)
        mm = np.arange(-nn, nn+1)
        m = np.append(m, mm)
    if rep is not None:
        n = np.tile(n[:, None], (1, rep))
        m = np.tile(m[:, None], (1, rep))
    return n, m


def nvec(order):
    return np.arange(-order, order+1)


def Kronecker_delta(x, y):
    if x == y:
        return 1
    else:
        return 0


if __name__ == '__main__':
    pass
