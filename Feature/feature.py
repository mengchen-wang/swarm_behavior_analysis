import numpy as np
from numba import njit

def integrate_field(bx, by, bz):
    return np.sqrt((bx**2+by**2+bz**2)/2)

@njit
def clip(x, minv, maxv):
    if x > maxv:
        return maxv
    if x < minv:
        return minv
    return x

@njit
def angular_velocity(fx, fy, fz, bx, by, bz, phi_x, phi_y, phi_z):
    t = np.linspace(0, 1.001, 1000)
    theta = 0
    for i, _ in enumerate(t):
        __ = t[i+1]
        B1 = np.array([bx*np.sin(2*np.pi*fx*_+phi_x), by*np.sin(2*np.pi*fy*_+phi_y), bz*np.sin(2*np.pi*fz*_+phi_z)])
        B2 = np.array([bx*np.sin(2*np.pi*fx*__+phi_x), by*np.sin(2*np.pi*fy*__+phi_y), bz*np.sin
        (2*np.pi*fz*__+phi_z)])
        if np.sqrt((B1*B1).sum())*np.sqrt((B2*B2).sum()) == 0:
            dtheta = 0
        else:
            dtheta = np.arccos(clip((B1*B2).sum()/(np.sqrt((B1*B1).sum())*np.sqrt((B2*B2).sum())),-1,1))
        theta += dtheta
    return theta

def radial(fx, fy, phi_x, phi_y):
    t = np.linspace(0, 1.001, 1000)
    count = 0
    for time in t:
        if np.sin(2*np.pi*fx*time+phi_x)*np.sin(2*np.pi*fy*time+phi_y)*np.cos(2*np.pi*fx*time+phi_x)*np.cos(2*np.pi*fy*time+phi_y)>=0:
            count += 1
    return count/1000

def radial3d(fx, fy, fz, phi_x, phi_y, phi_z):
    Radial = []
    Radial.append(radial(fx, fy, phi_x, phi_y))
    Radial.append(radial(fx, fz, phi_x, phi_z))
    Radial.append(radial(fz, fy, phi_z, phi_y))
    return Radial

def feature(X_train_pre):
    X_feature = []
    for j in range(X_train_pre.shape[0]):
        X = X_train_pre[j]
        in_f = integrate_field(X[3], X[4], X[5])
        an_v = angular_velocity(X[0], X[1], X[2], X[3], X[4], X[5], 0,  X[6], X[7])
        Radial = radial3d(X[0], X[1], X[2], 0,  X[6], X[7])
        X_feature.append([X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], Radial[0], Radial[1], Radial[2], in_f, an_v, in_f/an_v])
    return np.array(X_feature)

def data_augmentation(X_train_pre, y_train_pre, copynumber):
    X_train, y_train = [],[]
    for j in range(X_train_pre.shape[0]):
        X = X_train_pre[j]
        for i in range(copynumber):
            x = np.random.random()
            X_train.append([X[0], X[1], X[2], X[3], X[4], X[5], np.sin(x*2*np.pi*X[1]*X[2]), np.sin(X[6]*np.pi/6+x*2*np.pi*X[0]*X[2]), np.sin(X[7]*np.pi/2+x*2*np.pi*X[1]*X[0]), np.cos(x*2*np.pi*X[1]*X[2]), np.cos(X[6]*np.pi/6+x*2*np.pi*X[0]*X[2]), np.cos(X[7]*np.pi/2+x*2*np.pi*X[1]*X[0])])
            X_train.append([X[0], X[1], X[2], X[3], X[4], X[5], np.sin(X[6]*np.pi/6+x*2*np.pi*X[0]*X[2]), np.sin(x*2*np.pi*X[1]*X[2]), np.sin(X[7]*np.pi/2+x*2*np.pi*X[1]*X[0]), np.cos(X[6]*np.pi/6+x*2*np.pi*X[0]*X[2]), np.cos(x*2*np.pi*X[1]*X[2]), np.cos(X[7]*np.pi/2+x*2*np.pi*X[1]*X[0])])
            y_train.append(y_train_pre[j])
            y_train.append(y_train_pre[j])
    return np.array(X_train), np.array(y_train)