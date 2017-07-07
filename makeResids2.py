from __future__ import division
import numpy as np
import matplotlib.mlab as ml
import utils as PALutils
import os,sys

# make simulated observations
def makeSimObservationsFull(start, cadence, Tspan, rmss, obs,
                            RADEC=False, PX=False, DMX=False,
                            pepoch=2005, gap=True):

    # get number of points
    start -= pepoch
    Tspan -= pepoch
    dur = Tspan - start
    Np = int(dur * cadence)

    ut = np.zeros(Np)
    ut = np.linspace(start, Tspan, Np)
    freqs = np.ones(Np) * 1400

    # gap from GBT maintenence
    if gap and obs == 1:
        ut = np.array([x for x in ut if x < 0.5 or x > 1.5])
        Np = int(len(ut))
    err = np.zeros(Np)

    for ii in range(len(rmss)-1):
        ind = np.logical_and(ut >= rmss[ii][0]-pepoch, ut <= rmss[ii+1][0]-pepoch)
        err[ind] = rmss[ii][1]

    # create Design matrix
    M = PALutils.createDesignmatrix(ut*3.16e7, freqs, RADEC=RADEC, PX=PX, DMX=DMX)

    # create R matrix
    R = PALutils.createRmatrix(M, err)

    return (ut, err, freqs, R, M)
