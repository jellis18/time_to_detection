#!/usr/bin/env python

from __future__ import division
import numpy as np
import os,sys, time
import makeResids2 as m
import utils as PALutils
from scipy.optimize import brentq
import scipy.special as ss
import optparse

parser=optparse.OptionParser()

parser.add_option('-A','--Agw',action='store',type='float',dest='Agw',help='GW amplitude',default=1e-15)
parser.add_option('-G','--gamGW',action='store',type='float',dest='gam_gw',help='GW spectral index',default=13./3.)
parser.add_option('-a','--Ared',action='store',type='float',dest='Ared',help='red noise amplitude',default=4e-16)
parser.add_option('-g','--gamred',action='store',type='float',dest='gam_red',help='red noise spectral index',default=5.001)
parser.add_option('-T','--tspan',action='store',type='float',dest='tspan',help='Time span (years)',default=20)
parser.add_option('-r','--nreal',action='store',type='int',dest='nreal',help='Number of realizations per simulation',default=100)
parser.add_option('-l','--label',action='store',type='int',dest='label',help='label for output files',default=0)
parser.add_option('-o','--outDir',action='store',type='string',dest='outDir',help='output directory',default='./')
parser.add_option('-c','--cadence',action='store',type='int',dest='cadence',help='number of points per year',default=20)
parser.add_option('--seed',action='store',type='int',dest='seed',help='random # seed',default=111)
parser.add_option('--best', dest='best', action='store', type=int, default=0,
                   help='Only use best pulsars based on weighted rms (default = 0, use all)')
parser.add_option('--RADEC',action='store_true',dest='RADEC',help='fit for RA and DEC',default=False)
parser.add_option('--PX',action='store_true',dest='PX',help='fit for PX',default=False)
parser.add_option('--DMX',action='store_true',dest='DMX',help='fit for DMX',default=False)
parser.add_option('--freq',action='store',dest='freq', type='float',help='frequency',default=1e-8)
parser.add_option('--redFromFile',action='store_true',dest='redFromFile',
                  help='Read red noise from file',default=False)
parser.add_option('--bestcase',action='store_true',dest='bestcase',
                  help='Best case scenario',default=False)
parser.add_option('--worstcase',action='store_true',dest='worstcase',
                  help='Worst case scenario',default=False)
parser.add_option('--statuscase',action='store_true',dest='statuscase',
                  help='Status quo case scenario',default=False)
parser.add_option('--cwcase',action='store_true',dest='cwcase',
                  help='CW case scenario',default=False)
parser.add_option('--detprob',action='store',dest='detprob',
                  help='Detection probability',default=0.95, type=float)

# parse arguments
(opts,args)=parser.parse_args()

def ptSum(N, fp0):
    """
    Compute False alarm rate for Fp-Statistic. We calculate
    the log of the FAP and then exponentiate it in order
    to avoid numerical precision problems
    @param N: number of pulsars in the search
    @param fp0: The measured value of the Fp-statistic
    @returns: False alarm probability ad defined in Eq (64)
              of Ellis, Seiemens, Creighton (2012)
    """

    n = np.arange(0,N)

    return np.sum(np.exp(n*np.log(fp0)-fp0-np.log(ss.gamma(n+1))))

# compute f_p statistic
def fpStat(psr, f0):
    """
    Computes the Fp-statistic as defined in Ellis, Siemens, Creighton (2012)

    :param psr: List of pulsar object instances
    :param f0: Gravitational wave frequency

    :return: Value of the Fp statistic evaluated at f0

    """

    fstat=0.
    npsr = len(psr)

    # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
    N = np.zeros(2)
    M = np.zeros((2, 2))
    for ii,p in enumerate(psr):

        # Define A vector
        A = np.zeros((2, p.ntoa))
        A[0,:] = 1./f0**(1./3.) * np.sin(2*np.pi*f0*p.toas)
        A[1,:] = 1./f0**(1./3.) * np.cos(2*np.pi*f0*p.toas)

        N = np.array([np.dot(A[0,:], np.dot(p.invCov, p.res)), \
                      np.dot(A[1,:], np.dot(p.invCov, p.res))])

        # define M matrix M_ij=(A_i|A_j)
        for jj in range(2):
            for kk in range(2):
                M[jj,kk] = np.dot(A[jj,:], np.dot(p.invCov, A[kk,:]))

        # take inverse of M
        Minv = np.linalg.inv(M)
        fstat += 0.5 * np.dot(N, np.dot(Minv, N))

    # return F-statistic
    return fstat


# makeshift pulsar class
class pulsar(object):

    def __init__(self, theta, phi, ntoa, toas, res, err, designmatrix, invCov, obs,
                 freqs, name='J0000+0000',  dist=1.0, distErr=0.1, Ared=0.0,
                 gred=3.01, Cnoise12=None):

        self.theta = theta
        self.phi = phi
        self.ntoa = ntoa
        self.toas = toas
        self.res = res
        self.err = err
        self.invCov = invCov
        self.designmatrix = designmatrix
        self.dist = dist
        self.distErr = distErr
        self.freqs = freqs
        self.Ared = Ared
        self.gred = gred
        self.Cnoise12 = Cnoise12
        self.name = name

        if obs == 0:
            self.obs = 'AO'
        elif obs == 1:
            self.obs = 'GBT'
        elif obs == 2:
            self.obs = 'IPTA'
        elif obs == 3:
            self.obs = 'FAST'
        elif obs == 4:
            self.obs = 'MeerKat'
        else:
            print 'Unknown Observatory code'
            self.obs = 'Unknown'

    def cosMu(self, gwtheta, gwphi):
        """
        Calculate cosine of angle between pulsar and GW

        """
        # calculate unit vector pointing at GW source
        omhat = [np.sin(gwtheta)*np.cos(gwphi), np.sin(gwtheta)*np.sin(gwphi), np.cos(gwtheta)]

        # calculate unit vector pointing to pulsar
        phat = [np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), \
                np.cos(self.theta)]

        return np.dot(omhat, phat)

    def rms(self):

        """
        Return weighted RMS in seconds

        """

        W = 1/self.err**2

        return np.sqrt(np.sum(self.res**2*W)/np.sum(W))


def upperLimitFunc(h, freq, nreal, th=None, ph=None, dist=None, dp=0.95):
    """
    Compute the value of the fstat for a range of parameters, with fixed
    amplitude over many realizations.

    @param h: value of the strain amplitude to keep constant
    @param fstat_ref: value of fstat for real data set
    @param freq: GW frequency
    @param nreal: number of realizations

    """
    Tmaxyr = np.array([(p.toas.max() - p.toas.min())/3.16e7 for p in psr]).max()
    count = 0
    np.random.seed()
    hs = []
    for ii in range(nreal):

        # draw parameter values
        gwtheta = np.arccos(np.random.uniform(-1, 1))
        gwphi = np.random.uniform(0, 2*np.pi)
        gwphase = np.random.uniform(0, 2*np.pi)
        gwinc = np.arccos(np.random.uniform(-1, 1))
        gwpsi = np.random.uniform(0, np.pi)

        # check to make sure source has not coalesced during observation time
        coal = True
        while coal:
            gwmc = 10**np.random.uniform(7, 10)
            tcoal = 2e6 * (gwmc/1e8)**(-5/3) * (freq/1e-8)**(-8/3)
            if tcoal > Tmaxyr:
                coal = False


        # determine distance in order to keep strain fixed
        gwdist = 4 * np.sqrt(2/5) * (gwmc*4.9e-6)**(5/3) * (np.pi*freq)**(2/3) / h

        # convert back to Mpc
        gwdist /= 1.0267e14

        # check for fixed sky location
        if th is not None:
            gwtheta = th
        if ph is not None:
            gwphi = ph
        if dist is not None:
            gwdist = dist
            gwmc = ((gwdist*1.0267e14)/4/np.sqrt(2/5)/(np.pi*freq)**(2/3)*h)**(3/5)/4.9e-6

        # make stochastic background residuals
        if A_gw:
            stoch_res = PALutils.createGWB(psr, A_gw, gam_gw)

        if A_red:
            red_res = PALutils.createGWB(psr, A_red, gam_red, noCorr=True)

        # create residuals
        #psr2 = []
        for ct,p in enumerate(psr):

            #total time span
            #T = p.toas.max() - p.toas.min()
            #if 1/T <= freq:
            inducedRes = PALutils.createResiduals(p, gwtheta, gwphi, gwmc, gwdist, \
                            freq, gwphase, gwpsi, gwinc, evolve=False, psrTerm=False)

            # make white noise for each pulsar
            res = np.array([np.random.randn()*error for error in p.err])
            res += inducedRes

            if A_gw:
                res += stoch_res[ct]
            if A_red:
                res += red_res[ct]

            #psr2.append(p)

            # replace residuals in pulsar object
            #psr2[-1].res = np.dot(R[ct], res)
            p.res = np.dot(R[ct], res)

        # compute f-statistic
        #print len(psr), len(psr2)
        #fpstat = fpStat(psr, freq)
        feStat = PALutils.feStat(psr, gwtheta, gwphi, freq)
        if h == 1e-17:
            hs.append(feStat)

        # check to see if larger than in real data
        #if ptSum(len(psr), fpstat) < 1e-4:
        if np.exp(-feStat) < 1e-3:
            count += 1

    # now get detection probability
    detProb = count/nreal
    if h == 1e-17:
        np.savetxt('hs.txt', np.array(hs))

    if dist is not None:
        print '%e %e %f\n'%(freq, gwmc, detProb)
    else:
        print freq, h, detProb

    return detProb - dp

def parse_pulsar(pname):

    if '+' in pname:
        ra = pname.split('J')[-1].split('+')[0]
        rah = float(ra[:2])
        ram = float(ra[2:])
        phi = rah * np.pi/12 + ram * np.pi/12/60

        dec = pname.split('J')[-1].split('+')[-1]
        decd = float(dec[:2])
        decm = float(dec[2:])
        theta = np.pi/2 - (decd * np.pi/180 + decm * np.pi/180/60)

    if '-' in pname:
        ra = pname.split('J')[-1].split('-')[0]
        rah = float(ra[:2])
        ram = float(ra[2:])
        phi = rah * np.pi/12 + ram * np.pi/12/60

        dec = pname.split('J')[-1].split('-')[-1]
        decd = float(dec[:2])
        decm = float(dec[2:])
        theta = np.pi/2 + (decd * np.pi/180 + decm * np.pi/180/60)

    return theta, phi


# set constants
A_red = opts.Ared             # red noise amplitude (in strain units)
gam_red = opts.gam_red        # red noise spectral index
A_gw = opts.Agw               # GW amplitude (strain units)
gam_gw = opts.gam_gw          # GW spectral index (SMBHB)
Tspan = opts.tspan            # total time span in years
Nreal = opts.nreal            # number of realizations
fL = 1./50                    # low frequency cutoff (yr^-1)

# convert strain amplitude to power law amplitude
Amp_gw = A_gw**2/24./np.pi**2
Amp_red = A_red**2/24./np.pi**2

gwRMS = 1.12*(A_gw/1e-15)*Tspan**(5./3.)
redRMS = 2.05*(A_red/1e-15)*Tspan**((gam_red-1)/2)/(np.sqrt(gam_red-1))

print 'gw rms = %g ns'%gwRMS
print 'red rms = %g ns'%redRMS

# read in GBT file
if opts.bestcase:
    gbtfile = np.genfromtxt('files/gbt_pulsar_file.txt', dtype='S42')
    aofile = np.genfromtxt('files/ao_pulsar_file.txt', dtype='S42')
    gbtmed = np.loadtxt('files/gbt_med_rms_vals.txt')
elif opts.worstcase:
    gbtfile = np.genfromtxt('files/gbt_pulsar_file_worst2.txt', dtype='S42')
    aofile = np.genfromtxt('files/ao_pulsar_file_worst2.txt', dtype='S42')
    gbtmed = np.loadtxt('files/gbt_med_rms_vals_worst.txt')
elif opts.statuscase:
    gbtfile = np.genfromtxt('files/gbt_pulsar_file_status.txt', dtype='S42')
    aofile = np.genfromtxt('files/ao_pulsar_file_status.txt', dtype='S42')
    gbtmed = np.loadtxt('files/gbt_med_rms_vals_status.txt')
    aomed = np.loadtxt('files/ao_med_rms_vals_status.txt')
elif opts.cwcase:
    gbtfile = np.genfromtxt('files/gbt_pulsar_file_cw.txt', dtype='S42')
    aofile = np.genfromtxt('files/ao_pulsar_file_cw.txt', dtype='S42')
    gbtmed = np.loadtxt('files/gbt_med_rms_vals_cw.txt')

# update to status quo case 10/17/2017
# push wideband receiver back one year
if opts.statuscase:
    gbtfile = np.hstack((gbtfile[:,:18], gbtfile[:,17][:, None], gbtfile[:,18:-1]))
    aofile = np.hstack((aofile[:,:18], aofile[:,17][:, None], aofile[:,18:-1]))


gbtpsrlist = []
for gp in gbtfile:
    gpf = map(float, gp[1:])
    gbtdict = {}
    gbtdict['name'] = gp[0]
    gbtdict['start'] = gpf[1]
    gbtdict['sky'] = (gpf[2], gpf[3])
    gbtdict['rms'] = zip(np.double(np.arange(2005,2027)), np.array(gpf[4:]))
    gbtpsrlist.append(gbtdict)

# read in AO file
aopsrlist = []
for gp in aofile:
    gpf = map(float, gp[1:])
    aodict = {}
    aodict['name'] = gp[0]
    aodict['start'] = gpf[1]
    aodict['sky'] = (gpf[2], gpf[3])
    aodict['rms'] = zip(np.double(np.arange(2005,2027)), np.array(gpf[4:]))
    aopsrlist.append(aodict)

gbtmed = np.concatenate((np.zeros(11), gbtmed))
gbtmed = zip(np.double(np.arange(2005,2027)), gbtmed)
if opts.statuscase:
    aomed = np.concatenate((np.zeros(11), aomed))
    aomed = zip(np.double(np.arange(2005,2027)), aomed)

# red pulsar list
if opts.redFromFile:
    red_names = np.loadtxt('files/red_pulsars.txt', dtype='S42', usecols=[0])
    red_vals = np.loadtxt('files/red_pulsars.txt', usecols=[1,2])
    red_dict = {}
    for rn, rv in zip(red_names, red_vals):
        red_dict[rn] = rv

# start loop over dates
dates = np.arange(2006, opts.tspan+2006)
Nt = len(dates)
for ct1, date in enumerate(dates):

    # initialize pulsar class
    psr = []

    if date > 2026:
        gbtmed.append((date, gbtmed[-1][1]))
        for gb in gbtpsrlist:
            gb['rms'].append((date, gb['rms'][-1][1]))

    # add new GBT pulsars
    if date >= 2016:
        for ii in range(2):
            np.random.seed(opts.seed*(ii+1)*int(date))
            theta = np.arccos(np.random.uniform(-1, np.cos(0)))
            phi = np.random.uniform(0, 2*np.pi)
            x = {}
            x['name'] = 'J0000+0000'
            x['start'] = date
            x['sky'] = (theta, phi)
            x['rms'] = gbtmed
            gbtpsrlist.append(x)


    # GBT pulsars
    for gb in gbtpsrlist:
        if gb['start'] <= date-1:
            ut, err, freqs, RR, M = m.makeSimObservationsFull(
               gb['start'], opts.cadence, date,
               gb['rms'], 1, RADEC=opts.RADEC,
               PX=opts.PX, DMX=opts.DMX,
                pepoch=2005, gap=True)


            psr.append(pulsar(gb['sky'][0], gb['sky'][1], len(err),
                              ut*3.16e7, err*3.16e7, err*1e-6, M,
                              0, 1,freqs, gb['name']))

    if date > 2026:
        if opts.statuscase:
            aomed.append((date, aomed[-1][1]))
        for gb in aopsrlist:
            gb['rms'].append((date, gb['rms'][-1][1]))

    # add new AO pulsars
    if date >= 2016 and opts.statuscase:
        for ii in range(2):
            np.random.seed(opts.seed*(ii+2)*int(date))
            theta = np.arccos(np.random.uniform(np.cos(1.59), np.cos(0.9)))
            phi = np.random.uniform(0, 2*np.pi)
            x = {}
            x['name'] = 'J0000+0000'
            x['start'] = date
            x['sky'] = (theta, phi)
            x['rms'] = aomed
            aopsrlist.append(x)


    # AO pulsars
    for gb in aopsrlist:
        if gb['start'] <= date-1:
            ut, err, freqs, RR, M = m.makeSimObservationsFull(
               gb['start'], opts.cadence, date,
               gb['rms'], 0, RADEC=opts.RADEC,
               PX=opts.PX, DMX=opts.DMX,
                pepoch=2005, gap=True)

            psr.append(pulsar(gb['sky'][0], gb['sky'][1], len(err),
                              ut*3.16e7, err*3.16e7, err*1e-6, M,
                              0, 0,freqs, gb['name']))


rmss = [p.err.mean() for p in psr]
ind = np.argsort(rmss)
if opts.best:
    psr = [psr[ii] for ii in ind[:opts.best]]
    for ct, p in enumerate(psr):
        print p.name, p.err.mean()*1e6

# R matrices
R = [PALutils.createRmatrix(p.designmatrix, p.err) for p in psr]


# G matrices
G = [PALutils.createGmatrix(p.designmatrix) for p in psr]

# creat inverse covariance matrices
for ct, p in enumerate(psr):

    # matrix of time lags
    tm = PALutils.createTimeLags(p.toas, p.toas)

    Ared, gred = A_red, gam_red
    if opts.redFromFile:
        if p.name in red_dict:
            Ared = 10**red_dict[p.name][0]
            gred = red_dict[p.name][1]
            print 'Setting Ared={}, gred={} for PSR {}'.format(Ared, gred, p.name)

    # red noise
    C = PALutils.createRedNoiseCovarianceMatrix(tm.copy(), A_gw, gam_gw)
    C += PALutils.createRedNoiseCovarianceMatrix(tm.copy(), Ared, gred)

    # white
    C += PALutils.createWhiteNoiseCovarianceMatrix(p.err, 1, 0)

    # inverse
    tmp = np.dot(G[ct].T, np.dot(C, G[ct]))

    p.invCov = np.dot(G[ct], np.dot(np.linalg.inv(tmp), G[ct].T))


# compute minimum detectable amplitude
hhigh = 1e-12
hlow = 5e-18
xtol = 1e-17
freq = opts.freq
nreal = opts.nreal

# perfrom upper limit calculation
inRange = False
while inRange == False:

    try:    # try brentq method
        h_up = brentq(upperLimitFunc, hlow, hhigh, xtol=xtol, \
              args=(freq, nreal, None, None, None, opts.detprob))
        inRange = True
    except ValueError:      # bounds not in range
        if hhigh < 1e-10:   # don't go too high
            hhigh *= 2      # double high strain
        else:
            h_up = hhigh    # if too high, just set to upper bound
            inRange = True

fname = 'upper_{0}_{1}.txt'.format(freq, opts.nreal)

# output data
if not os.path.exists(opts.outDir):
    try:
        os.makedirs(opts.outDir)
    except OSError:
        pass

fout = open(opts.outDir + fname, 'w')
fout.write('%g %g\n'%(freq, h_up))

fout.close()
