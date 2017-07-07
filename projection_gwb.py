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
parser.add_option('--RADEC',action='store_true',dest='RADEC',help='fit for RA and DEC',default=False)
parser.add_option('--PX',action='store_true',dest='PX',help='fit for PX',default=False)
parser.add_option('--DMX',action='store_true',dest='DMX',help='fit for DMX',default=False)
parser.add_option('--f0',action='store',dest='f0', type='float',help='frequency for spectral break',default=0)
parser.add_option('--beta',action='store',dest='beta', type='float',help='slope of hc after spectral break',default=1)
parser.add_option('--power',action='store',dest='power', type='float',help='power to make turnover steeper',default=1)
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

# parse arguments
(opts,args)=parser.parse_args()

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


#### DEFINE UPPER LIMIT FUNCTION #####
def upperLimitFunc(A_gw, nreal):
    """
    Compute the value of the Optimal Statistic for different signal realizations

    @param A: value of GWB amplitude
    @param nreal: number of realizations

    """

    # creat inverse covariance matrices
    npsr = len(psr)
    for ct, p in enumerate(psr):

        # matrix of time lags
        tm = PALutils.createTimeLags(p.toas, p.toas)

        # GW
        Cgw = PALutils.createRedNoiseCovarianceMatrix(tm.copy(), A_gw, gam_gw)

        # white noise
        Cnoise = PALutils.createWhiteNoiseCovarianceMatrix(p.err, 1, 0)

        # red noise
        if p.Ared:
            Cnoise += PALutils.createRedNoiseCovarianceMatrix(tm.copy(), p.Ared, p.gred)


        # cholesky decomposition of noise
        u, s, v = np.linalg.svd(Cnoise)
        p.Cnoise12 = u * np.sqrt(s)

        # combine
        C = Cgw + Cnoise

        # inverse
        tmp = np.dot(G[ct].T, np.dot(C, G[ct]))

        p.invCov = np.dot(G[ct], np.dot(np.linalg.inv(tmp), G[ct].T))


    # cross covariances
    SIJ = []
    k = 0
    ORF = PALutils.computeORF(psr)
    for ii in range(npsr):
        for jj in range(ii+1, npsr):

            # matrix of time lags
            tm = PALutils.createTimeLags(psr[ii].toas, psr[jj].toas)

            # cross covariance matrix
            SIJ.append(ORF[k]/2 * PALutils.createRedNoiseCovarianceMatrix(tm.copy(), 1, gam_gw))

            # increment
            k += 1


    # precompute denominator
    k = 0
    norm = []
    opt_filter = []
    for ii in range(npsr):
        for jj in range(ii+1, npsr):
            tmp = np.dot(SIJ[k], psr[jj].invCov)
            filter = np.dot(psr[ii].invCov, tmp)
            tmp = np.dot(filter, SIJ[k].T)
            norm.append(np.trace(tmp))
            opt_filter.append(filter)
            k += 1


    count = 0
    rho = []
    OS = []
    np.random.seed()
    for ii in range(nreal):

        #tstart = time.time()

        # create residuals
        turnover=False
        if opts.f0:
            turnover=True

        ntoa = np.max([len(p.toas) for p in psr])
        inducedRes = PALutils.createGWB_clean(psr, A_gw, gam_gw,
                                              turnover=turnover,
                                              f0=opts.f0,
                                              beta=opts.beta,
                                              power=opts.power,
                                              npts=ntoa)


        #print 'Making GWB and red residuals = {0} s'.format(time.time()-tstart)

        #tstart = time.time()

        # create residuals
        for ct,p in enumerate(psr):

            # make noise for each pulsar
            w = np.random.randn(len(p.toas))
            res = np.dot(p.Cnoise12, w)
            res += inducedRes[ct]

            # replace residuals in pulsar object
            p.res = np.dot(R[ct], res)

        #print 'Making full residuals = {0} s'.format(time.time()-tstart)

        #tstart = time.time()

        # construct optimal statstic
        k = 0
        top = 0
        bot = 0
        for ll in range(npsr):
            for kk in range(ll+1, npsr):

                # compute numerator of optimal statisic
                top += np.dot(psr[ll].res, np.dot(opt_filter[k], psr[kk].res))

                # compute trace term
                bot += norm[k]

                # iterate counter
                k += 1


        # get optimal statistic and SNR
        optStat = top/bot
        snr = top/np.sqrt(bot)
        OS.append(optStat)
        rho.append(snr)

        # check to see if larger than in real data
        if snr > 3:
            count += 1

        #print 'Computing OS = {0} s'.format(time.time()-tstart)

    # now get detection probability
    detProb = count/nreal


    return detProb, np.array(OS), np.array(rho)

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
    gbtfile = np.genfromtxt('gbt_pulsar_file.txt', dtype='S42')
    aofile = np.genfromtxt('ao_pulsar_file.txt', dtype='S42')
    gbtmed = np.loadtxt('gbt_med_rms_vals.txt')
elif opts.worstcase:
    gbtfile = np.genfromtxt('gbt_pulsar_file_worst.txt', dtype='S42')
    aofile = np.genfromtxt('ao_pulsar_file_worst.txt', dtype='S42')
    gbtmed = np.loadtxt('gbt_med_rms_vals_worst.txt')
elif opts.statuscase:
    gbtfile = np.genfromtxt('gbt_pulsar_file_status.txt', dtype='S42')
    aofile = np.genfromtxt('ao_pulsar_file_status.txt', dtype='S42')
    gbtmed = np.loadtxt('gbt_med_rms_vals_status.txt')
    aomed = np.loadtxt('ao_med_rms_vals_status.txt')
elif opts.cwcase:
    gbtfile = np.genfromtxt('gbt_pulsar_file_cw.txt', dtype='S42')
    aofile = np.genfromtxt('ao_pulsar_file_cw.txt', dtype='S42')
    gbtmed = np.loadtxt('gbt_med_rms_vals_cw.txt')

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
    red_names = np.loadtxt('red_pulsars.txt', dtype='S42', usecols=[0])
    red_vals = np.loadtxt('red_pulsars.txt', usecols=[1,2])
    red_dict = {}
    for rn, rv in zip(red_names, red_vals):
        red_dict[rn] = rv


# start loop over dates
dates = np.arange(2006, opts.tspan+2006)
Nt = len(dates)
SNR = np.zeros((Nt,Nreal))
up = np.zeros((Nt,Nreal))
optimal_stat = np.zeros((Nt,Nreal))

for ct1, date in enumerate(dates):

    # initialize pulsar class
    psr = []
    R = []

    # add new GBT pulsars
    if date >= 2016:
        for ii in range(2):
            np.random.seed(opts.seed*(ii+1)*int(date))
            theta = np.arccos(np.random.uniform(np.cos(2.37), np.cos(0)))
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

            R.append(RR)

            psr.append(pulsar(gb['sky'][0], gb['sky'][1], len(err),
                              ut*3.16e7, err*3.16e7, err*1e-6, M,
                              0, 1,freqs, gb['name']))

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

            R.append(RR)

            psr.append(pulsar(gb['sky'][0], gb['sky'][1], len(err),
                              ut*3.16e7, err*3.16e7, err*1e-6, M,
                              0, 0,freqs, gb['name']))


    # number of pulsars
    npsr = len(psr)

    # G matrices
    G = [PALutils.createGmatrix(p.designmatrix) for p in psr]

    # red noise
    if A_red:
        for pp in psr:
            pp.Ared = A_red
            pp.gred = gam_red

    if opts.redFromFile:
        for pp in psr:
            if pp.name in red_dict:
                pp.Ared = 10**red_dict[pp.name][0]
                pp.gred = red_dict[pp.name][1]
                print 'Setting Ared={}, gred={} for PSR {}'.format(pp.Ared, pp.gred, pp.name)

    # run simulations
    detProb, optimal_stat[ct1,:], SNR[ct1,:] = upperLimitFunc(A_gw, Nreal)
    sigma = optimal_stat[ct1,:]/SNR[ct1,:]
    up[ct1,:] = np.sqrt(optimal_stat[ct1,:] + np.sqrt(2)*sigma*ss.erfcinv(2*(1-0.95)))

    print date, npsr, A_gw, detProb


# output data
if not os.path.exists(opts.outDir):
    try:
        os.makedirs(opts.outDir)
    except OSError:
        pass


snrName=opts.outDir+'/snr_data_{0}_{1}_{2}_{3}_{4}.npy'.format(A_gw,gam_gw,A_red,gam_red,opts.label)
optName=opts.outDir+'/opt_data_{0}_{1}_{2}_{3}_{4}.npy'.format(A_gw,gam_gw,A_red,gam_red,opts.label)
np.save(snrName,SNR)
np.save(optName,optimal_stat)
