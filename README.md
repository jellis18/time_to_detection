# Time to Detection Simulations

This repository contains scripts for running time-do-detection simulations
for both continuous gravitational waves (CWs) and a stochastic background
of gravitational waves (GWB).

The codes make use of the frequentist [Fe-statistic](http://adsabs.harvard.edu/abs/2012ApJ...756..175E) and
[Optimal Statistic (OS)](http://adsabs.harvard.edu/abs/2015PhRvD..91d4048C).


## Dependencies

1. numpy
2. scipy
3. matplotlib
4. numexpr

## Usage

These codes make use of somewhat idealized data in order to make time-do-detection
estimates based on current data and predictions for future data.

Both `projection_cw.py` and `projection_gwb.py` have very similar call
structures.

### Base PTA settings

All data sets are based on current NANOGrav data with additions from the IPTA
for pulsars that are not observed by NANOGrav (e.g. J0437-4715). All data sets
start in 2005 and are based on observed noise values and cadences up to 2015.
After 2015, the user has 4 options defined by the following command line options

* `--bestcase`: Massively increased telescope time used to add more pulsars and
time currently observed pulsars at higher cadence.
* `--cwcase`: Same massively increased telescope time as in `bestcase` but now
used to spend that extra time on beating down noise in best timed pulsars.
* `--statuscase`: Keep the status quo. Add two pulsars per year at both GBT
and AO. No additional observing time.
* `--worstcase`: Lose AO in 2017 and move all AO pulsars to GBT at cost of
observing time. Each pulsar gets less observing time and AO pulsars have poorer
timing precision at GBT.

Given a base data set we have a few more options:

* `--tspan`: Total PTA span in years. Max time is 25 years (ending in 2030).
* `--cadence`: Observations per year. Default is 20 which corresponds to
roughly bi-monthly observations.


We do not include a full timing model to be marginalized. The base model
always used contains the quadratic spin-down component which is most important
for simulating the timing model effects on GWs. However we do have two other
options to add more realism.

* `--RADEC`: Include sky location parameters in timing model.
* `--PX`: Include parallax parameter in timing model.

### Noise and GWB settings

The user has control over red noise and GWB strength but the white noise values
are fixed based on the observed values in current PTAs. At the moment, red noise
is applied uniformly to all pulsars. However, the user does have the option to
include the red noise values measured in the following [IPTA](http://adsabs.harvard.edu/abs/2016MNRAS.458.2161L)
and [NANOGrav](http://adsabs.harvard.edu/abs/2015ApJ...813...65T) papers.
The following command line options allow the user to specify these noise values
(note that both GWB and red noise is defined by a power-law):

* `--Agw`: GW amplitude in standard units (e.g. 1e-15)
* `--gamGW`: GW spectral index (e.g. 4.33)
* `--Ared`: Red noise amplitude in standard units (e.g. 1e-15)
* `--gamred`: Red noise spectral index (e.g. 4.33)
* `--redFromFile`: Read red noise values from the aforementioned papers. All
other pulsars will use the user defined values `Ared` and `gammared`.

### Simulation settings

After the PTA and noise settings are taken care of we have some basic
simulation options.

* `--nreal`: Number of realizations to run for upper limit and detection
probability calculations.
* `--seed`: Specify random number seed for PTA creation. New pulsars are drawn
from random distributions. This option allows the user to be sure that the same
PTA is used in each run.
* `--label`: Label for output if running multiple copies on a cluster. Only
for `projection_gwb.py`
* `--outDir`: Output directory for output files.
* `--detprob`: Detection probability to use in computation of minimum
detectable amplitude. Only for `projection_cw.py`.
* `--fap`: False Alarm Probability to use in computation of minimum
detectable amplitude. Only for `projection_cw.py`.
* `--freq`: GW frequency at which to compute minimum detectable amplitude.
Only for `projection_cw.py`.


### Outputs

The CW and GWB time-do-detection codes do have different outputs that we
describe here.

When `projection_cw.py` is given a `tspan` it only simulated data for that
timespan and then computes the minimum detectable amplitude for a given
input frequency `freq`.  The output file is `upper_FREQ_NREAL.txt` where
FREQ and NREAL are the GW frequency and number of realizations, respectively.
The output file only contains the GW frequency and minimum detectable strain
amplitude.

When `projection_gwb.py` is given a `tspan`. It computes the OS distribution (over noise and GWB realizations) for each year starting at 2005 and ending
in 2005 + `tspan`. The output files here are `numpy` arrays `snr_data_AGW_GAMMAGW_ARED_GAMMARED_LABEL.npy`  and
`opt_data_AGW_GAMMAGW_ARED_GAMMARED_LABEL.npy` where AGW, GAMMAGW, ARED,
GAMMARED and LABEL are the GWB amplitude, GWB spectral index, Red Noise Amplitude, Red Noise spectral index, and label respectively. The output arrays
are `tspan` x `nreal` where each row is the year starting with 2005 and the
columns are the values of the SNR or the OS for each realization.
