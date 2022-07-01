#!/usr/bin/python3
#
import numpy as np
import matplotlib.pyplot as plt

import scipy.special as specialf
import scipy.optimize as optimize

# P (prob.) of a delta chi^2 from chi^2 minimum: = gammainc(nParam/2, deltaChi/2)
# where nParam is the number of parameter's whose joint confidence region you want; and deltaChi is the change in chi^2 from the minimum.
# gammainc = regularized lower incomplete Gamma function: 
# = 1 / gamma(a) * integral(exp(-t) * t**(a-1), t=0..x) 
# (rearranged for root finding with a given P)
def funcProbDeltaChi2(deltaChi, nParam, targetP):
    return specialf.gammainc(nParam/2., deltaChi/2.) - targetP



incl, iterFit, entropy, chi2, test = np.loadtxt('outGridInc.dat', usecols=(1,3,5,7,9), unpack=True)

chiMaxPlt = 1.5*np.min(chi2)
plt.subplot(3,1,1)
plt.title('chi2')
plt.plot(incl, chi2) #, max=chiMaxPlt)
plt.axis(ymin=min(chi2), ymax=chiMaxPlt)

plt.subplot(3,1,2)
plt.title('entropy')
plt.plot(incl, entropy)
plt.axis(ymin=2.*np.max(entropy))

plt.subplot(3,1,3)
plt.title('test')
plt.plot(incl, test)
plt.axis(ymax=0.5)

plt.show()


################
try:
    import core.mainFuncs as mf
    import core.readObs as readObs
except ImportError:
    #If this is run from the utils sub-directory,
    #try adding the path to the main ZDIpy directory,
    #containing the core sub-directory/module.
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.mainFuncs as mf
    import core.readObs as readObs

#Count the number of observed datapoints used in ZDI
par = mf.readParamsZDI('inzdi.dat')

obsSet = readObs.obsProfSetInRange(par.fnames, par.velStart, par.velEnd, par.velRs)
nPointsTot = 0
for obs in obsSet:
    nPointsTot += obs.wl.shape[0]


targetProb = 0.9973 #the probability level you want to enclose.  Common choices are: 0.6827, 0.90, 0.9545, 0.99, 0.9973, 0.9999
#targetProb = 0.6827 #the probability level you want to enclose.  Common choices are: 0.6827, 0.90, 0.9545, 0.99, 0.9973, 0.9999
nParam = 1 #the number of fit parameters whose joint confidence region you want
nModelLines = nPointsTot #the number of degrees of freedom (often the number of datapoints - number of fit parameters, but for ZDI usually just number of points fit) used for the reduced chi^2
chi2nuMin = np.min(chi2) #the best achieved reduced chi^2
indexMin = np.argmin(chi2)
print('chi^2 min: {:} at {:}'.format(chi2nuMin, np.ravel(incl)[indexMin]))

#range limits for root finding of chi^2 probability function
rangeStart = 1e-3
rangeEnd = 1e9

#P value, roughly probability that the data disagree with the model
# P = regularized lower incomplete Gamma function: 
# = 1 / gamma(a) * integral(exp(-t) * t**(a-1), t=0..x) 
# where a = ndof; t = chi2
approxDOF = nModelLines
chi2Min = chi2nuMin*approxDOF
probOfMin = specialf.gammainc(approxDOF/2., chi2Min/2.)
print('prob. of dissagreement for chi^2 min: {:}'.format(probOfMin))

targetProbList = [0.6827, 0.9545, 0.9973]
for targetProb in targetProbList:
    # P (prob.) of a delta chi^2 from chi^2 minimum:
    # P = gammainc(nParam/2, deltaChi/2) # = 1 - gammaincc(nParam/2, deltaChi/2)
    # where nParam is the number of parameter's whose joint confidence region you want; and deltaChi is the change in chi^2 from the minimum.
    # To find the delta chi^2 for a given confidence level (probability), we invert this
    # and fund the root of the equation using brentq (faster than bisection). 
    rootDeltaChi = optimize.brentq(funcProbDeltaChi2, rangeStart, rangeEnd, args=(nParam,targetProb) )
    #print('Delta chi^2 {:} chi^2 {:} reduced chi^2 {:}'.format(rootDeltaChi, chi2nuMin*nModelLines + rootDeltaChi, (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines-nParam)))
    #Or to be consistent with how reduced chi^2 was calculated by ZDI:
    print(' Delta chi^2 {:} chi^2 {:} reduced chi^2 {:} for prob {:}'.format(rootDeltaChi, chi2nuMin*nModelLines + rootDeltaChi, (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines), targetProb))
    
    incBest = incl[np.argmin(chi2)]
    inChiRange = (chi2 <= (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines))
    print('inc {:} {:+}/{:+} (for p {:})'.format(incBest, np.max(incl[inChiRange]) - incBest, np.min(incl[inChiRange]) - incBest, targetProb))

chi2DiffArray = chi2*approxDOF-chi2Min
probArray = specialf.gammainc(approxDOF/2., chi2*nModelLines/2.)
probOfDiff = specialf.gammainc(nParam/2., chi2DiffArray/2.)

plt.subplot(3,1,1)
plt.title('chi2_nu')
plt.plot(incl, chi2)
plt.axis(ymin=min(chi2), ymax=max(chi2))
plt.xlabel('inclination (deg)')
plt.ylabel('chi^2_nu')

#chiMaxPlt = (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines-nParam)
#Or to be consistent with how reduced chi^2 was calculated by ZDI:
chiMaxPlt = (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines)  
plt.subplot(3,1,2)
plt.title('chi2_nu')
plt.plot(incl, chi2)
plt.axis(ymin=min(chi2), ymax=chiMaxPlt)
plt.xlabel('inclination (deg)')
plt.ylabel('chi^2_nu')
 
plt.subplot(3,1,3)
plt.title('realative prob.')
plt.plot(incl, probOfDiff)
plt.xlabel('inclination (deg)')
plt.ylabel('prob.')

plt.tight_layout()
plt.show()
