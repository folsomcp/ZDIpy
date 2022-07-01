#!/usr/bin/python3
#
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['ps.useafm'] = True
#plt.rcParams['text.usetex'] = True

import scipy.special as specialf
import scipy.optimize as optimize

# P (prob.) of a delta chi^2 from chi^2 minimum: = gammainc(nParam/2, deltaChi/2)
# where nParam is the number of parameter's whose joint confidence region you want; and deltaChi is the change in chi^2 from the minimum.
# gammainc = regularized lower incomplete Gamma function: 
# = 1 / gamma(a) * integral(exp(-t) * t**(a-1), t=0..x) 
# (rearranged for root finding with a given P)
def funcProbDeltaChi2(deltaChi, nParam, targetP):
    return specialf.gammainc(nParam/2., deltaChi/2.) - targetP



period, dOmega, iterFit, entropy, chi2, test = np.loadtxt('outGridPdO.dat', usecols=(1,3,5,7,9,11), unpack=True)

nRows = 0
while(period[nRows] == period[0]):
    nRows += 1
nColumns = period.shape[0]//nRows
if ((float(period.shape[0])/float(nColumns) % 1.) != 0):
    print('ERROR could not get array shapes {:} {:}'.format(period.shape, nColumns, nRows))
#or used: divmod(float(period.shape[0]), float(nColumns))

period = np.reshape(period, (nColumns, nRows))
dOmega = np.reshape(dOmega, (nColumns, nRows))
iterFit = np.reshape(iterFit, (nColumns, nRows))
entropy = np.reshape(entropy, (nColumns, nRows))
chi2 = np.reshape(chi2, (nColumns, nRows))
test = np.reshape(test, (nColumns, nRows))

periodCorn = np.zeros((nColumns+1, nRows+1))
dOmegaCorn = np.zeros((nColumns+1, nRows+1))
for i in range(nColumns+1):
    for j in range(nRows+1):
        if(i == 0):
            iL = i
            iR = i+1
            iO = i
        elif(i == nColumns):
            iL = i-1
            iR = i-2
            iO = i-1
        else:
            iL = i-1
            iR = i
            iO = i
        if(j == 0):
            jL = j
            jR = j+1
            jO = j
        elif(j == nRows):
            jL = j-1
            jR = j-2
            jO = j-1
        else:
            jL = j-1
            jR = j
            jO = j
             
        periodCorn[i,j] = period[iO,jO] - (period[iR,jR] - period[iL,jL])/2.
        dOmegaCorn[i,j] = dOmega[iO,jO] - (dOmega[iR,jR] - dOmega[iL,jL])/2.


pltCorners = (periodCorn[0,0], periodCorn[-1,-1], dOmegaCorn[0,0], dOmegaCorn[-1,-1])


################

#Count the number of observed datapoints used in ZDI
try:
    import core.mainFuncs as mf
    import core.readObs as readObs
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.mainFuncs as mf
    import core.readObs as readObs

par = mf.readParamsZDI('inzdi.dat')

obsSet = readObs.obsProfSetInRange(par.fnames, par.velStart, par.velEnd, par.velRs)
nPointsTot = 0
for obs in obsSet:
    nPointsTot += obs.wl.shape[0]
### Done counting

targetProb = 0.9973 #the probability level you want to enclose.  Common choices are: 0.6827, 0.90, 0.9545, 0.99, 0.9973, 0.9999
nParam = 2 #the number of fit parameters whose joint confidence region you want
nModelLines = nPointsTot #the number of degrees of freedom (often the number of datapoints - number of fit parameters, but for ZDI usually just number of points fit) used for the reduced chi^2
chi2nuMin = np.min(chi2) #the best achieved reduced chi^2
indexMin = np.argmin(chi2)
print('chi^2 min: {:} at P ={:} dO ={:}'.format(chi2nuMin, np.ravel(period)[indexMin], np.ravel(dOmega)[indexMin]))

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

# P (prob.) of a delta chi^2 from chi^2 minimum:
# P = gammainc(nParam/2, deltaChi/2) # = 1 - gammaincc(nParam/2, deltaChi/2)
# where nParam is the number of parameter's whose joint confidence region you want; and deltaChi is the change in chi^2 from the minimum.
# To find the delta chi^2 for a given confidence level (probability), we invert this
# and fund the root of the equation using brentq (faster than bisection). 
rootDeltaChi = optimize.brentq(funcProbDeltaChi2, rangeStart, rangeEnd, args=(nParam,targetProb) )
#print('Delta chi^2 {:} chi^2 {:} reduced chi^2 {:}'.format(rootDeltaChi, chi2nuMin*nModelLines + rootDeltaChi, (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines-nParam)))
#for a consistent deffinition of reduced chi^2 with the ZDI code
print('Delta chi^2 {:} chi^2 {:} reduced chi^2 {:}'.format(rootDeltaChi, chi2nuMin*nModelLines + rootDeltaChi, (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines)))

chi2DiffArray = chi2*nModelLines-chi2Min
probArray = specialf.gammainc(approxDOF/2., chi2*nModelLines/2.)
probOfDiff = specialf.gammainc(nParam/2., chi2DiffArray/2.)

#chiMaxPlt = (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines-nParam)
#for a consistent deffinition of reduced chi^2 with the ZDI code
chiMaxPlt = (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines)

#Get the chi^2 values for a set of probabilities.  For making a chi^2 contour of a given probability.
#probContours = [0.6827, 0.9545, 0.9973, 0.9999]
probContours = [0.6827, 0.9545, 0.9973]
probContoursChi = []
for probI in probContours:
    rootDeltaChi = optimize.brentq(funcProbDeltaChi2, rangeStart, rangeEnd, args=(nParam,probI) )
    #chi2nuI = (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines-nParam)
    #for a consistent deffinition of reduced chi^2 with the ZDI code
    chi2nuI = (chi2nuMin*nModelLines + rootDeltaChi)/(nModelLines)
    probContoursChi += [chi2nuI]


freq = 2*np.pi/period
freqCorn = 2.*np.pi/periodCorn
#pltFreqCorners = 2.*np.pi/pltCorners

plt.subplot(1,1,1)
#plt.title('chi2_nu')
plt.pcolormesh(freqCorn, dOmegaCorn, chi2, vmax=chiMaxPlt, rasterized=True, cmap='YlOrBr_r') #afmhot YlOrBr_r Oranges_r
#plt.axis(pltFreqCorners)
cbar = plt.colorbar()
#cbar.set_label('$\chi^2$', fontsize=16)
cbar.ax.tick_params(labelsize=14)

clables=['1$\sigma$', '2$\sigma$', '3$\sigma$']
clabdic = {probContoursChi[0]: '1$\sigma$', probContoursChi[1]: '2$\sigma$', probContoursChi[2]: '3$\sigma$'}

CS = plt.contour(freq, dOmega, chi2, probContoursChi, colors='k') #colors=('k','k','k','k')
#Some optional code for forcing contour label positions:
#plt.clabel(CS, fmt=clabdic, fontsize=14)
#lplaces = [(9.39, 0.11), (9.412, 0.119), (9.44, 0.13)]
#lplaces = [(2.*np.pi/9.39, 0.11), (2.*np.pi/9.412, 0.119), (2.*np.pi/9.44, 0.13)]
#plt.clabel(CS, manual=lplaces, fmt=clabdic, fontsize=18)
#Or with automatic contour label positions:
plt.clabel(CS, fmt=clabdic, fontsize=18)
plt.xlabel('$\Omega_{eq}$ (rad/d)', fontsize=18)
plt.ylabel('d$\Omega$ (rad/d)', fontsize=18)

plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()

plt.savefig('plot-p-dOmega-chi2.eps')
plt.show()
