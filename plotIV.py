#!/usr/bin/python3
#
# Plots observed and ZDI model Stokes I profiles, then Stokes V profiles.
# Profiles are shifted vertically according to rotational phase

# File I/O definitions
#
def readLineModels(fileName):
    fModels = open(fileName)

    #Get the number of rotational phases, and the rotational cycle of each model
    nPhases = 0
    cycles = np.array([])
    for line in fModels:
        if (line.strip()!= ''):
            if (line.strip()[0] == '#'):
                nPhases += 1
                cycles = np.append(cycles, float(line.split()[-1]))
    fModels.seek(0)

    #Read in the set of ZDI model line profiles
    synWlSet = []
    synISet = []
    synVSet = []
    synWl = np.array([])
    synI = np.array([])
    synV = np.array([])
    flagGap = 0
    for line in fModels:
        if (line.strip() != ''): #use blank lines to separate sets
            if (line.strip()[0] != '#'):
                synWl = np.append(synWl, float(line.split()[0]))
                synI = np.append(synI, float(line.split()[1]))
                synV = np.append(synV, float(line.split()[2]))
                flagGap = 0
        else:
            #if we have just entered a blank line (or set of blank lines, 
            # save the old set and start an new set.
            if (flagGap == 0):
                synWlSet += [synWl]
                synISet += [synI]
                synVSet += [synV]
                synWl = np.array([])
                synI = np.array([])
                synV = np.array([])
            flagGap = 1
    fModels.close()

    return synWlSet, synISet, synVSet, cycles, nPhases
######################################################

def readLineObserved(fileName):
    #Read in the set of observed line profiles
    fObserved = open(fileName)
    
    obsWlSet = []
    obsISet = []
    obsIsigSet = []
    obsVSet = []
    obsVsigSet = []
    obsWl = np.array([])
    obsI = np.array([])
    obsIsig = np.array([])
    obsV = np.array([])
    obsVsig = np.array([])
    for line in fObserved:
        if (line.strip() != ''): #use blank lines to separate sets
            obsWl = np.append(obsWl, float(line.split()[0]))
            obsI = np.append(obsI, float(line.split()[1]))
            obsIsig = np.append(obsIsig, float(line.split()[2]))
            obsV = np.append(obsV, float(line.split()[3]))
            obsVsig = np.append(obsVsig, float(line.split()[4]))
        else:
            obsWlSet += [obsWl]
            obsISet += [obsI]
            obsIsigSet += [obsIsig]
            obsVSet += [obsV]
            obsVsigSet += [obsVsig]
            obsWl = np.array([])
            obsI = np.array([])
            obsIsig = np.array([])
            obsV = np.array([])
            obsVsig = np.array([])
    fObserved.close()

    return obsWlSet, obsISet, obsIsigSet, obsVSet, obsVsigSet
######################################################

#Main body

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

synWlSet, synISet, synVSet, cycles, nPhases = readLineModels('outLineModels.dat')

obsWlSet, obsISet, obsIsigSet, obsVSet, obsVsigSet = readLineObserved('outObserved.dat')

#Find profile vertical shifts, for stacking when plotting
maxObsI = 0.
minObsI = 1.
maxObsV = 0.
minObsV = 0.
meanRangeI = 0.0
for i in range (0, nPhases):
    if (obsISet[i].max() > maxObsI):
        maxObsI = obsISet[i].max()
    if (obsISet[i].min() < minObsI):
        minObsI = obsISet[i].min()
    if (obsVSet[i].max() > maxObsV):
        maxObsV = obsVSet[i].max()
    if (obsVSet[i].min() < minObsV):
        minObsV = obsVSet[i].min()
    meanRangeI += obsISet[i].max()-obsISet[i].min()
meanRangeI = meanRangeI/nPhases
rangeObsI = maxObsI-minObsI
rangeObsV = maxObsV-minObsV
#assume a mean vertical spacing that is a fraction of rangeObsX.
shiftVertI = rangeObsI*0.4*float(nPhases)
shiftVertV = rangeObsV*0.6*float(nPhases)

##Plot Stokes I profiles
#fig = plt.figure(figsize=(10., 10.))
#for i in range (0, nPhases):
#    phase = cycles[i]-float(int(cycles[i]))
#    if (phase < 0.): phase += 1.
#    obsIShift = obsISet[i] - shiftVertI*phase
#    synIShift = synISet[i] - shiftVertI*phase
#    contIShift = np.ones(len(synIShift)) - shiftVertI*phase
#    plt.plot(synWlSet[i], contIShift, 'b--') #continuum level
#    plt.errorbar(obsWlSet[i], obsIShift, yerr=obsIsigSet[i], fmt='k.')
#    plt.plot(synWlSet[i], synIShift, 'r-')
#    plt.text(synWlSet[i][-1], synIShift[-1], '{:6.3f}'.format(cycles[i]))
#plt.xlabel('Velocity (km/s)')
#plt.ylabel('I/Ic')
#plt.show()

#Plot Stokes I profiles
#version 2, for more reliable multicolumn plots
fig = plt.figure(figsize=(8., 10.))
maxColums=2

axList = []
for j in range(maxColums):
    phaseMin = 1.0/maxColums*j
    phaseMax = 1.0/maxColums*(j+1)
    ymax = -100.0
    ymin = 100.0
    
    ax = plt.subplot(1, maxColums, j+1)
    axList += [ax]
    for i in range (0, nPhases):
        phase = cycles[i]-float(int(cycles[i]))
        if (phase < 0.): phase += 1.
        #If this profile is in the phase range for this panel, plot it.
        if (phase >=  phaseMin and phase < phaseMax):
            obsIShift = obsISet[i] - shiftVertI*phase
            synIShift = synISet[i] - shiftVertI*phase
            contIShift = np.ones(len(synIShift)) - shiftVertI*phase
            textShift = (synWlSet[i][-1] - synWlSet[i][0])*0.06
            
            plt.plot(synWlSet[i], contIShift, 'b--') #continuum level
            plt.errorbar(obsWlSet[i], obsIShift, yerr=obsIsigSet[i], fmt='k.')
            plt.plot(synWlSet[i], synIShift, 'r-')
            plt.text(synWlSet[i][-1]+textShift, synIShift[-1], '{:6.3f}'.format(cycles[i]))
            ymax = max(ymax, obsIShift.max() + 2*obsIsigSet[i].max(),
                         synIShift.max() + 1*obsIsigSet[i].max(),
                         contIShift.max() + 1*obsIsigSet[i].max())
            ymin = min(ymin, obsIShift.min() - 2*obsIsigSet[i].max(),
                        synIShift.min() - 1*obsIsigSet[i].max())
    #Optionally add extra spacing to indicate missing phases
    ymax = max(ymax, 1.0-shiftVertI*phaseMin)
    ymin = min(ymin, 1.0-shiftVertI*phaseMax-meanRangeI)
    #also optionally leaving slightly more space for outliers
    #ymax = max(ymax, maxObsI-shiftVertI*phaseMin)
    #ymin = min(ymin, minObsI-shiftVertI*phaseMax)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin*1.1, xmax+4.*textShift)
    plt.xlabel('Velocity (km/s)')
    if (j == 0):
        plt.ylabel('I/Ic')
    #add a small margin for text labels etc.
    ymax = ymax + (ymax-ymin)*0.02
    ymin = ymin - (ymax-ymin)*0.01
    plt.ylim(ymin, ymax)
    
plt.tight_layout()
#plt.savefig('fig-I.eps')
plt.show()


#Plot Stokes V profiles
fig = plt.figure(figsize=(5., 10.))
maxplt = -1.
minplt = 0.
for i in range (0, nPhases):
    phase = cycles[i]-float(int(cycles[i]))
    if (phase < 0.): phase += 1.
    obsVShift = obsVSet[i] - shiftVertV*phase
    synVShift = synVSet[i] - shiftVertV*phase
    contVShift = np.zeros(len(synVShift)) - shiftVertV*phase
    plt.plot(synWlSet[i], contVShift, 'b--') #continuum level
    plt.errorbar(obsWlSet[i], obsVShift, yerr=obsVsigSet[i], fmt='k.')
    plt.plot(synWlSet[i], synVShift, 'r-')
    textShift = (synWlSet[i][-1] - synWlSet[i][0])*0.06
    plt.text(synWlSet[i][-1]+textShift, synVShift[-1], '{:6.3f}'.format(cycles[i]))
    if(np.max(obsVShift) > maxplt):
        maxplt = np.max(obsVShift)
    if(np.min(obsVShift) < minplt):
        minplt = np.min(obsVShift)
                                            
xmin, xmax = plt.xlim()
plt.xlim(xmin*1.1, xmax+4.*textShift)
plt.xlabel('Velocity (km/s)')
ypad = (minplt-maxplt)*0.05
plt.ylim(minplt+ypad, maxplt-ypad)
plt.ylabel('V/Ic')
plt.tight_layout()
#plt.savefig('fig-V.eps')
plt.show()
