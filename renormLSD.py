#!/usr/bin/python3
#
#Renormalize LSD profiles continuum based on user specified continuum ranges.  
#Useful particularly when profiles were based on imperfectly normalized spectra.  
#Then rescale them all to the same equivalent width 
#(equal to the mean equivalent width), 
#Good if, e.g., varying amounts of scattered light/sky background 
#added a bit of continuum dilution at some phases

import numpy as np
import matplotlib.pyplot as plt
import core.readObs as readObs
import core.lineprofileVoigt as lineprofile
c = 2.99792458e5


# read input from inrenorm.dat
# used to get input file list, continuum ranges, ...
fInZDI = open('inrenorm.dat','r')
fnames = np.array([])
velRs = np.array([])
numObs = 0
i=0
for inLine in fInZDI:
    if(inLine.strip() == ''):  #skip blank lines
        continue
    #check for comments (ignoring white-space)
    if (inLine.strip()[0] != '#'):
        if (i==0):
            fNormContin = int(inLine.split()[0])
        elif(i==1):
            velLineStart = float(inLine.split()[0])
            velLineEnd = float(inLine.split()[1])
        elif(i==2):
            velProfStart = float(inLine.split()[0])
            velProfEnd = float(inLine.split()[1])
        elif(i==3):
            polyOrder = float(inLine.split()[0])
        elif(i==4):
            fSaveContin = int(inLine.split()[0])
        elif(i==5):
            fPlotContin = int(inLine.split()[0])
        elif(i==6):
            fNormEquivWdth = int(inLine.split()[0])
        elif(i==7):
            velEWStart = float(inLine.split()[0])
            velEWEnd = float(inLine.split()[1])
        elif(i >= 8):
            fnames = np.append(fnames, [inLine.split()[0]])
            velRs = np.append(velRs, [float(inLine.split()[1])])
            numObs += 1

        i += 1

#read the observed line profiles
obsSet = readObs.obsProfSet(fnames)

#set the outside edges of the LSD profile to use, if not specified
flagAutoEdges = 0
if ( (np.abs(velProfStart) <= 1e-6) & (np.abs(velProfEnd) <= 1e-6) ):
    flagAutoEdges = 1
    velBufferOutter = 2.0  #km/s

#Renormalize the continuum level
if(fNormContin == 1):
    nObs = 0
    for obs in obsSet:
        shiftWl = obs.wl - velRs[nObs]

        #set the outside edges of the LSD profile to use, if not specified
        if(flagAutoEdges == 1):
            velProfStart = shiftWl[0] + velBufferOutter
            velProfEnd = shiftWl[-1] - velBufferOutter

        #Get points in the desired velocity range
        #(this could be done with a loop and if statements, 
        # but this is more compact)
        iFitWl = np.where( ((shiftWl >= velProfStart)&(shiftWl <= velLineStart)) | ((shiftWl >= velLineEnd)&(shiftWl <= velProfEnd)) )
        #save ranges of the LSD profile to be used in the fit
        fitWl = shiftWl[iFitWl]
        fitI = obs.specI[iFitWl]
        fitIsig = obs.specIsig[iFitWl]

        fitCoeff = np.polyfit(fitWl, fitI, polyOrder)
        fitPoly = np.poly1d(fitCoeff)
        polyVal = fitPoly(shiftWl)

        if (fSaveContin == 1):
            fOutContin = open(fnames[nObs].strip()+'.contin', 'w')
            for i in range(polyVal.shape[0]):
                fOutContin.write('{:f} {:e}\n'.format(obs.wl[i], polyVal[i]))
            fOutContin.close()
        if (fPlotContin == 1):
            plt.plot(shiftWl, obs.specI,'k')
            plt.plot(fitWl, fitI,'b.-')
            plt.plot(shiftWl, polyVal,'r')
            plt.show()

        obs.specI    = obs.specI/polyVal
        obs.specIsig = obs.specIsig/polyVal
        obs.specV    = obs.specV/polyVal
        obs.specVsig = obs.specVsig/polyVal
        obs.specN    = obs.specN/polyVal
        obs.specNsig = obs.specNsig/polyVal

        nObs += 1

        
if (fNormEquivWdth == 1):

    print('calculating equivalent with in range: ', velEWStart, velEWEnd)
    lineData = lineprofile.lineData('model-voigt-line.dat')

    print("observed equivalent widths:")
    nObs = 0
    meanEquivWid = 0.
    meanSigEW = 0.
    setEquivWid = np.zeros([0])
    setSigEW = np.zeros([0])
    for obs in obsSet:
        shiftWl = obs.wl - velRs[nObs]

        #equivWidApprox = (np.sum(1.-obs.specI))*(obs.wl[1]-obs.wl[0])/c*lineData.wl0[0]
        equivWid = 0.
        sigEW = 0.
        for i in range(shiftWl.shape[0]):
            if ((shiftWl[i] >= velEWStart) & (shiftWl[i] <= velEWEnd)):
                equivWid += (1.-obs.specI[i])*(obs.wl[i+1]-obs.wl[i])/c*lineData.wl0[0]
                sigEW += (obs.specIsig[i]*(obs.wl[i+1]-obs.wl[i])/c*lineData.wl0[0])**2
        setEquivWid = np.append(setEquivWid, equivWid)
        meanEquivWid += equivWid
        sigEW = np.sqrt(sigEW)
        setSigEW = np.append(setSigEW, sigEW)
        meanSigEW += sigEW
        print("{:f} +/- {:f}".format(equivWid, sigEW))

        nObs += 1
    meanEquivWid /= float(nObs)
    meanSigEW /= float(nObs)
    varEquivWid = np.sqrt(np.sum((setEquivWid-meanEquivWid)**2)/float(setEquivWid.shape[0]))
    print('Mean equivalent width:', meanEquivWid, " variance (stdev):", varEquivWid, ' Mean uncertainty on EW:', meanSigEW)

    print('rescaling by:')
    nObs = 0
    for obs in obsSet:
        scaleEW = meanEquivWid/setEquivWid[nObs]
        print("{:f}".format(scaleEW))

        obs.specI    = 1.-(1.-obs.specI)*scaleEW
        obs.specIsig = obs.specIsig*scaleEW
        obs.specV    = obs.specV*scaleEW
        obs.specVsig = obs.specVsig*scaleEW
        obs.specN    = obs.specN*scaleEW
        obs.specNsig = obs.specNsig*scaleEW
        nObs += 1

########################################################################
#Save as LSD profile format
nObs = 0
for obs in obsSet:
    fOutLDS = open(fnames[nObs].strip()+'.norm', 'w')
    
    fOutLDS.write('#renormalized LSD prof\n')
    fOutLDS.write('{:n} {:n} \n'.format(obs.wl.shape[0], 6))
    
    for i in range(obs.wl.shape[0]):
        fOutLDS.write('{:f} {:e} {:e} {:e} {:e} {:e} {:e}\n'.format(obs.wl[i],  obs.specI[i], obs.specIsig[i], obs.specV[i], obs.specVsig[i], obs.specN[i], obs.specNsig[i]) )
    fOutLDS.close()
    nObs += 1
#########################################################################

