import numpy as np
c = 2.99792458e5

class obsProf:
    def __init__(self, obsFileName):
        #Read an observed (LSD) profile from obsFileName, and return it as an obsProf object.  
        #If no file name is given, create an empty obsProf object
        if(obsFileName == ''):
            self.wl = np.array([])
            self.specI = np.array([])
            self.specIsig = np.array([])
            self.specV = np.array([])
            self.specVsig = np.array([])
            self.specN = np.array([])
            self.specNsig = np.array([])
        else:
            fObsFile = open(obsFileName, 'r')
            inLine = fObsFile.readline()
            inLine = fObsFile.readline()
            numColums = int(inLine.split()[1])
            fObsFile.close()
            #For Stokes I only (in Donati's LSD format)
            if (numColums == 2):
                self.wl, self.specI, self.specIsig \
                    = np.loadtxt(obsFileName, skiprows=2, usecols = (0,1,2), unpack=True)
                #include dummy arrays for V and N (should not be used)
                self.specV = np.zeros(self.wl.shape)
                self.specVsig = np.ones(self.wl.shape)*1e-6
                self.specN = np.zeros(self.wl.shape)
                self.specNsig = np.ones(self.wl.shape)*1e-6
            else:  #assume I, V, & N format
                self.wl, self.specI, self.specIsig, self.specV, self.specVsig, self.specN, self.specNsig \
                    = np.loadtxt(obsFileName, skiprows=2, unpack=True)
    
    def scaleIsig(self, chiScaleI):
        #Re-scale the Stokes I error bars, to incorporate the chi^2 scaling for I (in an indirect fashion).
        self.specIsig /= np.sqrt(chiScaleI)

                
def obsProfSet(obsFileNameList):
    #read in observation files from an array of file names
    #returns an array of obsProf objects
    obsSet = np.array([])
    for obsFileName in obsFileNameList:
        obsSet = np.append(obsSet, obsProf(obsFileName))

    return obsSet


def obsProfSetInRange(obsFileNameList, velStart, velEnd, velRs):
    #read in observation files from an array of file names
    #returns an array of obsProf object, 
    #restricted to be within velStart and velEnd of the central velRs
    obsSet = np.array([])
    nFile = 0 
    for obsFileName in obsFileNameList:
        obsTmp = obsProf(obsFileName)

        #find the portion of the observation in the desired velocity range
        iStart = -1
        iEnd = -1
        for i in range(obsTmp.wl.shape[0]):
            if( (obsTmp.wl[i] >= velStart + velRs[nFile]) & (obsTmp.wl[i] <= velEnd + velRs[nFile]) ):
                if(iStart < 0):
                    iStart = i
                iEnd = i
        #Save only the portion of the observation in range.
        obsTmp.wl       = obsTmp.wl[iStart:iEnd+1]
        obsTmp.specI    = obsTmp.specI[iStart:iEnd+1]
        obsTmp.specIsig = obsTmp.specIsig[iStart:iEnd+1]
        obsTmp.specV    = obsTmp.specV[iStart:iEnd+1]
        obsTmp.specVsig = obsTmp.specVsig[iStart:iEnd+1]
        obsTmp.specN    = obsTmp.specN[iStart:iEnd+1]
        obsTmp.specNsig = obsTmp.specNsig[iStart:iEnd+1]
        
        obsSet = np.append(obsSet, obsTmp)
        nFile += 1

    return obsSet


def getObservedEW(obsSet, lineData, verbose=1):
    #Calculate equivalent widths from a set of observations
    #assuming the continuum is at 1.
    meanEquivWidObs = 0.
    nObs = 0
    if(verbose > 1):
        print('observed equivalent widths')
    for obs in obsSet:
        equivWidApprox = 0.
        for i in range(obs.wl.shape[0]-1):
            equivWidApprox += (1.-obs.specI[i])*(obs.wl[i+1]-obs.wl[i])/c*lineData.wl0[0]
        equivWidApprox += (1.-obs.specI[-1])*(obs.wl[-1]-obs.wl[-2])/c*lineData.wl0[0]
        meanEquivWidObs += equivWidApprox
        nObs += 1
        if(verbose > 1):
            print("{:f} ".format(equivWidApprox))
    meanEquivWidObs /= float(nObs)
    if(verbose == 1):
        print('Mean observed equivalent width {:}'.format(meanEquivWidObs))

    return meanEquivWidObs
            


