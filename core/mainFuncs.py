import numpy as np
c = 2.99792458e5

class readParamsZDI:
    #This holds the input model parameters for (Z)DI
    def __init__(self, inParamsName):
        #Read in the model and control parameters
        fInZDI = open(inParamsName,'r')
        self.fnames = np.array([])
        self.jDates = np.array([])
        self.velRs = np.array([])
        self.numObs = 0
        i=0
        for inLine in fInZDI:
            if(inLine.strip() == ''):  #skip blank lines
                continue
            #check for comments (ignoring white-space)
            if (inLine.strip()[0] != '#'):
                if (i==0):
                    self.inclination = float(inLine.split()[0])
                    self.vsini = float(inLine.split()[1])
                    self.period = float(inLine.split()[2])
                    self.dOmega = float(inLine.split()[3])
                elif(i==1):
                    self.mass = float(inLine.split()[0])
                    self.radius = float(inLine.split()[1])
                elif(i==2):
                    self.nRingsStellarGrid = int(inLine.split()[0])
                elif(i==3):
                    self.targetForm = inLine.split()[0]
                    self.targetValue = float(inLine.split()[1])
                    self.numIterations = int(inLine.split()[2])
                elif(i==4):
                    self.test_aim = float(inLine.split()[0])
                elif(i==5):
                    self.fitMag = int(inLine.split()[0])
                    self.lMax = int(inLine.split()[1])
                    self.defaultBent = float(inLine.split()[2])
                    self.magGeomType = inLine.split()[3]
                elif(i==6):
                    self.initMagFromFile = int(inLine.split()[0])
                    self.initMagGeomFile = inLine.split()[1]
                elif(i==7):
                    self.fitBri = int(inLine.split()[0])
                    self.chiScaleI = float(inLine.split()[1])
                    self.brightEntScale = float(inLine.split()[2])
                elif(i==8):
                    self.fEntropyBright = int(inLine.split()[0])
                    self.defaultBright = float(inLine.split()[1])
                    self.maximumBright = float(inLine.split()[2])
                elif(i==9):
                    self.initBrightFromFile = int(inLine.split()[0])
                    self.initBrightFile = inLine.split()[1]
                elif(i==10):
                    self.estimateStrenght = int(inLine.split()[0])
                elif(i==11):
                    self.instrumentRes = float(inLine.split()[0])
                elif(i==12):
                    self.velStart = float(inLine.split()[0])
                    self.velEnd = float(inLine.split()[1])
                elif(i==13):
                    self.jDateRef = float(inLine.split()[0])
                elif(i >= 14):
                    self.fnames = np.append(self.fnames, [inLine.split()[0]])
                    self.jDates = np.append(self.jDates, [float(inLine.split()[1])])
                    self.velRs = np.append(self.velRs, [float(inLine.split()[2])])
                    self.numObs += 1
                    if (np.abs(self.jDateRef-self.jDates[self.numObs-1]) > 500.):
                        print('Warning: possible miss-match between date and reference date {:} {:}'.format(self.jDateRef,  self.jDates[self.numObs-1]))
                    if (np.abs(self.velRs[self.numObs-1]) > 500.):
                        print('Warning: extreem Vr read:{:}'.format(self.velRs[self.numObs-1]))
        
                i += 1
        self.incRad = self.inclination/180.*np.pi
        self.velEq = self.vsini/np.sin(self.incRad)

        magGeomType = self.magGeomType.lower()
        if not(magGeomType == 'full' or magGeomType == 'poloidal'
               or magGeomType == 'pottor' or magGeomType == 'potential'):
            print(('ERROR: read an unrecognized magnetic geometry type ({:}).  '
                  +'Use one of: Full, Poloidal, PotTor, Potential').format(
                      self.magGeomType) )
            import sys
            sys.exit()
        self.magGeomType = magGeomType

    def calcCycles(self, verbose=1):
        #Calculate rotation cycle/phase from the period and Julian dates
        self.cycleList = (self.jDates - self.jDateRef)/self.period
        if ((self.dOmega != 0.) & (verbose == 1)):
            print('equator-pole lap time: {:8.4f} days, or {:8.4f} rotation cycles'.format(2.*np.pi/self.dOmega, (2.*np.pi/self.dOmega)/self.period) )
            print('    observations span: {:8.4f} days, or {:8.4f} rotation cycles'.format(np.max(self.jDates)-np.min(self.jDates), np.max(self.cycleList)-np.min(self.cycleList)) )

    def setTarget(self):
        #Check whether to fit to a target chi^2 or to a target entropy
        if(self.targetForm == 'C'):
            self.fixedEntropy = 0
            self.chiTarget = self.targetValue
            self.ent_aim = -1e6
        elif(self.targetForm == 'E'):
            self.fixedEntropy = 1
            self.ent_aim = self.targetValue
            self.chiTarget = 1.0
        else:
            print('ERROR unknown format for goodness of fit target: {:}'.format(self.targetForm))
            import sys
            sys.exit()


    def setCalcdIdV(self, verbose=1):
    #Check which set of line profile to fit and which derivatives needs to be calculated
        self.calcDI = 0
        self.calcDV = 0
        if (self.fitBri == 1):
            self.calcDI = 1
        elif((self.fitBri > 1) | (self.fitBri < 0)):
             print("ERROR: invalid value of fitBri: {:}".format(self.fitBri))
        if (self.fitMag == 1):
            self.calcDV = 1
        elif((self.fitMag > 1) | (self.fitMag < 0)):
            print("ERROR: invalid value of fitMag: {:}".format(self.fitMag))
        if ((self.calcDI == 0) & (self.calcDV == 0)):
            if (verbose==1):
                print("Warning: no parameters to fit!")
            self.numIterations = 0
            #import sys
            #sys.exit()
        
        if ((self.fEntropyBright != 1) & (self.fEntropyBright != 2)):
            print('error unrecognized brightness entropy flag: {:}'.format(self.fEntropyBright))
            import sys
            sys.exit()


def getWavelengthGrid(velRs, obsSet, lineData, verbose=1):
    #Generate wavelength grid for synthesis, using the observed grid (from velocity for LSD)
    #in stellar rest frame
    wlSynSet = []
    nObs = 0
    nDataTot = 0
    for obs in obsSet:
        tmpWlSyn = (obs.wl - velRs[nObs])/c*lineData.wl0 + lineData.wl0
        wlSynSet += [tmpWlSyn]
        nObs += 1
        nDataTot += obs.wl.shape[0]
    
    return wlSynSet, nDataTot



def mainFittingLoop(par, lineData, wlSynSet, sGrid, briMap, magGeom, listGridView,
                    dMagCart0, setSynSpec, coMem, nDataTot, Data, sig2,
                    allModeldIdV, weightEntropy, verbose=1):

    import core.memSimple3 as memSimple
    import core.lineprofileVoigt as lineprofile
    
    chi_aim = par.chiTarget*float(coMem.nDataTotIV)
    if (par.fixedEntropy == 1):
        target_aim = par.ent_aim
    else:
        target_aim = chi_aim

    fOutFitSummary = open('outFitSummary.txt', 'w')

    #Initialize goodness of fit and convergence parameters
    Chi2 = chi2nu = 0.0
    entropy = 0.0
    test = 1.0
    meanBright = meanBrightDiff = meanMag = 0.0
    iIter = 0
    bConverged = False

    #loop over fitting iterations
    #this version better allows for fitting to target entropy or chi^2
    while ((iIter < par.numIterations) and (bConverged == False)):
        
        #Compute set of new spectra and derivatives
        iIter += 1
        if (iIter > 1):
            memSimple.unpackImageVector(Image, briMap, magGeom, par.magGeomType,
                                        par.fitBri, par.fitMag)
        
        #First get magnetic vectors (their derivatives can be pre-calculated)
        vecMagCart = magGeom.getAllMagVectorsCart()
        
        nObs=0
        for phase in par.cycleList:
            #get stellar geometry calculations for this phase
            sGridView = listGridView[nObs]
            #calculate spectrum and derivatives
            spec = setSynSpec[nObs]
            spec.updateIntProfDeriv(sGridView, vecMagCart, dMagCart0, briMap,
                                    lineData, par.calcDI, par.calcDV)
            spec.convolveIGnumpy(par.instrumentRes)
            
            nObs += 1
        #finished computing spectra and derivatives
    
        #Pack the input arrays for mem_iter
        allModelIV = memSimple.packModelVector(setSynSpec, par.fitBri, par.fitMag)
        Image = memSimple.packImageVector(briMap, magGeom, par.magGeomType,
                                          par.fitBri, par.fitMag)
        if ((par.calcDI == 1) | (par.calcDV == 1)):
            allModeldIdV = memSimple.packResponseMatrix(setSynSpec, nDataTot,
                                    coMem.npBriMap, magGeom, par.magGeomType,
                                    par.calcDI, par.calcDV)
    
        #Call the mem_iter routine controlling the fit.  This returns the entropy, Chi2, test 
        # for the current model, and then proposes a new best fit model in Image
        entropy, Chi2, test, Image, entStand, entFF, entMag = \
            memSimple.mem_iter(coMem.n1Model, coMem.n2Model, coMem.nModelTot, \
                               Image, Data, allModelIV, sig2, allModeldIdV, \
                               weightEntropy, par.defaultBright, par.defaultBent, \
                               par.maximumBright, target_aim, par.fixedEntropy)
        
        meanBright = np.sum(briMap.bright*sGrid.area)/np.sum(sGrid.area)
        meanBrightDiff = np.sum(np.abs(briMap.bright-par.defaultBright)*sGrid.area)/np.sum(sGrid.area)
        absMagCart = np.sqrt(vecMagCart[0,:]**2 + vecMagCart[1,:]**2 + vecMagCart[2,:]**2)
        meanMag = np.sum(absMagCart*sGrid.area)/np.sum(sGrid.area)
        
        #evaluate some convergence criteria:
        if (par.fixedEntropy == 1):
            bConverged = ((entropy >= par.ent_aim*1.001) and (test < par.test_aim) and (iIter > 2))
        else:
            bConverged = ((Chi2 <= chi_aim*1.001) and (test < par.test_aim))
        if (coMem.nDataTotIV > 0): #protect against fitting nothing
            chi2nu = Chi2/float(coMem.nDataTotIV)
        else:
            chi2nu = Chi2
        
        if(verbose == 1):
            print('it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}'.format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff, meanMag) )
        fOutFitSummary.write('it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}\n'.format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff, meanMag) )
        if((verbose == 1) and (bConverged == True)):
            print('Success: sufficiently small value of Test achieved')

    #In case this was run with no iterations, just calculate the model diagonstics
    if(iIter == 0):
        allModelIV = memSimple.packModelVector(setSynSpec, par.fitBri, par.fitMag)
        chi2nu = np.sum((allModelIV - Data)**2/sig2)/float(coMem.nDataTotIV)

        Image = memSimple.packImageVector(briMap, magGeom, par.magGeomType,
                                          par.fitBri, par.fitMag)
        if(par.fitBri == 1 or par.fitMag == 1):
            entropy, tmpgS, tmpggS, tmp3, tmp4, SI1, SI2, SB \
                = memSimple.get_s_grads(coMem.n1Model, coMem.n2Model, coMem.nModelTot,
                                        Image, weightEntropy, par.defaultBright,
                                        par.defaultBent, par.maximumBright)
        
        meanBright = np.sum(briMap.bright*sGrid.area)/np.sum(sGrid.area)
        meanBrightDiff = np.sum(np.abs(briMap.bright-par.defaultBright)*sGrid.area) \
                         /np.sum(sGrid.area)
        vecMagCart = magGeom.getAllMagVectorsCart()
        absMagCart = np.sqrt(vecMagCart[0,:]**2 + vecMagCart[1,:]**2 + vecMagCart[2,:]**2)
        meanMag = np.sum(absMagCart*sGrid.area)/np.sum(sGrid.area)

    if(verbose != 1 and par.numIterations > 0) or (verbose == 1 and iIter == 0):
        print('it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}'.format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff, meanMag) )
        fOutFitSummary.write('it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}\n'.format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff, meanMag) )
    fOutFitSummary.close()
    
    return iIter, entropy, chi2nu, test, meanBright, meanBrightDiff, meanMag


def saveModelProfs(par, setSynSpec, lineData, saveName):
    #save final model spectra
    fOutputSpec = open(saveName,'w')
    nPhase=0
    for spec in setSynSpec:
        wl = (spec.wl-lineData.wl0)/lineData.wl0*c+par.velRs[nPhase]  #velocity in observer rest frame
        fOutputSpec.write('#cycle %f\n' % (par.cycleList[nPhase]))
        for i in range(spec.wl.shape[0]):
            fOutputSpec.write('%e %e %e\n' % (wl[i], spec.IIc[i], spec.VIc[i]) )
        fOutputSpec.write('\n')
    
        #And save in individual files as ZDI input ready, LSD profile format
        sigmaOut = 1e-8
        fOutLDS = open(par.fnames[nPhase].strip()+'.model', 'w')
        fOutLDS.write('#synthetic prof cycle %f\n' % (par.cycleList[nPhase]))
        fOutLDS.write('%i %i \n' % (spec.wl.shape[0], 6))
        for i in range(spec.wl.shape[0]):
            fOutLDS.write('%f %e %e %e %e %e %e\n' % (wl[i],  spec.IIc[i], sigmaOut, spec.VIc[i], sigmaOut, 0., sigmaOut) )
        fOutLDS.close()
        #########################################################################
        ##And save as ZDI input ready, LSD profile format with Gaussian noise.
        #sigmaIOut = 1e-4
        #sigmaVOut = 1e-5
        #fOutLDS = open(par.fnames[nPhase].strip()+'.modelN', 'w')
        #fOutLDS.write('#synthetic prof cycle %f\n' % (par.cycleList[nPhase]))
        #fOutLDS.write('%i %i \n' % (spec.wl.shape[0], 6))
        #for i in range(spec.wl.shape[0]):
        #    #fOutLDS.write('%f %e %e %e %e %e %e\n' % (wl[i],  spec.IIc[i]+noiseI[i], sigmaIOut, spec.VIc[i]+noiseV[i], sigmaVOut, 0.+noiseN[i], sigmaVOut) )
        #    noiseI = np.random.normal(spec.IIc[i], sigmaIOut)
        #    noiseV = np.random.normal(spec.VIc[i], sigmaVOut)
        #    noiseN = np.random.normal(0.0, sigmaVOut)
        #    fOutLDS.write('%f %e %e %e %e %e %e\n' % (wl[i],  noiseI, sigmaIOut, noiseV, sigmaVOut, noiseN, sigmaVOut) )
        #fOutLDS.close()
        ##########################################################################
    
        nPhase += 1
    fOutputSpec.close()


def saveObsUsed(obsSet, fName):
    outObserved = open(fName,'w')
    for tmpObs in obsSet:
        for i in range(tmpObs.wl.shape[0]):
            outObserved.write('%e %e %e %e %e\n' % (tmpObs.wl[i], tmpObs.specI[i], tmpObs.specIsig[i], tmpObs.specV[i], tmpObs.specVsig[i]))
        outObserved.write('\n')
    outObserved.close()
 
    
###############################
def equivWidComp2(lineStr, meanEquivWidObs, setSynSpec, lineData):
    #Calculate the difference between a scaled synthetic profile
    #equivalent width and an input observed equivalent width.
    #Mostly for use when fitting line strength by equivalent width.
    lineStr0 = lineData.str[0]
    nObs=0
    meanEW = 0.
    for spec in setSynSpec:

        scaleI = 1. - (1. - spec.IIc)*(lineStr/lineStr0)
        equivWidApprox = 0.
        for i in range(spec.wl.shape[0]-1):
            equivWidApprox += (1.-scaleI[i])*(spec.wl[i+1]-spec.wl[i])
        equivWidApprox += (1.-scaleI[-1])*(spec.wl[-1]-spec.wl[-2])
        meanEW += equivWidApprox

        nObs += 1
    meanEW /= float(nObs)
    diffEW = np.abs(meanEquivWidObs - meanEW)
    return diffEW


def fitLineStrength(meanEquivWidObs, par, listGridView, vecMagCart, dMagCart0, briMap, lineData, wlSynSet, verbose=1):
    # Try fitting the model line strength
    # Match the model equivalent width to the mean observed equivalent width.
    # Since the simple brightness model used in this code cannot change 
    # the equivalent width of a line, these quantities should generally match.
    from scipy.optimize import minimize_scalar
    import core.lineprofileVoigt as lineprofile
    
    if(verbose == 1):
        print('fitting line strength (by equivalent width)')
    #Generate a set of spectra, then re-scale them (saves recalculating full spectra). 
    #Valid for Gaussian or Voigt local profiles, but not radiative transfer solutions
    nObs = 0
    setSynSpec = []
    for phase in par.cycleList:
        spec = lineprofile.diskIntProfAndDeriv(listGridView[nObs], vecMagCart, dMagCart0, briMap, lineData, par.velEq, wlSynSet[nObs], 0, 0)
        spec.convolveIGnumpy(par.instrumentRes)
        setSynSpec += [spec]
        nObs += 1

    fitLineStr = minimize_scalar(equivWidComp2 , bracket=(lineData.str[0]*0.01, lineData.str[0], lineData.str[0]*100.0), method='brent', tol=1e-5, args=(meanEquivWidObs, setSynSpec, lineData))

    if(verbose == 1):
        ewidth = equivWidComp2(fitLineStr.x, 0, setSynSpec, lineData)
        print('best match line strength is {:f} (ew {:f} )'.format(fitLineStr.x, ewidth))
    lineData.str[0] = fitLineStr.x
############################################

