#!/usr/bin/python3
#
#Generate a set of synthetic line profiles, in Stokes I and V
# use a set of magnetic spherical harmonic coefficients from magCoeff.dat
# and use a model line profile with parameters from model-line.dat
#

__verison__ = "0.4.3"

#import numpy as np
import core.mainFuncs as mf
import core.readObs as readObs
import core.geometryStellar as geometryStellar
import core.magneticGeom as magneticGeom
import core.brightnessGeom as brightnessGeom
import core.lineprofileVoigt as lineprofile
import core.memSimple3 as memSimple

#(Making the main program a function allows it to be called from other programs)
def main(par=None, obsSet=None, lineData=None, verbose = 1):
    
    ###############################
    #Read in model and observation details
    
    #read file of model/control parameters
    if par is None:
        par = mf.readParamsZDI('inzdi.dat')
    # read in the observed spectra/LSD profiles
    if obsSet is None :
        obsSet = readObs.obsProfSetInRange(par.fnames, par.velStart,
                                           par.velEnd, par.velRs)
    par.setTarget()
    par.setCalcdIdV(verbose)
    par.calcCycles(verbose)
    
    #Get the model line data, for use in diskIntProf (and localProfile)
    if lineData is None:
        lineData = lineprofile.lineData('model-voigt-line.dat',
                                        par.instrumentRes)
    
    ###############################
    #initialize model
    
    #Generate wavelength grid for synthesis
    wlSynSet, nDataTot = mf.getWavelengthGrid(par.velRs, obsSet, lineData,
                                              verbose)
    
    #Rescale the Stokes I error bars, to incorporate the chi^2 scaling (indirectly)
    for obs in obsSet:
        obs.scaleIsig(par.chiScaleI)
    
    #initialize the stellar grid
    sGrid = geometryStellar.starGrid(par.nRingsStellarGrid, par.period,
                                     par.mass, par.radius, verbose)
    
    
    #initialize the magnetic geometry spherical harmonics from a file
    #of coefficients, or to a constant value (of 0).
    magGeom = magneticGeom.SetupMagSphHarmoics(sGrid, par.initMagFromFile,
                                               par.initMagGeomFile, par.lMax,
                                               par.magGeomType, verbose)
    
    #initialize the brightness map from a file, or set to the 'default brightness'
    briMap = brightnessGeom.SetupBrightMap(sGrid, par.initBrightFromFile,
                                           par.initBrightFile,
                                           par.defaultBright, verbose)
    
    
    ######################
    #Pre-calculate and save quantities that do not change during fitting iterations 
    #(but do change between rotation phases/observations)
    
    #Calculate projected stellar geometry parameters for each observed phase
    listGridView = geometryStellar.getListGridView(par, sGrid)
    
    #Magnetic vectors and their derivatives with respect to alpha, beta, 
    #gamma coefficients (the derivatives are constant throughout the program,
    #but the magnetic vectors vary when fitting)
    vecMagCart = magGeom.getAllMagVectorsCart()
    if(par.fitMag == 1):
        dMagCart0 = magGeom.getAllMagDerivsCart()
    else:
        dMagCart0 = 0.
        
    if(par.estimateStrenght == 1):
        #estimate observed equivalent widths
        meanEquivWidObs = readObs.getObservedEW(obsSet, lineData, verbose)
        mf.fitLineStrength(meanEquivWidObs, par, listGridView, vecMagCart,
                           dMagCart0, briMap, lineData, wlSynSet, verbose)

    #Initialize synthetic spectra, pre-calculating some quantities that don't 
    #change when fitting (e.g. local profile shapes), at the cost of memory.
    setSynSpec = lineprofile.getAllProfDiriv(par, listGridView, vecMagCart,
                                             dMagCart0, briMap, lineData,
                                             wlSynSet)
    
    ############################################
    
    #setup constants for mem_iter
    constMem = memSimple.constantsMEM(par, briMap, magGeom, nDataTot)
    
    Data, sig2 = memSimple.packDataVector(obsSet, par.fitBri, par.fitMag)
    
    #If response matrix was pre-calculated above and won't need recalculating
    #(fitting only V).
    if((par.calcDV == 1) & (par.calcDI != 1)):
        allModeldIdV = memSimple.packResponseMatrix(setSynSpec, nDataTot,
                                                     constMem.npBriMap,
                                                     magGeom, par.magGeomType, 
                                                     par.fitBri, par.fitMag)
        #in this case the problem is sufficiently linear that the response
        #matrix will not change.
        par.calcDV = 0
    else:
        allModeldIdV = 0
    
    #Extra weighting applied to the entropy terms
    weightEntropy = memSimple.setEntropyWeights(par, magGeom, sGrid)
    
    #Run the main fitting loop
    iIter, entropy, chi2, test, meanBright, meanBrightDiff, meanMag = \
         mf.mainFittingLoop(par, lineData, wlSynSet, sGrid, briMap, magGeom,
                            listGridView, dMagCart0, setSynSpec, constMem,
                            nDataTot, Data, sig2, allModeldIdV, weightEntropy,
                            verbose)
    
    #save final spherical harmonic coefficients
    #magGeom.saveToFile('outMagCoeff.dat')
    magGeom.saveToFile('outMagCoeff.dat', compatibility = True)
    
    #save final brightness map to file
    brightnessGeom.saveMap(briMap, 'outBrightMap.dat')
    #Also generate and save the brightness map weighted by gravity darkening
    briMapGDark = brightnessGeom.brightMap(sGrid.clat, sGrid.long)
    briMapGDark.bright = briMap.bright*sGrid.gravityDarkening(lineData.gravDark)
    brightnessGeom.saveMap(briMapGDark, 'outBrightMapGDark.dat')
    
    #save final model spectra
    mf.saveModelProfs(par, setSynSpec, lineData, 'outLineModels.dat')
    
    #save exact observed datapoints used
    mf.saveObsUsed(obsSet, 'outObserved.dat')
    
    return iIter, entropy, chi2, test, meanBright, meanBrightDiff, meanMag



# Boilerplate for running the main function #
if __name__ == "__main__":
    main()
