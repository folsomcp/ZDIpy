#!/usr/bin/python3
#
# Run a grid of ZDI model fits
#
import numpy as np
try:
    import zdipy
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import zdipy
import core.mainFuncs as mf
import core.readObs as readObs
import core.lineprofileVoigt as lineprofile
 
#Just read input files once
par = mf.readParamsZDI('inzdi.dat')
obsSet = readObs.obsProfSetInRange(par.fnames, par.velStart, par.velEnd,
                                   par.velRs)
lineData = lineprofile.lineData('model-voigt-line.dat', par.instrumentRes)

fOutGrid = open('outGridInc.dat', 'w')

# Loop over grid quantities #
incRange = np.linspace(10.0, 90., 80+1)
for inclination in incRange:
    
    #Save this iteration's paramters into par
    par.incRad = inclination/180.*np.pi
    #Keep vsini constaint:
    #The equitorial velocity needs to be updated for i to keep vsini constaint.
    par.velEq = par.vsini/np.sin(par.incRad)
    
    print('i {:f}'.format(inclination))
    
    iIter, entropy, chi2, test, meanBright, meanBrightDiff, meanMag = zdipy.main(par, obsSet, lineData, 0)
    
    fOutGrid.write('i {:3f} it {:3n} entropy {:13.5f} chi2 {:10.6f} Test {:10.6f} bright {:10.7f} spot {:10.7f} mag {:10.4f}\n'.format(inclination, iIter, entropy, chi2, test, meanBright, meanBrightDiff, meanMag) )
    
fOutGrid.close()
