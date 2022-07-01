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
obsSet = readObs.obsProfSetInRange(par.fnames, par.velStart, par.velEnd, par.velRs)
lineData = lineprofile.lineData('model-voigt-line.dat', par.instrumentRes)

fOutGrid = open('outGridPdO.dat', 'w')

# Loop over grid quantities #
#For a very coarse, quick-look, grid try
periodRange = np.linspace(0.420, 0.424, 10+1) #(0.1, 1.5, 150+1)
dOmeagRange = np.linspace(0.0, 0.10, 10+1) #(-0.01, 0.01, 2+1)
#For a finer grid (this may be slow!) try:
#periodRange = np.linspace(0.420, 0.424, 20+1)# (0.420, 0.424, 40+1) 
#dOmeagRange = np.linspace(0.0, 0.10, 20+1)# (0.0, 0.10, 40+1) 
for period in periodRange:
    for dOmega in dOmeagRange:
        
        #Save this iteration's paramters into par
        par.period = period
        par.dOmega = dOmega
        
        print('p {:f} dO {:f}'.format(period, dOmega))
        
        iIter, entropy, chi2, test, meanBright, meanBrightDiff, meanMag = zdipy.main(par, obsSet, lineData, 0)
        
        fOutGrid.write('p {:f} dO {:f} it {:3n} entropy {:13.5f} chi2 {:10.6f} Test {:10.6f} bright {:10.7f} spot {:10.7f} mag {:10.4f}\n'.format(period, dOmega, iIter, entropy, chi2, test, meanBright, meanBrightDiff, meanMag) )
        
fOutGrid.close()
