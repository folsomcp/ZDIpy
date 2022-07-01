# ZDIpy

Zeeman Doppler Imaging, implemented in Python by C. P. Folsom, following the method of Donati et al.

## Overview

This readme includes some basic information about running ZDIpy, however for any serious applications I strongly recommend contacting an expert in ZDI for more detailed advice.

The main DI and ZDI code is zdipy.py, which uses the input files inzdi.dat, model-voigt-line.dat, and the line profiles specified inside inzdi.dat (in this example they are in the LSDprof directory).

There are plotting functions for the results of the ZDI code:
plotIV.py plots the fits to observed LSD profiles (from outLineModels.dat and outObserved.dat).
plotMag.py plots the result of the magnetic mapping (from outMagCoeff.dat)
plotBright.py plots the result of brightness mapping (from outBrightMap.dat by default)

The comp-ana.py provides some information about the strength and geometry of a magnetic map (from outMagCoeff.dat).

renormLSD.py provides a utility for renormalizing LSD (or line) profiles if necessary.  This can be useful for precise brightness mapping.  This uses the input file inrenorm.dat and and the line profiles specified inside inrenorm.dat

Thses programs are based off of Python 3, but may run in Python 2.7.  The programs requires the scipy and numpy libraries, and use matplotlib for visualization.  

If you use these codes for research please cite [Folsom et al. 2018, MNRAS, 474,4956](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.4956F/abstract).  If you use the oblate geometry for very rapid rotators, please also cite [Cang et al. 2020, A&A, 643 A39](https://ui.adsabs.harvard.edu/abs/2020A%26A...643A..39C/abstract)
