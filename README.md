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

If you use these codes for research please cite [Folsom et al. 2018, MNRAS, 474,4956](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.4956F/abstract).  If you use the oblate geometry for very rapid rotators, please also cite [Cang et al. 2020, A&A, 643 A39](https://ui.adsabs.harvard.edu/abs/2020A%26A...643A..39C/abstract).

## Details for zdipy.py

### Input files

inzdi.dat

This contains the majority of the model parameters used by this ZDI program (with a few additional parameters in model-voigt-line.dat).  This file also specifies the LSD profiles to be read in and fit.  Lines beginning with a # are treated as comments and ignored. 
  All input values are labelled by comments in the file (the order of values on a line matters, and the order of lines matters, but the spacing between values on a line does not matter).  
- numRings specifies the number of surface elements in colatitude that make up the model star.  The number of longitudinal elements is automatically generated based on this (as sine of the colatitude).  The total number of surface will be larger, and is reported by the first line of output from the code.
- The code can fit to target chi^2 (as in the standard Skilling & Bryan routine) or to a target entropy.  This is controlled by the flags 'C' for target chi^2 or 'E' for target entropy, followed by the target value to fit to, then the maximum number of fitting iterations allowed.
- test_aim controls the convergence criteria of the code, and is the target value of Skilling & Bryan's Test statistic.  Generally smaller is better converged.  
- There are flags for fitting the magnetic field and the brightness map.  Generally I recommend only fitting one at a time to make things easier.  Since the code uses a weak field approximation for Zeeman splitting, the magnetic field has no impact on Stokes I.  So one can fit a brightness map independent of magnetic field, then fit a magnetic map using that brightness map as input.
- lMax is the maximum l order of the spherical harmonic expansion.  
- B entropy slope controls the 'default value' in the magnetic entropy calculation. This is effectively a break point in the slope of the entropy curve, so spherical harmonic coefficient values beyond this are penalized more strongly.
- The 'mag. geom. type' flag controls the kind of magnetic geometry used in the fit, it accepts values of 'Full', 'Poloidal', 'PotTor', 'Potential'.  'Full' is the standard approach with coefficients alpha beta and gamma all as free parameters.  'Poloidal' fixes the toroidal field (gamma) to 0.  'PotTor' fixes the tangential poloidal field to be consistent with the radial poloidal field (beta = alpha), making the poloidal part of the field potential but also including a toroidal field in the free parameters.  'Potential' fixes the tangential poloidal to the radial poloidal field, and fixes the toroidal field to 0 (beta = alpha & gamma = 0), using a purely potential field geometry.  In the output coefficients file all alpha, beta, and gamma are written, but they will follow these rules.
- The 'chi^2 scaling for Stokes I' and 'Brightness entropy scaling' scale the chi^2 (or equivalently error bars) and entropy terms for Stokes I brightness mapping.  This is useful when simultaneously fitting brightness and magnetic field, but otherwise should be set to 1. 
- There are two modes of brightness fitting, 'traditional image' allows bright and dark spots with no limit on the brightness, while 'filling factor' maps dark spots with a limiting value on brightness.  In both modes the 'default brightness' is the value brightness will tend to in the absence of information (maximum entropy). 'limiting brightness' is only used in filling factor mode, and is the upper limit on brightness (for fitting dark spots only, limiting brightness should only be a bit larger than default brightness).
- 'Automatically estimate line strength' will calculate an equivalent width of the observed line profile in the velocity range below, then set the model line strength by matching that equivalent width (overriding the value in model-voigt-line.dat).  If the observed line profile shape is well reproduced this can be accurate, but if there problems in the profiles (e.g. distortions at the line edges) there can be errors.
- vel_start and vel_end define the start and end point of the model line profile, relative to the line centre (specified below), in velocity space.  
- The last block of lines contains the file names for the set of LSD profiles extracted from observations, followed by the Julian Date of the observation and the velocity of the line cent-er for that observation.  The program will continue reading these lines (and ignoring comments) until it reaches the end of the file.  
- The velocities here are the Doppler shift of the centre of the observed line relative to the line's reference wavelength.  This value should be the same for all observations, unless the star is in a spectroscopic binary (or hosts an extremely massive planet!).  


A set of LSD profiles, in this example in the ./LSDprof directory.  
- The file names (potentially including a directory path) are specified in inzdi.dat.  The LSD profiles are expected to have two lines of header, followed by columns of: velocity (km/s), Stokes I/Ic, sigma for I/Ic, Stokes V/Ic, sigma of V/Ic, a diagnostic null (N), and sigma for N.  (where Ic refers to the Stokes I continuum).  (This should be the usual output format for Donati's LSD.)  For an alternative LSD code see [LSDpy](https://github.com/folsomcp/LSDpy).


model-voigt-line.dat
- This file contains information controlling the model line profile, as well as the limb darkening coefficient.  Lines beginning with a # are treated as comments and ignored.  At the moment only one model line is used, and hence only one (non-comment) line of data can be provided (additional lines may cause the program to crash).  However, this could be expanded to consider multiple lines in the future.  
- The line model should contain, in order: the wavelength of the line (in nm).  The central depth of the local line profile.  The width of the Gaussian line profile (in km/s, this is sqrt(2)*sigma of the Gaussian, equivalent to thermal and microturbulent velocity).  The Lorentzian width for the line (as a fraction of the Gaussian width, this is 1/2*FWHM of the Lorentzian)  The effective Lande factor for the line.  The limb darkening coefficient; a linear limb darkening law is used (e.g. Gray 2005, eq. 17.11, in the form: brightness = 1-coefficient*(1-cos(theta)) ).  A gravity darkening coefficient, used for models that are oblate due to rapid rotation (if this is value omitted gravity darkening is neglected), in the form g^beta.  For fully radiative stars the coefficient should be beta=1.0, for convective stars it should be smaller, see Claret & Bloemen (2011) for temperature and wavelength dependent values.


### Output Files

outMagCoeff.dat
 This file contains the output spherical harmonic coefficients, in the form of Donati et al. 2006.  These coefficients are what actually constitute the ZDI map.  
-  The first line of the file is a header (an unused comment).  
-  The second line contains the number of spherical harmonic l and m combinations, followed by the number of blocks of spherical harmonic coefficients (always 3, since this code always uses alpha, beta and gamma).  The last number on this line is the mode of ZDI in J-F's code (usually -3 for poloidal + toroidal with alpha, beta and gamma coefficients).  However, J-F's code actually uses the complex conjugate of the coefficients (relative to what is written in Donati et al. 2006).  So to maintain compatibility with that code, if the last number is -3, the complex conjugates are used, or if I break compatibility and use the values exactly as written in the paper the last number is set to -30.  ... I may need a better long term solution here!
- The following lines contain the l order and m order of the spherical harmonic, followed by the real and the imaginary component of the coefficient.  The first block of these lines is for the alpha coefficient, followed by a blank line, then the beta coefficient, then a blank line, then the gamma coefficient.  (If the last number in the second line of the file is -3 then the imaginary parts of the coefficients are negative relative to Donati et al. 2006, while if it is -30 the imaginary coefficients are positive.  Yes that is overly confusing.)

outBrightMap.dat
  This is the resulting brightness map (only created if brightness mapping was done).  It contains columns of colatitude, longitude (in radians), and relative brightness.

outLineModels.dat
  This file contains the best fit model line profiles.  
  The file contains blocks of data for each observation.  Each block begins with a comment (starting in a #) listing the rotational cycle at which the model was computed.  The subsequent lines contain columns of velocity, I/Ic and V/Ic.  The block ends in a blank line.  This is repeated for each observed rotational cycle.
  Models for individual phases are also saved in same directory as the observed SOD profiles, with file names of [observation name].model

outObserved.dat
  This file contains the observed (LSD) line profiles data-points that were actually fit.  It could be useful for verifying the program properly read in the observations, and for comparison with lineModels.dat.  Columns are velocity, I/Ic, sigma I/Ic, V/Ic, and sigma(V/Ic).  Each block begins with a comment listing the rotational cycle and file name of the observation, and ends in a black line.  Observations are in the same order as given in inzdi.dat.

outFitSummary.txt
  This file contains some diagnostic output of the fit, for each iteration of the fitting routine (this is also printed to the terminal).


## Other details

renormLSD.py  Renormalize the continuum, and optionally the equivalent widths, of observed LSD profiles.  Parameters controlling this are set in inrenorm.dat  Ideally this would not be necessary, if the observation is well normalized and the LSD process well behaved.  However, the DI code cannot handle the continuum errors that LSD often produces, and thus this may be necessary for brightness mapping when fitting Stokes I.  Additionally, when brightness mapping the code cannot produce variations in equivalent width, and such variations can easily be caused by small systematic errors (e.g. very small amounts of scattered moonlight).  Thus better results when brightness fitting can sometimes be achieved by normalizing the LSD profiles to have the same equivalent width. 
