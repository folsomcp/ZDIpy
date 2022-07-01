#!/usr/bin/python3
#
#Plot a magnetic field from set of magnetic spherical harmonic coefficients
#officiants are read in from a file, and plots can be in 2D (magnetic components) or 3D (magnetic vectors)
#
# This file is the main program.  It calls: geometryStellar.py, magneticGeom.py, plotFunc.py

import numpy as np
try:
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom
    import core.brightnessGeom as brightnessGeom
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom
    import core.brightnessGeom as brightnessGeom

def main(fileBrightMap = 'outBrightMap.dat', magCoeffFile = 'outMagCoeff.dat', incDeg = 90.):
    """
    Optionally set an input file of spherical harmonic coefficients.
    Set the inclination of the star, to only plot the visible portion
    of the surface.
    """
    # Parameters defining the observed orientation of the star
    inc = incDeg/180.*np.pi
    phaseList = [0.]
    
    #Read brightness map
    clat, lon, inBrigthMap = np.loadtxt(fileBrightMap, unpack=True)
    #setup brightMap object from read array
    briMap = brightnessGeom.brightMap(clat, lon)
    briMap.bright = inBrigthMap
    #determin number of points in co-latitude, for the stellar geometry 
    lastClat = -100.
    nPtsClat = 0
    for thisClat in clat:
        if (thisClat > lastClat):
            nPtsClat += 1
            lastClat = thisClat
    #initialise the stellar grid
    sGrid = geometryStellar.starGrid(nPtsClat)
    
    #initilaise the magnetic geometry spherical harmonics from a file of coifficents
    magGeom = magneticGeom.magSphHarmoicsFromFile(magCoeffFile)
    
    #the number of latitudinal grid points to use, for cartesian magnetic field map
    nGridLatRec = nPtsClat
    #total number of grid points is nGridLatRec*(2*nGridLatRec)
    
    plotBrightMagRectPhaseLat(magGeom, nGridLatRec, briMap, sGrid, incDeg)


##########################################################
def plotBrightMagRectPhaseLat(inMagHarm, npClat, briMap, starGrid, inclination=90.):
    #2D plot of the brightness map, using the same pixels as on the surface of the modle stars
    #2D plot of the magnetic map based on the coefficients magHarm
    #Uses matplotlib, produces a map in colatitude and longitude,
    #and plots the radial, longitudinal and latitudinal components of the magnetic field
    #Note: The latitudnal magnetic component runs in the opposit direction from the 
    # colatitudinal component (+ve towards the 'north' rotational pole).  
    # The longitudinal component runs in same direction as allways 
    # (increasing right handed), but phase runs in the opposite direction to 
    # longitude (since we assume the star is rotating in the right handed sense)

    import matplotlib.pyplot as plt

    eps=1e-5  #used as a small epsion value to avoid the poles in spherical coordinates
    fig = plt.figure(figsize=(5., 10.))


    #The block for setting up the brightness map
    xLong = np.zeros(4*starGrid.numPoints)
    yClat = np.zeros(4*starGrid.numPoints)
    zBright = np.zeros(2*starGrid.numPoints)
    triangles = np.zeros([2*starGrid.numPoints,3], dtype=int)

    for i in range(starGrid.numPoints):
        corners =  starGrid.GetCellCorners(i)

        xLong[i*4] = corners[0,0]
        xLong[i*4+1] = corners[1,0]
        xLong[i*4+2] = corners[2,0]
        xLong[i*4+3] = corners[3,0]

        yClat[i*4] = corners[0,1] 
        yClat[i*4+1] = corners[1,1]
        yClat[i*4+2] = corners[2,1]
        yClat[i*4+3] = corners[3,1]

        triangles[2*i,:]   = [i*4, i*4+1, i*4+3]
        triangles[2*i+1,:] = [i*4, i*4+2, i*4+3]
        
        zBright[2*i] = briMap.bright[i]
        zBright[2*i+1] = briMap.bright[i]

    xLong = 1. - xLong/(2*np.pi)
    yClat = 90. - yClat*(180./np.pi)

    plt.subplot(5,1,1, aspect=1./360.)
    #plt.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='copper')
    plt.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='RdYlBu', vmin = 0., vmax = 2.)
    # use edgecolors='k' to show triangle edges, cmap= 'afmhot', 'hot', 'copper'
    #plt.axis([xLong.min(), xLong.max(), -inclination, yClat.max()])
    plt.ylim(-inclination, yClat.max())
    plt.xlim(xLong.min(), xLong.max())
    #plt.axes().set_aspect(1./360., adjustable='box')
    plt.colorbar()
    #plt.xlabel('phase')
    plt.ylabel('latitude')
    plt.text(0.05,-inclination+5.,'Brightness')
    #Optionally set the number of tics on the x-axis
    plt.locator_params(axis='x', nbins=5)

    #The block for the magnetic map
    #set up a grid in colatitude and longitude
    npLon = 2*npClat
    lon = np.linspace(0., 2.*np.pi, 2*npClat)
    phase = np.linspace(0., 1., 2*npClat)
    clat = np.linspace(0.+eps, np.pi-eps, npClat)
    lat = np.linspace(0.5*np.pi+eps, -0.5*np.pi-eps, npClat)
    #Faster array operations (~4x faster)
    #setup 1D arrays with the latitude and colongitude
    fullClat = np.repeat(clat, npLon) #one step in clat for each full set of lon
    fullLon = np.tile(lon, npClat) #repeating sets of lon for each step in clat

    #get the magnetic field compoenets 
    magHarm = magneticGeom.magSphHarmoics(inMagHarm.nl)
    magHarm.alpha = inMagHarm.alpha
    magHarm.beta = inMagHarm.beta
    magHarm.gamma = inMagHarm.gamma
    magHarm.initMagGeom(fullClat, fullLon)
    Br_long, Bclat_long, Blon_long = magHarm.getAllMagVectors()
    #reshape the output arrays into 2D arrays, and swap the longitude direction
    Br = np.reshape(Br_long, (npClat,npLon))[:,::-1]
    Blat = np.reshape(Bclat_long, (npClat,npLon))[:,::-1]
    Blon = np.reshape(Blon_long, (npClat,npLon))[:,::-1]
    #since latitude runs in the opposite direction to colatitude:
    Blat = -Blat 

    Btot = np.sqrt(Br**2+Blon**2+Blat**2)
    
    #Automatically set a scale for all the maps, symmetric in +ve and -ve, and the same for all componenets
    BrMax = np.amax(np.fabs(Br))
    BlatMax = np.amax(np.fabs(Blat))
    BlonMax = np.amax(np.fabs(Blon))
    BMax = np.amax([BrMax, BlatMax, BlonMax])
    BTotMax = np.amax(Btot)

    lat = lat*180./np.pi
    #Radial map
    plt.subplot(5,1,3)
    plt.imshow(Br, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=-BMax, vmax=BMax, aspect=0.5/180., interpolation='nearest', cmap='RdBu_r') #'bwr' 'jet'
    plt.ylabel('latitude')
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Radial')
    plt.text(0.05,-inclination+5.,'B Radial')
    plt.colorbar()
    #Azimuthal map
    plt.subplot(5,1,4)
    plt.imshow(Blon, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=-BMax, vmax=BMax, aspect=0.5/180., interpolation='nearest', cmap='RdBu_r') #'bwr' plt.cm.jet
    plt.ylabel('latitude')
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Longitudinal')
    plt.text(0.05,-inclination+5.,'B Longitudinal')
    plt.colorbar()
    #Meridional map
    plt.subplot(5,1,5)
    plt.imshow(Blat, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=-BMax, vmax=BMax, aspect=0.5/180., interpolation='nearest', cmap='RdBu_r') #'bwr' plt.cm.jet
    plt.xlabel('rot. phase')
    plt.ylabel('latitude')
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Latitudinal')
    plt.text(0.05,-inclination+5.,'B Latitudinal')
    plt.colorbar()

    plt.subplot(5,1,2)
    plt.imshow(Btot, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=0, vmax=BTotMax, aspect=0.5/180., interpolation='nearest', cmap='YlOrRd') #plt.cm.jet #'bwr'
    #plt.xlabel('rot. phase')
    plt.ylabel('latitude')
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Latitudinal')
    plt.text(0.05,-inclination+5.,'B Total')
    plt.colorbar()


    plt.tight_layout()
    plt.show()

#####################################################

if __name__ == "__main__":
    main()
