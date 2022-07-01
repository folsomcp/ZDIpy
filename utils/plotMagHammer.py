#!/usr/bin/python3
#
#Plot a magnetic field from set of magnetic spherical harmonic coefficients
#coefficients are read in from a file. This version plots magnetic vector
#components in 2D, a Hammer projection.

import numpy as np
try:
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom

def main(magCoeffFile = 'outMagCoeff.dat', incDeg = 45.):
    """
    Optionally set an input file of spherical harmonic coefficients.
    Set the inclination of the star, to only plot the visible portion
    of the surface.
    """
    # Parameters defining the observed orientation of the star
    inc = incDeg/180.*np.pi
    phaseList = [0.]
    
    #the number of latitudinal grid points to use, for the spherical star
    nGridLatSph = 90
    
    #the number of latitudinal grid points to use, for cartesian magnetic field map
    nGridLatRec = 90
    #total number of grid points is nGridLatRec*(2*nGridLatRec)
    
    #initialise the stellar grid
    sGrid = geometryStellar.starGrid(nGridLatSph)
    
    #initialize the magnetic geometry spherical harmonics from a file of coefficients
    magGeom = magneticGeom.magSphHarmoicsFromFile(magCoeffFile)


    plotHammerPhaseLat(magGeom, nGridLatRec, incDeg)



##########################################################
def plotHammerPhaseLat(inMagHarm, npClat, inclination=90.):
    #2D plot of the magnetic map based on the coefficients magHarm
    #Uses matplotlib, produces a map in colatitude and longitude,
    #and plots the radial, longitudinal and latitudinal components of the magnetic field
    #Note: The latitudnal magnetic component runs in the opposit direction from the 
    # colatitudinal component (+ve towards the 'north' rotational pole).  
    # The longitudinal component runs in same direction as allways 
    # (increasing right handed), but phase runs in the opposite direction to 
    # longitude (since we assume the star is rotating in the right handed sense)

    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import cm
    import copy
    import numpy.ma as ma
    
    #for 'type 3' fonts
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ##optionally set font size
    #matplotlib.rcParams.update({'font.size': 12})
    
    eps=1e-5  #used as a small epsion value to avoid the poles in spherical coordinates

    #Get phases of observations, a clunky way.
    specname = './outLineModels.dat'
    cycle = []
    nspec = 0
    inSpec = open(specname, 'r')
    for line in inSpec:
        if(line.strip() == ''):  #skip blank lines
            continue
        if (line.strip()[0] == '#'):
            cycle += [float(line.split()[1])]
            nspec += 1
    obsphase = np.array(cycle) % 1.
    inSpec.close()
    
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

    #Automatically set a scale for all the maps, symmetric in +ve and -ve, and the same for all componenets
    BrMax = np.amax(np.fabs(Br))
    BlatMax = np.amax(np.fabs(Blat))
    BlonMax = np.amax(np.fabs(Blon))
    BMax = np.amax([BrMax, BlatMax, BlonMax])

    fullLat = 0.5*np.pi-fullClat
    fullLat = np.reshape(fullLat, (npClat,npLon))[:,::-1]
    fullLonPlt = np.pi-np.reshape(fullLon, (npClat,npLon))[:,::-1]
    incLat = -inclination*np.pi/180.
    BrMask = ma.masked_where(fullLat < incLat, Br)
    BlatMask = ma.masked_where(fullLat < incLat, Blat)
    BlonMask = ma.masked_where(fullLat < incLat, Blon)

    labelPhases = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    #cmapMagMask = copy.copy(cm.get_cmap('bwr'))
    cmapMagMask = copy.copy(cm.get_cmap('RdBu_r'))
    cmapMagMask.set_bad('lightgrey')

    fig = plt.figure(figsize=(5., 6.))
    
    #Radial map
    ax1 = plt.subplot(3,1,1, projection="hammer")
    #ax1 = plt.subplot(3,1,1, projection="mollweide")
    mesh = plt.pcolormesh(fullLonPlt, fullLat, BrMask, cmap=cmapMagMask, vmin=-BMax, vmax=BMax, rasterized=True)
    #mesh.set_edgecolor("face")
    plt.grid(True)
    plt.ylabel('latitude (deg)')
    ax1.xaxis.set_tick_params(label1On=False)
    #by hand x-axis phase labels
    for j in labelPhases:
        plt.text(np.pi*(j-0.5)*2, 0, '{:3.1f}'.format(j), ha='center')
    
    ax1.set_title('radial', loc='right')
    #plt.text(0.99, 0.02, 'radial', transform=ax1.transAxes, ha='right')
    plt.colorbar()
    #alternately force number of color bar ticks
    #tickerCBTrim = matplotlib.ticker.MaxNLocator(8)
    #plt.colorbar(ticks=tickerCBTrim)
    
    #Add ticks for observed phases
    for j in range (0, nspec):
        xpos = (obsphase[j]-0.5)*2*np.pi
        ystart = -inclination*np.pi/180.
        plt.plot([xpos, xpos], [ystart, ystart-np.pi/15.], 'k', linewidth=1)

    
        
    #Azimuthal map
    ax2 = plt.subplot(3,1,2, projection="hammer")
    #ax2 = plt.subplot(3,1,2, projection="mollweide")
    mesh = plt.pcolormesh(fullLonPlt, fullLat, BlonMask, cmap=cmapMagMask, vmin=-BMax, vmax=BMax, rasterized=True)
    #mesh.set_edgecolor("face")
    plt.grid(True)
    plt.ylabel('latitude (deg)')
    ax2.xaxis.set_tick_params(label1On=False)
    #by hand x-axis phase labels
    for j in labelPhases:
        plt.text(np.pi*(j-0.5)*2, 0, '{:3.1f}'.format(j), ha='center')
    
    ax2.set_title('azimuthal', loc='right')
    #plt.text(0.99, 0.02, 'azimuthal', transform=ax2.transAxes, ha='right')
    plt.colorbar()
    #alternately force number of color bar ticks
    #tickerCBTrim = matplotlib.ticker.MaxNLocator(8)
    #plt.colorbar(ticks=tickerCBTrim)
    
    #Add ticks for observed phases
    for j in range (0, nspec):
        xpos = (obsphase[j]-0.5)*2*np.pi
        ystart = -inclination*np.pi/180.
        plt.plot([xpos, xpos], [ystart, ystart-np.pi/15.], 'k', linewidth=1)


    
    #Meridional map
    ax3 = plt.subplot(3,1,3, projection="hammer")
    #ax3 = plt.subplot(3,1,3, projection="mollweide")
    mesh = plt.pcolormesh(fullLonPlt, fullLat, BlatMask, cmap=cmapMagMask, vmin=-BMax, vmax=BMax, rasterized=True)
    #mesh.set_edgecolor("face")
    plt.grid(True)
    plt.ylabel('latitude (deg)')
    ax3.xaxis.set_tick_params(label1On=False)
    #by hand x-axis phase labels
    for j in labelPhases:
        plt.text(np.pi*(j-0.5)*2, 0, '{:3.1f}'.format(j), ha='center')
    
    ax3.set_title('meridional', loc='right')
    #plt.text(0.99, 0.02, 'meridional', transform=ax3.transAxes, ha='right')
    plt.colorbar()
    #alternately force number of color bar ticks
    #tickerCBTrim = matplotlib.ticker.MaxNLocator(8)
    #plt.colorbar(ticks=tickerCBTrim)
    
    #Add ticks for observed phases
    for j in range (0, nspec):
        xpos = (obsphase[j]-0.5)*2*np.pi
        ystart = -inclination*np.pi/180.
        plt.plot([xpos, xpos], [ystart, ystart-np.pi/15.], 'k', linewidth=1)

    plt.tight_layout()
    plt.show()


##################################################################

if __name__ == "__main__":
    main()
