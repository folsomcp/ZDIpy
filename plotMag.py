#!/usr/bin/python3
#
#Plot a magnetic field from set of magnetic spherical harmonic coefficients
#coefficients are read in from a file. This version plots magnetic vector
#components in 2D, rectangular coordinates.

import numpy as np
import core.magneticGeom as magneticGeom
import core.geometryStellar as geometryStellar

def main(magCoeffFile = 'outMagCoeff.dat', incDeg = 90.):
    """
    Optionally set an input file of spherical harmonic coefficients.
    Set the inclination of the star, to only plot the visible portion
    of the surface.
    """
    # Parameters defining the observed orientation of the star.
    inc = incDeg/180.*np.pi
    phaseList = [0.]
    
    #the number of latitudinal grid points to use, for the spherical star
    nGridLatSph = 80
    
    #the number of latitudinal grid points to use, for cartesian magnetic field map
    nGridLatRec = 80
    #total number of grid points is nGridLatRec*(2*nGridLatRec)
    
    #initialise the stellar grid
    sGrid = geometryStellar.starGrid(nGridLatSph)
    
    #initialize the magnetic geometry spherical harmonics from a file of coefficients
    magGeom = magneticGeom.magSphHarmoicsFromFile(magCoeffFile)
    
    #print('ploting spherical B components rectangularly')
    plotRectPhaseLat(magGeom, nGridLatRec, incDeg)


##########################################################
def plotRectPhaseLat(inMagHarm, npClat, inclination=90.):
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
    try:
        inSpec = open(specname, 'r')
    except:
        print('not plotting phase ticks, no info from {:}'.format(specname))
        obsphase = np.array([0.0])
    else:
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

    lat = lat*180./np.pi
    fig = plt.figure(figsize=(5., 6.))
    #Radial map
    plt.subplot(3,1,1)
    plt.imshow(Br, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=-BMax, vmax=BMax, aspect=0.5/180., interpolation='nearest', cmap='RdBu_r') #'bwr' 'jet'
    plt.ylabel('latitude (deg)')
    plt.yticks(np.linspace(-90,90,7)) #force tick locations
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Radial')
    plt.text(0.05,-inclination+10.,'radial')
    plt.colorbar()
    #alternately force number of color bar ticks
    #tickerCBTrim = matplotlib.ticker.MaxNLocator(8)
    #plt.colorbar(ticks=tickerCBTrim)
    
    #Add ticks for observed phases
    for j in range (0, nspec):
        plt.text(obsphase[j], 92., "|", size=8)

    #Azimuthal map
    plt.subplot(3,1,2)
    plt.imshow(Blon, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=-BMax, vmax=BMax, aspect=0.5/180., interpolation='nearest', cmap='RdBu_r') #'bwr' plt.cm.jet
    plt.ylabel('latitude (deg)')
    plt.yticks(np.linspace(-90,90,7)) #force tick locations
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Longitudinal')
    plt.text(0.05,-inclination+10.,'azimuthal')
    plt.colorbar()
    #plt.colorbar(ticks=tickerCBTrim)
    
    #Meridional map
    plt.subplot(3,1,3)
    plt.imshow(Blat, origin='upper', extent=(phase[0], phase[npLon-1], lat[npClat-1], lat[0]), vmin=-BMax, vmax=BMax, aspect=0.5/180., interpolation='nearest', cmap='RdBu_r') #'bwr' plt.cm.jet
    plt.xlabel('rot. phase')
    plt.ylabel('latitude (deg)')
    plt.yticks(np.linspace(-90,90,7)) #force tick locations
    plt.ylim(-inclination,90)
    #plt.text(0.05,lat[npClat-1]+5.,'Latitudinal')
    plt.text(0.05,-inclination+10.,'meridional')
    plt.colorbar()
    #plt.colorbar(ticks=tickerCBTrim)

    plt.tight_layout()
    plt.show()
##################################################################

if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Plot a given magnetic map from spherical harmonic coefficents (defaults to outMagCoeff.dat)')
    argParser.add_argument("magCoeffFile", nargs='?', default='outMagCoeff.dat')
    args = argParser.parse_args()
    
    magCoeffFile = args.magCoeffFile
    main(magCoeffFile = magCoeffFile)
