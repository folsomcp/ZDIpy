#!/usr/bin/python3
#
#Plot brightness maps from DI+ZDI

import numpy as np
try:
    import core.geometryStellar as geometryStellar
    import core.brightnessGeom as brightnessGeom
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.geometryStellar as geometryStellar
    import core.brightnessGeom as brightnessGeom

def main(fileBrightMap = 'outBrightMap.dat'):

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
    sGrid = geometryStellar.starGrid(nPtsClat)
    
    #plot maps
    plotBrightPolar(briMap, sGrid, flagLog=False, flagSym=True)

##########################################################
def plotBrightPolar(briMap, starGrid, flagLog, flagSym):
    #2D plot of the magnetic map based on the coefficients magHarm
    #Uses matplotlib, produces a map in colatitude and longitude,
    #and plots the radial, longitudinal and colatitudinal components of the magnetic field

    import matplotlib.pyplot as plt

    xLong = np.zeros(4*starGrid.numPoints)
    yClat = np.zeros(4*starGrid.numPoints)
    zBright = np.zeros(2*starGrid.numPoints)
    triangles = np.zeros([2*starGrid.numPoints,3], dtype=int)

    if(flagLog):
        bright = np.log10(briMap.bright)
    else:
        bright = briMap.bright
    
    minBright = 1e9
    for i in range(starGrid.numPoints):
        if(starGrid.clat[i] < np.pi*120./180.):
            corners =  starGrid.GetCellCorners(i)

            xLong[i*4] = corners[0,0]
            xLong[i*4+1] = corners[1,0]
            xLong[i*4+2] = corners[2,0]
            xLong[i*4+3] = corners[3,0]
            
            yClat[i*4] = corners[0,1] 
            yClat[i*4+1] = corners[1,1]
            yClat[i*4+2] = corners[2,1]
            yClat[i*4+3] = corners[3,1]
            
            #padd out the central 3 cells, for dispaly
            if(i < 3):  
                #The first and second verticies are repeats, so modify order 
                #(suppres second) and add an extra 'fourth' verticy
                xLong[i*4] = corners[0,0]
                yClat[i*4] = corners[0,1] 
                xLong[i*4+1] = corners[2,0]
                yClat[i*4+1] = corners[2,1]
                xLong[i*4+2] = corners[3,0]
                yClat[i*4+2] = corners[3,1]
                xLong[i*4+3] = (corners[0,0]+corners[1,0])*0.5
                yClat[i*4+3] = corners[3,1]*1.2

            triangles[2*i,:]   = [i*4, i*4+1, i*4+3]
            triangles[2*i+1,:] = [i*4, i*4+2, i*4+3]
            
            zBright[2*i] = bright[i]
            zBright[2*i+1] = bright[i]

    #minBright = np.min(bright)
    minBright = 0.

    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('S')
    ax.set_rgrids([np.pi*30./180., np.pi*60./180., np.pi*90./180., np.pi*119./180.], labels=['30', '60', '90', '120'], angle=0., ha='center')
    if(flagSym & flagLog):
        plt.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='RdYlBu', vmin = minBright, vmax = -minBright)
    elif(flagSym):
        plt.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='RdYlBu', vmin = minBright, vmax = 2.-minBright)
    else:
        plt.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='hot') #, vmin = minBright, vmax = zBright.max()
    # use edgecolors='k' or 'none' to show triangle edges, 
    # cmap= 'afmhot', 'hot', 'copper', 'PuOr', 'RdBu', 'RdYlBu'
    plt.axis([xLong.min(), xLong.max(), yClat.min(), yClat.max()])
    #plt.axis([xLong.min(), xLong.max(), 0., 0.5])
    plt.colorbar()
    #plt.xlabel('longitude')
    #plt.ylabel('colatitude')
    
    line30x = np.linspace(0., 2.*np.pi, 360)
    line30y = np.ones(line30x.shape)*np.pi*30./180.
    line60x = np.linspace(0., 2.*np.pi, 360)
    line60y = np.ones(line60x.shape)*np.pi*60./180.
    line90x = np.linspace(0., 2.*np.pi, 360)
    line90y = np.ones(line90x.shape)*np.pi*90./180.
    plt.plot(line30x, line30y, 'k--')
    plt.plot(line60x, line60y, 'k--')
    plt.plot(line90x, line90y, 'k-')

    plt.show()
##########################################################

if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Plot a given brightness map (defaults to outBrightMap.dat)')
    argParser.add_argument("brightMap", nargs='?', default='outBrightMap.dat')
    args = argParser.parse_args()
    
    fileBrightMap = args.brightMap
    
    main(fileBrightMap = fileBrightMap)
