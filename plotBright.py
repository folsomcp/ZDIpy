#!/usr/bin/python3
#
#Plot brightness maps from DI+ZDI

import numpy as np
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
    plotBrightRectPhaseLat(briMap, sGrid)

##########################################################
def plotBrightRectPhaseLat(briMap, starGrid, minBriPlt=0, maxBriPlt=2):
    #2D plot of the brightness map, using the same pixels as on the surface of the modle stars
    #Uses matplotlib, produces a map in colatitude and longitude.x

    import matplotlib.pyplot as plt

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

    ax = plt.subplot(111)
    #plt.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='copper', vmin=minBriPlt, vmax=maxBriPlt)
    briplt = ax.tripcolor(xLong, yClat, triangles, facecolors=zBright, edgecolors='none', cmap='RdYlBu', vmin=minBriPlt, vmax=maxBriPlt)
    # use edgecolors='k' to show triangle edges, cmap= 'afmhot', 'hot', 'copper'
    ax.axis([xLong.min(), xLong.max(), yClat.min(), yClat.max()])
    ax.set_aspect(1./360., adjustable='box')
    plt.colorbar(briplt, ax=ax)
    ax.set_xlabel('phase')
    ax.set_ylabel('latitude')
    plt.tight_layout()
    plt.show()

#####################################################

if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Plot a given brightness map (defaults to outBrightMap.dat)')
    argParser.add_argument("brightMap", nargs='?', default='outBrightMap.dat')
    args = argParser.parse_args()
    
    fileBrightMap = args.brightMap
    
    main(fileBrightMap = fileBrightMap)
