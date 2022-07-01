#!/usr/bin/python3
#
#Plot a brightness map from a outBrightMap.dat file
#plotted in 3D using matplotlib (if possible)

import numpy as np
try:
    import core.geometryStellar as geometryStellar
    import core.brightnessGeom as brightnessGeom
    import core.mainFuncs as mf
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.geometryStellar as geometryStellar
    import core.brightnessGeom as brightnessGeom
    import core.mainFuncs as mf


def main(fileBrightMap = 'outBrightMap.dat', phase=0.0):
    
    fileModelParams = 'inzdi.dat'
    
    clat, lon, inBrigthMap = np.loadtxt(fileBrightMap, unpack=True)
    
    print('reading i, P, M, and R from {:}'.format(fileModelParams))
    par = mf.readParamsZDI('inzdi.dat')
    incDeg = par.inclination
    incRad = par.incRad
    
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
    sGrid = geometryStellar.starGrid(nPtsClat, par.period, par.mass, par.radius)

    plotBright3Dsmooth(incDeg, phase, briMap, sGrid)

##########################################################
#actual plotting...
def plotBright3Dsmooth(incDeg, phase, briMap, sGrid):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.tri as mtri
    
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    xp, yp, zp = sGrid.GetCartesianCells()
    
    
    #Set up a grid in colatitude and longitude for interpolating onto
    npSphGrid = 100
    tmpClat = np.linspace(0., np.pi, npSphGrid)
    tmpLong = np.linspace(0., 2*np.pi, 2*npSphGrid)
    gridClat, gridLong = np.meshgrid(tmpClat, tmpLong)
    
    sGridPadClat = sGrid.clat
    sGridPadLong = sGrid.long
    radiusPad = sGrid.radius
    brightPad = briMap.bright
    
    #padd clat < 0
    indCMin =  np.where(sGrid.clat == np.min(sGrid.clat))
    sGridPadClat = np.append(sGrid.clat[indCMin] - sGrid.dClat[indCMin], sGridPadClat)
    sGridPadLong = np.append(sGrid.long[indCMin], sGridPadLong)
    radiusPad = np.append(sGrid.radius[indCMin], radiusPad)
    brightPad = np.append(briMap.bright[indCMin], brightPad)
    
    #padd clat > pi
    indCMax =  np.where(sGrid.clat == np.max(sGrid.clat))
    sGridPadClat = np.append(sGridPadClat, sGrid.clat[indCMax] + sGrid.dClat[indCMax])
    sGridPadLong = np.append(sGridPadLong, sGrid.long[indCMax])
    radiusPad = np.append(radiusPad, sGrid.radius[indCMax])
    brightPad = np.append(brightPad, briMap.bright[indCMax])
    
    #padd long>0, long < 2pi
    lastClat = -100.
    indLastEdge = 0
    for i in range(sGrid.numPoints):
        if sGrid.clat[i] > lastClat:
            lastClat = sGrid.clat[i]
            #the long > 2pi edge
            sGridPadClat = np.append(sGridPadClat, sGrid.clat[i-1])
            sGridPadLong = np.append(sGridPadLong, sGrid.long[i-1]+sGrid.dLong[i-1])
            radiusPad = np.append(radiusPad, sGrid.radius[i-1])
            #Use the pixel at clat[i-1] and long=0 (for clat[i-1] long>2pi)
            brightPad = np.append(brightPad, briMap.bright[indLastEdge])
    
            #find the pixel at this (i) clat and long = 2pi 
            j = i
            while j < sGrid.numPoints-1 and sGrid.clat[j] <= lastClat:
                j += 1
            indNextEdge = j - 1
            #the long < 0 edge
            sGridPadClat = np.append(sGridPadClat, sGrid.clat[i])
            sGridPadLong = np.append(sGridPadLong, sGrid.long[i]-sGrid.dLong[i])
            radiusPad = np.append(radiusPad, sGrid.radius[i])
            #Use the pixel at clat[i] and long=2pi (for clat[i] long<0)
            brightPad = np.append(brightPad, briMap.bright[indNextEdge])
            
            indLastEdge = i
    
    ##Interpolate using scipy multi-dimensional interpolation from unstructured data
    ##(This seems to be flexible if possibly a little imprecise)
    from scipy.interpolate import griddata
    
    #brightGrid = griddata( (sGrid.clat,sGrid.long), brightPad, (gridClat, gridLong), method='nearest')
    #brightGrid = griddata( (sGrid.clat,sGrid.long), brightPad, (gridClat, gridLong), method='linear', fill_value=0.0)
    radiusGrid = griddata( (sGridPadClat,sGridPadLong), radiusPad, (gridClat, gridLong), method='linear', fill_value=0.0)
    brightGrid = griddata( (sGridPadClat,sGridPadLong), brightPad, (gridClat, gridLong), method='linear', fill_value=-10.0)
    
    #Convert the grid to Cartesian coordinates for the plotting routine
    xgrid = radiusGrid*np.sin(gridClat)*np.cos(gridLong)
    ygrid = radiusGrid*np.sin(gridClat)*np.sin(gridLong)
    zgrid = radiusGrid*np.cos(gridClat)
    
    
    
    ##########################################################
    ##Plotting setup details
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #get colors from a colormap, for optional later use
    #First get a colormap object.  Mostly you could just use cmap='xxx' in functions, exept for the line below.
    colormap = plt.cm.afmhot
    #colormap = plt.cm.binary
    #colormap = plt.cm.Spectral
    #colormap = plt.cm.inferno
    #colormap = plt.cm.inferno_r
    #colormap = plt.cm.magma
    #colormap = plt.cm.magma_r
    
    #tmpFrac = brightGrid/np.max(brightGrid)
    tmpFrac = (brightGrid-np.min(brightGrid))/(np.max(brightGrid)-np.min(brightGrid))
    tmpcolors=colormap(tmpFrac)
    
    tmpsurf= ax.plot_surface(xgrid, ygrid, zgrid, facecolors=tmpcolors, rstride=1, cstride=1, shade=False)
    
    #set the image z position of the sphere to be 0, otherwise mplot3d seems to do something odd/wrong for some quiver plots.
    #Hypothesis: Things (quiver points) in the image deeper than 0.0 should be beyond the limb of the sphere and not drawn, while things shallower than that should be on the visible surface and drawn on top of the sphere.  So setting the sort_zpos to 0 is correct.
    tmpsurf.set_sort_zpos(0.0)
    
    #Plot the rotation axis, this only works when viewed from the northern hemisphere
    #tmpLineax1 = ax.plot([0,0],[0,0],[1.0,1.6], color='k', linewidth=2, zorder=+1)
    #tmpLineax2 = ax.plot([0,0],[0,0],[-1.0,-1.6], color='k', linewidth=2, zorder=-1)
    #Since that only works if you force the zorder, instead use quiver objects
    #with arrow head set to 0, since the get the zorder right automatically.
    tmpLineax1 = ax.quiver(0.0, 0.0, 1.0, 0.0, 0.0, 0.6, normalize=False,
                           linewidth=1, color='k', pivot='tail',
                           arrow_length_ratio=0.0)
    tmpLineax2 = ax.quiver(0.0, 0.0,-1.0, 0.0, 0.0,-0.6, normalize=False,
                           linewidth=1, color='k', pivot='tail',
                           arrow_length_ratio=0.0)
    
    #And try to build an equator out of line segments too
    eqR = np.max(sGrid.radius)
    nSegments = 100
    for i in range(nSegments):
        theta = 2*np.pi*(float(i)/nSegments)
        dtheta = 2*np.pi*(1./nSegments)
        circ_x1 = eqR*np.sin(theta)
        circ_y1 = eqR*np.cos(theta)
        #circ_x2 = eqR*np.sin(theta+dtheta)
        #circ_y2 = eqR*np.cos(theta+dtheta)
        #tmpCirc = ax.plot3D([circ_x1,circ_x2], [circ_y1,circ_y2], [0.0,0.0],
        #                  color='k', linewidth=1)
        circ_dx = eqR*(np.sin(theta+dtheta) - np.sin(theta))
        circ_dy = eqR*(np.cos(theta+dtheta) - np.cos(theta))
        #Again use quiver objects here, since they get the zorder right.
        tmpCirc = ax.quiver(circ_x1, circ_y1, 0.0, circ_dx, circ_dy, 0.0,
                            normalize=False, linewidth=1, color='k',
                            pivot='tail', arrow_length_ratio=0.0)
    
    #work around to make a color bar work for the plot
    #Need to make a "mappable" colormap object, as a proxy for the plot_surface
    #(the 2D color array and 2D data arrays are not "mappable", and seem not to generate one automatically,
    # unlike most plotting functions.  Probably because I have to color things manualy in plot_surface.)
    tmpCBmap = plt.cm.ScalarMappable(norm=None, cmap=colormap)
    tmpCBmap.set_array(brightGrid)
    fig.colorbar(tmpCBmap) #shrink=0.5, aspect=5)
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    
    #remove position axes with tickmarks etc.
    ax.set_axis_off()
    
    ax.view_init(90-incDeg, -phase*360.)
    
    plt.show()

#####################################################

if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Plot a given brightness map on a 3D surface (defaults to outBrightMap.dat)')
    argParser.add_argument("brightMap", nargs='?', default='outBrightMap.dat')
    argParser.add_argument("phase", nargs='?', default='0.0')
    args = argParser.parse_args()
    
    phase = 0.0
    phase = float(args.phase)
    
    fileBrightMap = args.brightMap
    
    main(fileBrightMap = fileBrightMap, phase = phase)
