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

    plotBright3D(incDeg, phase, briMap, sGrid)

##########################################################
#actual plotting...
def plotBright3D(incDeg, phase, briMap, sGrid):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.tri as mtri
    
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    
    #initialize a bunch of lists (or arrays)
    x = np.zeros(len(sGrid.long)*4)   #cell corners in Cartesian coordinates
    y = np.zeros(len(sGrid.long)*4)
    z = np.zeros(len(sGrid.long)*4)
    triangCorn = np.zeros((len(sGrid.long)*2, 3),dtype=int) #triangles, referring triplets of cell corners
    
    
    cbBright = np.repeat(briMap.bright,2)
    
    for i in range(len(sGrid.long)):
    
        #get 4 cell conrners, x,y,z positions, and save them into flat arrays
        corners = sGrid.GetCartesianCellCorners(i)
        for j in range(4):
            x[i*4+j] = corners[j,0]
            y[i*4+j] = corners[j,1]
            z[i*4+j] = corners[j,2]
                
        #define triangles (triplets of indices for vertices) for matplotlib Triangulation (3 corners anti-clockwise)
        triangCorn[2*i,   0] = i*4+0
        triangCorn[2*i,   1] = i*4+2
        triangCorn[2*i,   2] = i*4+3
        triangCorn[2*i+1, 0] = i*4+0
        triangCorn[2*i+1, 1] = i*4+3
        triangCorn[2*i+1, 2] = i*4+1
    
    
    ##########################################################
    ##Plotting setup details
            
    triFix = mtri.Triangulation(x, y, triangles=triangCorn)
    
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
    #tmpFrac = cbBright/np.max(cbBright)
    
    if np.max(cbBright)-np.min(cbBright) > 1e-8:
        tmpFrac = (cbBright-np.min(cbBright))/(np.max(cbBright)-np.min(cbBright))
    else:
        tmpFrac = np.ones_like(cbBright)*0.5
    tmpcolors=colormap(tmpFrac)
    
    tmpsurf= ax.plot_trisurf(triFix, z, cmap=colormap, shade=False, linewidth=0.5, zsort='average') #, rasterized=True) #
    #set the 'image array' from the array of brightness values, and use the colormap from cmap=xxx above
    tmpsurf.set_array(cbBright)
    #tmpsurf.autoscale()
    #Or set triangle colors directly (this won't work if cmap is allready set)
    #tmpsurf= ax.plot_trisurf(triFix, z, linewidth=0.5)
    #tmpsurf.set_facecolors(tmpcolors)
    #Try using small appropriately colored lines to fill in 1 pixel gaps between triangles. 
    tmpsurf.set_edgecolor(tmpcolors) 
    
    #set the image z position of the sphere to be 0, otherwise mplot3d seems to do something odd/wrong for some quiver plots.
    #Hypothesis: Things (quiver points) in the image deeper than 0.0 should be beyond the limb of the sphere and not drawn, while things shallower than that should be on the visible surface and drawn on top of the sphere.  So setting the sort_zpos to 0 is correct.
    tmpsurf.set_sort_zpos(0.0)
    
    #Plot the rotation axis, this only works when viewed from the northern hemisphere
    #tmpLineax1 = ax.plot3D([0,0],[0,0],[1.0,1.6], color='k', linewidth=1, zorder=+1)
    #tmpLineax2 = ax.plot3D([0,0],[0,0],[-1.0,-1.6], color='k', linewidth=1, zorder=-1)
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
    #(the 2D color array and 2D data arrays are not "mappable", and seem not to
    #generate one automatically, unlike most plotting functions.  Probably because
    #I have to color things manualy in plot_surface.)
    tmpCBmap = plt.cm.ScalarMappable(norm=None, cmap=colormap)
    tmpCBmap.set_array(cbBright)
    fig.colorbar(tmpCBmap) #shrink=0.5, aspect=5)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
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
