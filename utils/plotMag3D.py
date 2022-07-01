#!/usr/bin/python3
#
#Plot a magnetic field from set of magnetic spherical harmonic coefficients
#cofficiants are read in from a file, and plotted in 3D using matplotlib (if possible)

import numpy as np
try:
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom
    import core.mainFuncs as mf
except ImportError:
    #If this is run from the utils sub-directory, try adding the path
    #to the main ZDIpy directory, containing the core sub-directory/module.
    #(There may be a better way to do this?)
    import sys
    sys.path += [sys.path[0]+'/../']
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom
    import core.mainFuncs as mf


#incDeg = 60.
#inc = incDeg/180.*np.pi
phaseList = [0]

magCoeff = 'outMagCoeff.dat'
fileModelParams = 'inzdi.dat'

#the number of latitudinal grid points to use, for the spherical star
nGridLatSph = 60
nGridLatVec = 30

print('reading i, P, M, and R from {:}'.format(fileModelParams))
par = mf.readParamsZDI('inzdi.dat')
incDeg = par.inclination
incRad = par.incRad

#initialise the stellar grid
sGrid = geometryStellar.starGrid(nGridLatSph, par.period, par.mass, par.radius)

sGridVec = geometryStellar.starGrid(nGridLatVec, par.period, par.mass, par.radius)

#initilaise the magnetic geometry spherical harmonics from a file of coifficents
inMagGeom = magneticGeom.magSphHarmoicsFromFile(magCoeff)



##########################################################
#actual plotting...

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



##########################################################


#offset above the surface of the sphere for displaying magnetic vectors
#used to keep the vectors from disappearing below the surface of the plot
offsetBvec = 0.05

#initialize a bunch of lists (or arrays)
x = np.zeros(len(sGrid.long)*4)   #cell corners in Cartesian coordinates
y = np.zeros(len(sGrid.long)*4)
z = np.zeros(len(sGrid.long)*4)
triangCorn = np.zeros((len(sGrid.long)*2, 3),dtype=int) #triangles, referring triplets of cell corners


##Magnetic stuff for the spherical surface
#generatea new copy of the magnetic vector map (probably not necessary)
magGeom = magneticGeom.magSphHarmoics(inMagGeom.nl)
magGeom.alpha = inMagGeom.alpha
magGeom.beta = inMagGeom.beta
magGeom.gamma = inMagGeom.gamma
magGeom.initMagGeom(sGrid.clat, sGrid.long)
Bxc, Byc, Bzc = magGeom.getAllMagVectorsCart()

_cbBmag = np.sqrt(Bxc**2 + Byc**2 + Bzc**2) #abs magnetic strength at cell center, one for each cell corner
cbBmag = np.repeat(_cbBmag,2)

for i in range (len(sGrid.long)):
    
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
##Magnetic stuff for the vector arrows
#generatea new copy of the magnetic vector map
magGeomV = magneticGeom.magSphHarmoics(inMagGeom.nl)
magGeomV.alpha = inMagGeom.alpha
magGeomV.beta = inMagGeom.beta
magGeomV.gamma = inMagGeom.gamma
magGeomV.initMagGeom(sGridVec.clat, sGridVec.long)
BxcV, BycV, BzcV = magGeomV.getAllMagVectorsCart()
BrV, BclatV, BlonV = magGeomV.getAllMagVectors()

cbBmagV = np.sqrt(BxcV**2 + BycV**2 + BzcV**2) #abs magnetic strength at cell center, one for each cell corner

centers = sGridVec.GetCartesianCells()
#position values for the plot of magnetic vectors ("3D quiver plot")

xc = centers[0,:] + offsetBvec*centers[0,:]
yc = centers[1,:] + offsetBvec*centers[1,:]
zc = centers[2,:] + offsetBvec*centers[2,:]



##########################################################
##Plotting setup details
        
triFix = mtri.Triangulation(x, y, triangles=triangCorn)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

quivNorm = 0.2/np.max(cbBmagV)
for i in range(xc.shape[0]):
    #lenV = cbBmagV[i] / np.max(cbBmagV)*0.001+0.0003  #not necessary in newer versions of mplot3d, can just set the normalize keyword to False.  Otherwise use length=lenV
    if BrV[i] > 0.0:
        quiv = ax.quiver(xc[i], yc[i], zc[i], BxcV[i]*quivNorm, BycV[i]*quivNorm, BzcV[i]*quivNorm, normalize=False, color=(1,0,0), pivot='tail')
    else:
        quiv = ax.quiver(xc[i], yc[i], zc[i], BxcV[i]*quivNorm, BycV[i]*quivNorm, BzcV[i]*quivNorm, normalize=False, color=(0,0,1), pivot='tip')

    quiv.set_sort_zpos(zc[i]) #probably not necessary, should be roughly set automatically

#get colors from a colormap, for optional later use
#First get a colormap object.  Mostly you could just use cmap='xxx' in functions, exept for the line below.
#colormap = plt.cm.binary
#colormap = plt.cm.cividis_r
#colormap = plt.cm.Purples
colormap = plt.cm.BuPu
#colormap = plt.cm.viridis_r
#colormap = plt.cm.YlGn
#colormap = plt.cm.YlOrBr
#colormap = plt.cm.afmhot_r
#colormap = plt.cm.inferno_r
#colormap = plt.cm.magma_r
#Or for a hand-made color map:
#from matplotlib.colors import ListedColormap
#Ncolors = 265
#cvals = np.ones((Ncolors, 4))
#cvals[:, 0] = np.linspace(1., 96./256, Ncolors) #This gives a white to purple
#cvals[:, 1] = np.linspace(1.,  0./256, Ncolors) #color scale
#cvals[:, 2] = np.linspace(1., 128./256, Ncolors)
#colormap = ListedColormap(cvals)
tmpFrac = cbBmag/np.max(cbBmag)
tmpcolors=colormap(tmpFrac)

tmpsurf= ax.plot_trisurf(triFix, z, cmap=colormap, shade=False, linewidth=.5)
#set triangle colors directly
#tmpsurf.set_facecolors(tmpcolors)
#or set the 'image array' from the array of |B| values, and use the colormap from cmap=xxx above
tmpsurf.set_array(cbBmag)
#tmpsurf.autoscale()
#Try using small appropriately colored lines to fill in 1 pixel gaps between triangles. 
tmpsurf.set_edgecolor(tmpcolors) 

#set the image z position of the sphere to be 0, otherwise mplot3d seems to do something odd/wrong for some quiver plots.
#Hypothesis: Things (quiver points) in the image deeper than 0.0 should be beyond the limb of the sphere and not drawn, while things shallower than that should be on the visible surface and drawn on top of the sphere.  So setting the sort_zpos to 0 is correct.
tmpsurf.set_sort_zpos(0.0)


##Plot the rotation axis, this only works when viewed from the northern hemisphere
#This dosen't interact well with the 3D quiver, seems to get z sorting wrong
#tmpLineax1 = ax.plot3D([0,0],[0,0],[1.0,1.5], color='k', linewidth=2, zorder=+1.3)
#tmpLineax2 = ax.plot3D([0,0],[0,0],[-1.0,-1.5], color='k', linewidth=2, zorder=-1.3)
#Instead build the rotaion axis from quiver objects 
tmpLineax1 = ax.quiver(0.0, 0.0, 1.0, 0.0, 0.0, 0.8, normalize=False, linewidth=2, color=(0,0,0), pivot='tail',arrow_length_ratio=0.0)
tmpLineax2 = ax.quiver(0.0, 0.0,-1.0, 0.0, 0.0,-0.8, normalize=False, linewidth=2, color=(0,0,0), pivot='tail',arrow_length_ratio=0.0)
#And try to build an equator out of quiver objects too
eqR = np.max(sGrid.radius)
for i in range(100):
    theta = 2*np.pi*(i/100.)
    dtheta = 2*np.pi*(1./100.)
    circ_x1 = eqR*np.sin(theta)
    circ_y1 = eqR*np.cos(theta)
    circ_dx = eqR*(np.sin(theta+dtheta) - np.sin(theta))
    circ_dy = eqR*(np.cos(theta+dtheta) - np.cos(theta))
    tmpCirc = ax.quiver(circ_x1, circ_y1, 0.0, circ_dx, circ_dy, 0.0, normalize=False, linewidth=1, color=(0.0,0.0,0.0), pivot='tail',arrow_length_ratio=0.0)

#work around to make a color bar work for the plot
#Need to make a "mappable" colormap object, as a proxy for the plot_surface
#(the 2D color array and 2D data arrays are not "mappable", and seem not to generate one automatically,
# unlike most plotting functions.  Probably because I have to color things manualy in plot_surface.)
tmpCBmap = plt.cm.ScalarMappable(norm=None, cmap=colormap)
tmpCBmap.set_array(cbBmag)
colorbar = fig.colorbar(tmpCBmap) #shrink=0.5, aspect=5)
colorbar.set_label('|B| (G)')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax.view_init(90-incDeg, -phaseList[0]*360.)

plt.show()
