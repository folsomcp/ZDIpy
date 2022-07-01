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
nGridLatSph = 90
nGridLatVec = 20
#use ~30-40 for a a star with a total of ~1000-2000 surface elements

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
##Magnetic stuff for the spherical surface
#generatea new copy of the magnetic vector map (probably not necessary)
magGeom = magneticGeom.magSphHarmoics(inMagGeom.nl)
magGeom.alpha = inMagGeom.alpha
magGeom.beta = inMagGeom.beta
magGeom.gamma = inMagGeom.gamma
magGeom.initMagGeom(sGrid.clat, sGrid.long)
Bxc, Byc, Bzc = magGeom.getAllMagVectorsCart()

Bmag = np.sqrt(Bxc**2 + Byc**2 + Bzc**2) #abs magnetic strength at cell center

xp, yp, zp = sGrid.GetCartesianCells()


#Set up a grid in colatitude and longitude for interpolating onto
npSphGrid = 100
tmpClat = np.linspace(0., np.pi, npSphGrid)
tmpLong = np.linspace(0., 2*np.pi, 2*npSphGrid)
gridClat, gridLong = np.meshgrid(tmpClat, tmpLong)

#Padd the edges of the ZDI stellar grid, so that the linear interpolation routine can handle the edges
sGridPadClat = sGrid.clat
sGridPadLong = sGrid.long
radiusPad = sGrid.radius
BmagPad = Bmag

#padd clat < 0
indCMin =  np.where(sGrid.clat == np.min(sGrid.clat))
sGridPadClat = np.append(sGrid.clat[indCMin] - sGrid.dClat[indCMin], sGridPadClat)
sGridPadLong = np.append(sGrid.long[indCMin], sGridPadLong)
radiusPad = np.append(sGrid.radius[indCMin], radiusPad)
BmagPad = np.append(Bmag[indCMin], BmagPad)

#padd clat > pi
indCMax =  np.where(sGrid.clat == np.max(sGrid.clat))
sGridPadClat = np.append(sGridPadClat, sGrid.clat[indCMax] + sGrid.dClat[indCMax])
sGridPadLong = np.append(sGridPadLong, sGrid.long[indCMax])
radiusPad = np.append(radiusPad, sGrid.radius[indCMax])
BmagPad = np.append(BmagPad, Bmag[indCMax])

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
        BmagPad = np.append(BmagPad, Bmag[indLastEdge])
        
        #find the pixel at this (i) clat and long = 2pi 
        j = i
        while j < sGrid.numPoints-1 and sGrid.clat[j] <= lastClat:
            j += 1
        indNextEdge = j - 1
        #the long < 0 edge
        sGridPadClat = np.append(sGridPadClat, sGrid.clat[i])
        sGridPadLong = np.append(sGridPadLong, sGrid.long[i]-sGrid.dLong[i])
        radiusPad = np.append(radiusPad, sGrid.radius[i])
        BmagPad = np.append(BmagPad, Bmag[indNextEdge])
        
        indLastEdge = i

##Interpolate using scipy multi-dimensional interpolation from unstructured data
##(This seems to be flexible if possibly a little imprecise)
from scipy.interpolate import griddata

#Bgrid = griddata( (sGrid.clat,sGrid.long), Bmag, (gridClat, gridLong), method='nearest')
#Bgrid = griddata( (sGrid.clat,sGrid.long), Bmag, (gridClat, gridLong), method='linear', fill_value=0.0)
radiusGrid = griddata( (sGridPadClat,sGridPadLong), radiusPad, (gridClat, gridLong), method='linear', fill_value=0.0)
Bgrid = griddata( (sGridPadClat,sGridPadLong), BmagPad, (gridClat, gridLong), method='linear', fill_value=-10.0)

#Convert the grid to Cartesian coordinates for the plotting routine
xgrid = radiusGrid*np.sin(gridClat)*np.cos(gridLong)
ygrid = radiusGrid*np.sin(gridClat)*np.sin(gridLong)
zgrid = radiusGrid*np.cos(gridClat)



##########################################################
##Magnetic stuff for the vector arrows

#offset above the surface of the sphere for displaying magnetic vectors
#used to keep the vectors from disappearing below the surface of the plot
offsetBvec = 0.05

#generatea new copy of the magnetic vector map
magGeomV = magneticGeom.magSphHarmoics(inMagGeom.nl)
magGeomV.alpha = inMagGeom.alpha
magGeomV.beta = inMagGeom.beta
magGeomV.gamma = inMagGeom.gamma
magGeomV.initMagGeom(sGridVec.clat, sGridVec.long)
BxcV, BycV, BzcV = magGeomV.getAllMagVectorsCart()
BrV, BclatV, BlonV = magGeomV.getAllMagVectors()

cbBmagV = np.sqrt(BxcV**2 + BycV**2 + BzcV**2) #abs magnetic strength at cell center

centers = sGridVec.GetCartesianCells()
#position values for the plot of magnetic vectors ("3D quiver plot")

xc = centers[0,:] + offsetBvec*centers[0,:]
yc = centers[1,:] + offsetBvec*centers[1,:]
zc = centers[2,:] + offsetBvec*centers[2,:]



##########################################################
##Plotting setup details
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


quivNorm = 0.2/np.max(cbBmagV)
for i in range(xc.shape[0]):
    #lenV = cbBmagV[i] / np.max(cbBmagV)*0.3+0.03  #not necessary in newer versions of mplot3d, can just set the normalize keyword to False.  Otherwise use length=lenV
    if BrV[i] > 0.0:
        quiv = ax.quiver(xc[i], yc[i], zc[i], BxcV[i]*quivNorm, BycV[i]*quivNorm, BzcV[i]*quivNorm, normalize=False, color=(1,0,0), pivot='tail')
    else:
        quiv = ax.quiver(xc[i], yc[i], zc[i], BxcV[i]*quivNorm, BycV[i]*quivNorm, BzcV[i]*quivNorm, normalize=False, color=(0,0,1), pivot='tip')

    #quiv.set_sort_zpos(zc[i]) #probably not necessary, should be roughly set automatically

#get colors from a colormap, for optional later use
#First get a colormap object.  Mostly you could just use cmap='xxx' in functions, exept for the line below.
#colormap = plt.cm.afmhot
#colormap = plt.cm.binary
#colormap = plt.cm.Spectral
#colormap = plt.cm.inferno
#colormap = plt.cm.inferno_r
#colormap = plt.cm.magma
#colormap = plt.cm.magma_r
#colormap = plt.cm.cividis_r
#colormap = plt.cm.Purples
colormap = plt.cm.BuPu
#colormap = plt.cm.viridis_r

tmpFrac = Bgrid/np.max(Bgrid)
tmpcolors=colormap(tmpFrac)
tmpsurf= ax.plot_surface(xgrid, ygrid, zgrid, facecolors=tmpcolors, rstride=1, cstride=1, shade=False)

#set the image z position of the sphere to be 0, otherwise mplot3d seems to do something odd/wrong for some quiver plots.
#Hypothesis: Things (quiver points) in the image deeper than 0.0 should be beyond the limb of the sphere and not drawn, while things shallower than that should be on the visible surface and drawn on top of the sphere.  So setting the sort_zpos to 0 is correct.
tmpsurf.set_sort_zpos(0.0)


##Plot the rotation axis, this only works when viewed from the northern hemisphere
##This dosen't interact well with the 3D quiver, seems to get z sorting wrong
#tmpLineax1 = ax.plot3D([0,0],[0,0],[1.0,1.3], color='k', linewidth=2, zorder=+1.3)
#tmpLineax2 = ax.plot3D([0,0],[0,0],[-1.0,-1.3], color='k', linewidth=2, zorder=-1.3)
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

#And label this rotation phase
ax.text2D(0.1, 0.95, 'Phase {:0.2f}'.format(phaseList[0]), transform=ax.transAxes, fontsize='large')

#work around to make a color bar work for the plot
#Need to make a "mappable" colormap object, as a proxy for the plot_surface
#(the 2D color array and 2D data arrays are not "mappable", and seem not to generate one automatically,
# unlike most plotting functions.  Probably because I have to color things manualy in plot_surface.)
tmpCBmap = plt.cm.ScalarMappable(norm=None, cmap=colormap)
tmpCBmap.set_array(Bgrid)
colorbar = fig.colorbar(tmpCBmap) #shrink=0.5, aspect=5)
colorbar.set_label('|B| (G)')

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)

#remove position axes with tickmarks etc.
ax.set_axis_off()

ax.view_init(90-incDeg, -phaseList[0]*360.)

plt.show()
