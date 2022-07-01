#!/usr/bin/python3
#
# Analyze a ZDI spherical harmonic coefficients file, 
# and generate a set of fractional energies and magnetic strengths 
# for the various components of the magnetic field. 

import numpy as np
import scipy.special
import core.geometryStellar as geometryStellar
import core.magneticGeom as magneticGeom

magCoeff = 'outMagCoeff.dat'
#the number of latitudinal grid points to use, for the spherical star
nGridLatSph = 180

#initialise the stellar grid
sGrid = geometryStellar.starGrid(nGridLatSph)

#initilaise the magnetic geometry spherical harmonics from a file of coifficents
magGeom = magneticGeom.magSphHarmoicsFromFile(magCoeff)



### Magnetic field strengths ###
magGeomTot = magneticGeom.magSphHarmoics(magGeom.nl)
magGeomTot.initMagGeom(sGrid.clat, sGrid.long)
magGeomTot.alpha[:] = magGeom.alpha[:]
magGeomTot.beta[:] = magGeom.beta[:]
magGeomTot.gamma[:] = magGeom.gamma[:]
magVecTot = magGeomTot.getAllMagVectors()
#get magnitudes of vectors
magTot = np.sqrt(magVecTot[0,:]**2 + magVecTot[1,:]**2 + magVecTot[2,:]**2)

iMaxTot = np.argmax(magTot)
maxBtot = magTot[iMaxTot]

meanBtot = np.sum(magTot*sGrid.area)/np.sum(sGrid.area)

indHemi = np.where(sGrid.clat <= np.pi/2.)
meanBhemi = np.sum(magTot[indHemi]*sGrid.area[indHemi])/np.sum(sGrid.area[indHemi])

print('Bmean = {:9.3f} G'.format(meanBtot))
print('Bmax = {:10.3f} G'.format(maxBtot))
print('Bmean(fully visable hemisphere) = {:9.3f} G'.format(meanBhemi))

### Magnetic field energies ###
#### This version evaulates energies from the coefficents directly ####
# Note: this has been checked both algebraically, 
#and numerically by evaluating the magnetic field, then squaring and integrating it.
lTerm = magGeom.l/(magGeom.l+1)
m0mask = np.zeros(magGeom.nTot)
for i in range(magGeom.nTot):
    if (magGeom.m[i] == 0):
        m0mask[i] = 1.

alphaEs = 0.5*magGeom.alpha*np.conj(magGeom.alpha)
alphaM0s = m0mask*0.25*(magGeom.alpha**2 + np.conj(magGeom.alpha)**2)
alphaEs = np.real(alphaEs + alphaM0s)

betaEs = 0.5*lTerm*magGeom.beta*np.conj(magGeom.beta)
betaM0s = m0mask*0.25*lTerm*(magGeom.beta**2 + np.conj(magGeom.beta)**2)
betaEs = np.real(betaEs + betaM0s)

gammaEs = 0.5*lTerm*magGeom.gamma*np.conj(magGeom.gamma)
gammaM0s = m0mask*0.25*lTerm*(magGeom.gamma**2 + np.conj(magGeom.gamma)**2)
gammaEs = np.real(gammaEs + gammaM0s)

totE = np.sum(alphaEs) + np.sum(betaEs) + np.sum(gammaEs)
totEpol = np.sum(alphaEs) + np.sum(betaEs)
totEtor = np.sum(gammaEs)
#magSqu = (magVecTot[0,:]**2 + magVecTot[1,:]**2 + magVecTot[2,:]**2)
#print 'totalE:', totE/(4*np.pi), '(B^2) alt:',  np.sum(magSqu*sGrid.area)/np.sum(sGrid.area)

print('poloidal:    {:7.3%} (% tot)  (<B^2_poloidal> {:.2f} G^2)'.format(
    totEpol/totE, totEpol/(4*np.pi)))
print('toroidal:    {:7.3%} (% tot)  (<B^2_toroidal> {:.2f} G^2)'.format(
    totEtor/totE, totEtor/(4*np.pi)))

Epol_l1 = 0.
Epol_l2 = 0.
Epol_l3 = 0.
Etor_l1 = 0.
Etor_l2 = 0.
Etor_l3 = 0.
for i in range(magGeom.nTot):
    if(magGeom.l[i] == 1): #should be i=0,1
        Epol_l1 += alphaEs[i] + betaEs[i]
        Etor_l1 += gammaEs[i]
    elif(magGeom.l[i] == 2): #should be i=2,3,4
        Epol_l2 += alphaEs[i] + betaEs[i]
        Etor_l2 += gammaEs[i]
    elif(magGeom.l[i] == 3): #should be i=5,6,7,8
        Epol_l3 += alphaEs[i] + betaEs[i]
        Etor_l3 += gammaEs[i]

if totEpol > 0.0:
    print('dipole:      {:7.3%} (% pol)  (<B^2_dip> {:.2f} G^2)'.format(
        Epol_l1/totEpol, Epol_l1/(4*np.pi) ))
    print('quadrupole:  {:7.3%} (% pol)  (<B^2_quad> {:.2f} G^2)'.format(
        Epol_l2/totEpol, Epol_l2/(4*np.pi) ))
    print('octopole:    {:7.3%} (% pol)  (<B^2_oct> {:.2f} G^2)'.format(
        Epol_l3/totEpol, Epol_l3/(4*np.pi) ))
if totEtor > 0.0:
    print('toroidal l1: {:7.3%} (% tor)  (<B^2_tor_l1> {:.2f} G^2)'.format(
        Etor_l1/totEtor, Etor_l1/(4*np.pi) ))
    print('toroidal l2: {:7.3%} (% tor)  (<B^2_tor_l2> {:.2f} G^2)'.format(
        Etor_l2/totEtor, Etor_l2/(4*np.pi) ))
    print('toroidal l3: {:7.3%} (% tor)  (<B^2_tor_l3> {:.2f} G^2)'.format(
        Etor_l3/totEtor, Etor_l3/(4*np.pi) ))

totEaxi = 0.
polEaxi = 0.
torEaxi = 0.
for i in range(magGeom.nTot):
    if(magGeom.m[i] == 0):
        totEaxi += alphaEs[i] + betaEs[i] + gammaEs[i]
        polEaxi += alphaEs[i] + betaEs[i]
        torEaxi += gammaEs[i]
        
print('axisymmetric: {:7.3%} (% tot)  (<B^2_axi> {:.2f} G^2)'.format(
    totEaxi/totE, totEaxi/(4*np.pi) ))
if totEpol > 0.0:
    print('poloidal axisymmetric: {:7.3%} (% pol)  (<B^2_pol_axi> {:.2f} G^2)'.format(
        polEaxi/totEpol, polEaxi/(4*np.pi) ))
if totEtor > 0.0:
    print('toroidal axisymmetric: {:7.3%} (% tor)  (<B^2_tor_axi> {:.2f} G^2)'.format(
        torEaxi/totEtor, torEaxi/(4*np.pi) ))
print('dipole axisymmetric: {:7.3%} (% dip)  (<B^2_dip_axi> {:.2f} G^2)'.format(
    (alphaEs[0] + betaEs[0])/(alphaEs[0] + alphaEs[1] + betaEs[0] + betaEs[1]),
    (alphaEs[0] + betaEs[0])/(4*np.pi) ))


#print('Fraction of magnetic energy in each component')
#print('This can be summed, and should sum to 1.')
#print('l   m    E(alpha)   E(beta)    E(gamma)')
#for i in range(magGeom.nTot):
#    print('%2i %2i %10.5f %10.5f %10.5f' % (magGeom.l[i], magGeom.m[i], alphaEs[i]/totE, betaEs[i]/totE, gammaEs[i]/totE))


### Magnetic helicity density, from Lund et al. 2020 ##########################
# use a magnetic geometry, and an optional l value to limit
# the spherical harmonic expansion
def calcMagHelicityDensity(magGeom, lmax=0):
    """
    Calculate magnetic helicity density, from Lund et al. 2020,
    using a magnetic geometry.
    """
    try:
        from scipy.special import factorial
    except ImportError:  #For some older scipy versions
        from scipy.misc import factorial
    
    if len(magGeom.clat) < 1:
        print('in calcMagHelicityDensity: Need an initialized magGeom. \n'
              + ' (use magGeom.initMagGeom(sGrid.clat, sGrid.long) or '
              + 'SetupMagSphHarmoics(...) first)' )
    if lmax < 1 or lmax > magGeom.nl:
        lmax = magGeom.nl
        
    #Calculate Legendre polynomials and their derivatives
    nCells = len(magGeom.clat)
    cosClat = np.cos(magGeom.clat)
    
    pLegendre = np.zeros((magGeom.nTot, nCells))
    dPdTh = np.zeros((magGeom.nTot, nCells))
    #get a set of array indices corresponding to the upper triangle 
    # of the scipy.special.lpmn arrays, switching the order from (m,l)
    # to (l,m) as used elsewhere, and rejecting the 1st entry
    # (since we don't use the l=0 m=0 associated Legendre polynomial)
    indTri = np.tril_indices(magGeom.nl+1)
    indTri = (indTri[1][1:], indTri[0][1:])
    for i in range(nCells):
         #re-use Legendre values where possible (for same colatitudes)
         if ((cosClat[i] == cosClat[i-1])):
              pLegendre[...,i] = pLegendre[...,i-1]
              dPdTh[...,i] = dPdTh[...,i-1]
         else:
              pLegendreLoc = scipy.special.lpmn(magGeom.nl, magGeom.nl, cosClat[i])
              pLegendre[...,i] = pLegendreLoc[0][indTri]
              dPdTh[...,i] = pLegendreLoc[1][indTri]
    #Renormalize the derivatives so they are dP(x)/d(theta) 
    #rather than dP(x)/dx (where x = cos(theta))
    dPdTh = (dPdTh*(-np.sqrt(1.-cosClat**2)))
    
    expTerm = np.exp(1j*np.outer(magGeom.m,magGeom.lon))
    c_lm = np.sqrt( (2.*magGeom.l + 1.)/(4.*np.pi) * factorial(magGeom.l - magGeom.m)/factorial(magGeom.l + magGeom.m) )
    invSinClat = 1./np.sin(magGeom.clat)

    #From Lund et al. 2020 eq. 22
    #sum(sum'( (alpha*gamma')/((l'+1)*l*(l+1)) * c_lm*c_lm'*expTerm*expTerm' * (Plm*Plm'*(l*(l+1) - m*m'/sin(clat)**2) + [dPlm/dth]*[dPlm'/dth] ) ))
    hd = 0
    n1=0
    for l1 in range(1,lmax+1):
        for m1 in range(l1+1):
            n2=0
            for l2 in range(1,lmax+1):
                for m2 in range(l2+1):
                    hdt = (-magGeom.alpha[n1]*magGeom.gamma[n2])/(l1*(l1+1)*(l2+1)) \
                          *c_lm[n1]*c_lm[n2]*expTerm[n1,:]*expTerm[n2,:] \
                          *(pLegendre[n1,:]*pLegendre[n2,:] \
                            *(l1*(l1+1) - m1*m2*invSinClat**2) \
                            + dPdTh[n1,:]*dPdTh[n2,:] )
                    
                    #hdt = np.real( (-magGeom.alpha[n1]*magGeom.gamma[n2])/float(l1*(l1+1)*(l2+1)) \
                    #      *c_lm[n1]*c_lm[n2]*np.exp(1j*magGeom.lon*float(m1+m2)) \
                    #      *(pLegendre[n1,:]*pLegendre[n2,:] \
                    #        *(float(l1*(l1+1)) - float(m1*m2)/np.sin(magGeom.clat)**2) \
                    #        - dPdTh[n1,:]*dPdTh[n2,:] ) )
                    #There is an extra negative sign since Lund et al. 2020 appear
                    #to define gamma (and beta) with the opposite sign to me.
                    hd += np.real(hdt)
                    n2 += 1
            n1 += 1
    
    return np.real(hd)

#Run the magnetic helicity density calculation
magHelicityDensity = calcMagHelicityDensity(magGeomTot, 4)
meanMagHelDen_hemi = np.sum(magHelicityDensity[indHemi]*sGrid.area[indHemi])/np.sum(sGrid.area[indHemi])
print('mean magnetic helicity density on a hemisphere / Rstar {:.3f} (G^2 i.e. Mx^2/cm^4) \n for l <= 4 (see Lund et al. 2020)'.format(abs(meanMagHelDen_hemi)))
Rsun = 6.9598e10 #cm


### Dipole component field strength ####################################
magGeomDip = magneticGeom.magSphHarmoics(magGeom.nl) #initializes to 0
magGeomDip.initMagGeom(sGrid.clat, sGrid.long)
#Uses dipole (poloidal) components only
for i in range(magGeom.nTot):
    if(magGeom.l[i] == 1): #should be i=0,1
        magGeomDip.alpha[i] = magGeom.alpha[i]
        magGeomDip.beta[i] = magGeom.beta[i]
magVecDip = magGeomDip.getAllMagVectors()
#get magnitudes of vectors
magDip = np.sqrt(magVecDip[0,:]**2 + magVecDip[1,:]**2 + magVecDip[2,:]**2)
    
iMaxDip = np.argmax(magDip)
maxBdip = magDip[iMaxDip]

meanBdip = np.sum(magDip*sGrid.area)/np.sum(sGrid.area)

#print('Dipolar mean = {:9.3f} G'.format(meanBdip))
print('Dipolar max = {:10.3f} G, colat {:6.1f} long {:6.1f}'.format(maxBdip, sGrid.clat[iMaxDip]*180./np.pi, sGrid.long[iMaxDip]*180./np.pi))

#for just the radial component
iMaxDipR = np.argmax(magVecDip[0,:])
maxBdipR = magVecDip[0,iMaxDipR]
iMinDipR = np.argmin(magVecDip[0,:])
minBdipR = magVecDip[0,iMinDipR]
print('Dipole radial + pole = {:10.3f} G, colat {:6.1f} long {:6.1f}'.format(maxBdipR, sGrid.clat[iMaxDipR]*180./np.pi, sGrid.long[iMaxDipR]*180./np.pi))
print('Dipole radial - pole = {:10.3f} G, colat {:6.1f} long {:6.1f}'.format(minBdipR, sGrid.clat[iMinDipR]*180./np.pi, sGrid.long[iMinDipR]*180./np.pi))



###########################################################
#def plotMagHelRect(magHelMap, starGrid):
#    #2D plot of the magnetic helicity density map,
#    #using the same pixels as on the surface of the modle stars
#    #Uses matplotlib, produces a map in colatitude and longitude.
#
#    import matplotlib.pyplot as plt
#
#    xLong = np.zeros(4*starGrid.numPoints)
#    yClat = np.zeros(4*starGrid.numPoints)
#    zMagHel = np.zeros(2*starGrid.numPoints)
#    triangles = np.zeros([2*starGrid.numPoints,3], dtype=int)
#
#    for i in range(starGrid.numPoints):
#        corners =  starGrid.GetCellCorners(i)*(180./np.pi)
#
#        xLong[i*4] = corners[0,0]
#        xLong[i*4+1] = corners[1,0]
#        xLong[i*4+2] = corners[2,0]
#        xLong[i*4+3] = corners[3,0]
#
#        yClat[i*4] = corners[0,1] 
#        yClat[i*4+1] = corners[1,1]
#        yClat[i*4+2] = corners[2,1]
#        yClat[i*4+3] = corners[3,1]
#
#        triangles[2*i,:]   = [i*4, i*4+1, i*4+3]
#        triangles[2*i+1,:] = [i*4, i*4+2, i*4+3]
#        
#        zMagHel[2*i] = magHelMap[i]
#        zMagHel[2*i+1] = magHelMap[i]
#
#    w, h = plt.figaspect(0.5)
#    fig = plt.figure(figsize=(w, h), tight_layout=True)
#    ax = fig.add_subplot(111)
#    #plt.tripcolor(xLong, yClat, triangles, facecolors=zMagHel, edgecolors='none', cmap='copper')
#    scaleHel = np.max(np.abs(magHelMap))
#    helPlt = ax.tripcolor(xLong, yClat, triangles, facecolors=zMagHel, edgecolors='none', cmap='RdYlBu', vmin=-scaleHel, vmax=scaleHel)
#    # use edgecolors='k' to show triangle edges, cmap= 'afmhot', 'hot', 'copper'
#    ax.axis([xLong.min(), xLong.max(), yClat.max(), yClat.min()])
#    ax.set_aspect('equal') #1.0)#, adjustable='box')
#    fig.colorbar(helPlt, ax=ax, label='magnetic helicity density')
#    ax.set_xlabel('longitude')
#    ax.set_ylabel('colatitude')
#    ax.set_title('Magnetic Helicity Density')
#    #plt.tight_layout()
#    plt.show()
#
################################
#plotMagHelRect(magHelicityDensity, sGrid)
