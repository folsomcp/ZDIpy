#Notes:
#Spherical harmonic calculations of (X,Y,Z) moved to an initialization step.  However this does not allow (conveniently) for different stellar grids at different rotation phases/observations.  
#Is it possible to efficiently compute an array of associated Legendre polynomials for l and m up to l_max and m_max, using a (stable) recursive definition of the associated Legendre polynomial?  (Now (probably) doing this with scipy.special.lpmn)

import numpy as np
import scipy.special
try:
     from scipy.special import factorial
except ImportError:  #For some older scipy versions
     from scipy.misc import factorial


#Saves a set of spherical harmonic coefficients for magnetic fields,
# and has functions to return magnetic vectors based on those coefficients
class magSphHarmoics:
     def __init__(self, nHarmics):

          self.nl = nHarmics
          #for a given l, m goes from 0 to l
          #so the total number of values is: sum of i+1 from i=1 to nl
          self.nTot = self.nl*(self.nl + 1)//2 + self.nl
          
          #setup arrays of l and m values, to go with the subsequent arrays of coefficients
          self.l = np.zeros(self.nTot)
          self.m = np.zeros(self.nTot)
          index=0
          for i in range(self.nl):
               for j in range(i + 2):
                    self.l[index] = i + 1
                    self.m[index] = j
                    index += 1

          #initialize coefficients alpha, beta, and gamma.
          #actual values must be set elsewhere
          self.magGeomType = 'full'
          self.alpha = np.zeros(self.nTot, dtype=complex)
          self.beta = np.zeros(self.nTot, dtype=complex)
          self.gamma = np.zeros(self.nTot, dtype=complex)

          #XTerm, YTerm, and ZTerm (and clat, lon) are properly initialized by initMagGeom
          #This is done so that alpha, beta and gamma can be stored 
          #even if we don't care about the stellar geometry.  
          self.clat=[]
          self.lon=[]
          self.XTerm = []
          self.YTerm = []
          self.ZTerm = []

     def setMagGeomType(self, magGeomType):
          #Set the magnetic geometry type flag to one of the allowed values.
          #Error and exit if this does not receive an allowed value.
          test = magGeomType.lower()
          if not(test == 'full' or test == 'poloidal'
                 or test == 'pottor' or test == 'potential'):
               print(('ERROR: read an unrecognized magnetic geometry type ({:}).  '
                      +'Use one of: Full, Poloidal, PotTor, Potential').format(
                           self.magGeomType) )
               import sys
               sys.exit()
          else:
               self.magGeomType = test
               #Generally the type should be set before coefficient values 
               #are set, but just in case modify the coefficient accordingly.
               if self.magGeomType == 'poloidal':
                    self.gamma[:] = 0.0
               elif self.magGeomType == 'pottor':
                    self.beta = self.alpha
               elif self.magGeomType == 'potential':
                    self.beta = self.alpha
                    self.gamma[:] = 0.0
               
          
     def initMagGeom(self, clat, lon):
          
          nCells = len(clat)
          cosClat = np.cos(clat)
          self.clat = clat
          self.lon = lon
          
          #Donati et al 2006 write P_l,m(clat), but spherical harmonics actually use P_l,m(cos(clat)) 
          #i.e. the cos is not built into the typical definition of the associated Legendre polynomial.

          ##Simple but slower method:
          #cosClat_all = np.tile(cosClat, (self.nTot,1))  #for Legendre (nlm, nCells)
          #l_all = np.tile(self.l, (nCells, 1)).T #for Legendre (nCells, nlm)
          #m_all = np.tile(self.m, (nCells, 1)).T #for Legendre (nCells, nlm)
          #pLegendre_all = scipy.special.lpmv(m_all, l_all, cosClat_all)
          #dPdTh_all = dLegendre_dTheta(l_all, m_all, cosClat_all)

          #More complicated but ~10x more efficient calculation:
          pLegendre_all = np.zeros((self.nTot,nCells))
          dPdTh_all = np.zeros((self.nTot,nCells))
          #get a set of array indices corresponding to the upper triangle 
          # of the scipy.special.lpmn arrays, switching the order from (m,l)
          # to (l,m) as used elsewhere, and rejecting the 1st entry
          # (since we don't use the l=0 m=0 associated Legendre polynomial)
          indTri = np.tril_indices(self.nl+1)
          indTri = (indTri[1][1:], indTri[0][1:])
          for i in range(nCells):
               #re-use Legendre values where possible (for same colatitudes)
               if ((cosClat[i] == cosClat[i-1])):
                    pLegendre_all[...,i] = pLegendre_all[...,i-1]
                    dPdTh_all[...,i] = dPdTh_all[...,i-1]
               else:
                    pLegendre = scipy.special.lpmn(self.nl, self.nl, cosClat[i])
                    pLegendre_all[...,i] = pLegendre[0][indTri]
                    dPdTh_all[...,i] = pLegendre[1][indTri]
          #Renormalize the derivatives so they are dP(x)/d(theta) 
          #rather than dP(x)/dx (where x = cos(theta))
          dPdTh_all = (dPdTh_all*(-np.sqrt(1.-cosClat**2)))

          expTerm_all = np.exp(1j*np.outer(self.m,lon))

          cTerm_lm = np.sqrt( (2.*self.l + 1.)/(4.*np.pi) * factorial(self.l - self.m)/factorial(self.l + self.m) )
          c2Term_lm = cTerm_lm/(self.l + 1.)
          c3Term_lm = 1j*c2Term_lm*self.m
          invSinClat = 1./np.sin(clat)
          
          #YTerm = cTerm*pLegendre*np.exp(1j*self.m*lon)
          self.YTerm = cTerm_lm[:,np.newaxis]*pLegendre_all*expTerm_all

          #XTerm = cTerm/(self.l + 1.)*pLegendre/np.sin(clat)*1j*self.m*np.exp(1j*self.m*lon)
          self.XTerm = c3Term_lm[:,np.newaxis]*invSinClat[np.newaxis,:]*pLegendre_all*expTerm_all

          #ZTerm = cTerm/(self.l + 1.)*dPdTh*np.exp(1j*self.m*lon)
          self.ZTerm = c2Term_lm[:,np.newaxis]*dPdTh_all*expTerm_all


     def getAllMagVectors(self):
          #Returns an array of magnetic vectors in spherical coordinates, 
          # for positions in colatitude and longitude given to initMagGeom.
          #Requires initMagGeom to have been called.


          #Matching Pascal Petit's definitions, for axes of colatitude and rotation phase
          #Br = np.real(np.sum(self.alpha*YTerm))
          #Bclat = np.real(np.sum(self.beta*ZTerm - self.gamma*XTerm))  #Btheta
          #Blon = np.real(np.sum(self.beta*XTerm + self.gamma*ZTerm))  #Bphi

          #Matching JF Donati's line profiles, making some unusual assumptions:
          #for left handed longitude in Donati's equations and -ve latitude vs colatitude term
          #Br = np.real(np.sum(self.alpha*YTerm))
          #Bclat -ve of Pascal and what is codded in Donati's ZDI, because those both use 
          #latitude internally as opposed to colatitude, and the direction of increasing latitude 
          #is opposite to my direction of increasing colatitude. Therefore the +ve Bclat 
          #direction is in the opposite direction, hence the -ve sign here.  
          #Bclat = -np.real(np.sum(self.beta*ZTerm - self.gamma*XTerm))
          #Blon for left handed longitude in Donati's equations
          #Blon = -np.real(np.sum(self.beta*XTerm + self.gamma*ZTerm)) 

          #Matching JF Donati's line profiles, using the complex conjugates of 
          #alpha, beta and gamma, and thereby getting closest to the published equations.
          #This is closest to what appears in JF Donati ZDI code, but he has a +ve clat term 
          #due to Donati using latitude increasing in the opposite direction to the 
          #colatitude used here

          Br = np.real(np.dot(self.alpha, self.YTerm))
          Bclat = -np.real(np.dot(self.beta, self.ZTerm) + np.dot(self.gamma, self.XTerm)) 
          Blon = -np.real(np.dot(self.beta, self.XTerm) - np.dot(self.gamma, self.ZTerm))

          vMagAll = np.array([Br, Bclat, Blon])
          return vMagAll


     def getAllMagDerivs(self):
          #Returns an array of derivatives of B with respect to the spherical harmonic 
          #coefficients at positions in colatitude and longitude given to initMagGeom.
          #Only calculate for components that are used in the fit
          #based on the magGeomType flag (other components are just set to 0).
          #Requires initMagGeom to have been called.

          dBr_dAlpha = np.conj(self.YTerm)
          if self.magGeomType == 'full': #(free alpha, beta gamma)
               dBclat_dBeta = -np.conj(self.ZTerm)
               dBclat_dGamma = -np.conj(self.XTerm)
               dBlon_dBeta = -np.conj(self.XTerm)
               dBlon_dGamma = np.conj(self.ZTerm)
               derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dBeta, dBclat_dGamma,
                                             dBlon_dBeta, dBlon_dGamma), axis=0)
               derivMagAll = derivMagAll.reshape((5, self.nTot, self.YTerm.shape[1]))
          elif self.magGeomType == 'poloidal': #(gamma = 0)
               dBclat_dBeta = -np.conj(self.ZTerm)
               dBlon_dBeta = -np.conj(self.XTerm)
               derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dBeta, 
                                             dBlon_dBeta), axis=0)
               derivMagAll = derivMagAll.reshape((3, self.nTot, self.YTerm.shape[1]))
          elif self.magGeomType == 'pottor': #(beta = alpha)
               dBclat_dAlpha = -np.conj(self.ZTerm)
               dBclat_dGamma = -np.conj(self.XTerm)
               dBlon_dAlpha = -np.conj(self.XTerm)
               dBlon_dGamma = np.conj(self.ZTerm)
               derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dAlpha, dBclat_dGamma,
                                             dBlon_dAlpha, dBlon_dGamma), axis=0)
               derivMagAll = derivMagAll.reshape((5, self.nTot, self.YTerm.shape[1]))
          elif self.magGeomType == 'potential': #(beta = alpha & gamma = 0)
               dBclat_dAlpha = -np.conj(self.ZTerm)
               dBlon_dAlpha = -np.conj(self.XTerm)
               derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dAlpha, dBlon_dAlpha))
               derivMagAll = derivMagAll.reshape((3, self.nTot, self.YTerm.shape[1]))

          return derivMagAll


     def getAllMagVectorsCart(self):
          #Returns an array of magnetic vectors in Cartesian coordinates, 
          #for the positions in colatitude and longitude given to initMagGeom
          #Requires initMagGeom to have been called.

          sinClat = np.sin(self.clat)
          cosClat = np.cos(self.clat)
          sinLon = np.sin(self.lon)
          cosLon = np.cos(self.lon)

          #First get the magnetic vector in spherical coordinates
          vecB = self.getAllMagVectors()
          Br = vecB[0,:]
          Bclat = vecB[1,:]
          Blon = vecB[2,:]

          #Then convert the magnetic vector from spherical to Cartesian coordinates
          # vecB = Br*r^ + Btheta*theta^ + Bphi*phi^   (where x^ denotes a unit vector)
          # and in cartesian coordinates [x^, y^, z^]  (note: theta=clat, phi=long)
          # r^ = sin(theta)cos(phi)*x^ + sin(theta)sin(phi)*y^ + cos(theta)*z^
          # theta^ = cos(theta)cos(phi)*x^ + cos(theta)sin(phi)*y^ - sin(theta)*z^
          # phi^ = -sin(phi)*x^ + cos(phi)*y^
          # then the x^, y^ and z^ components of B_vec are:
          Bx = Br*sinClat*cosLon + Bclat*cosClat*cosLon - Blon*sinLon
          By = Br*sinClat*sinLon + Bclat*cosClat*sinLon + Blon*cosLon
          Bz = Br*cosClat - Bclat*sinClat
          Bcart = np.concatenate((Bx[np.newaxis], By[np.newaxis], Bz[np.newaxis]), axis=0)

          return Bcart


     def getAllMagDerivsCart(self):
          #Returns an array of derivatives of B with respect to spherical harmonic coefficients,
          #in Cartesian coordinates, for the positions given to initMagGeom
          #Requires initMagGeom to have been called.

          sinClat = np.sin(self.clat)
          cosClat = np.cos(self.clat)
          sinLon = np.sin(self.lon)
          cosLon = np.cos(self.lon)

          dB = self.getAllMagDerivs()

          #Unpack the array of derivatives in spherical coordinates
          dBr_dAlpha    = dB[0,...]
          if self.magGeomType == 'full':
               dBclat_dBeta  = dB[1,...]
               dBclat_dGamma = dB[2,...]
               dBlon_dBeta   = dB[3,...]
               dBlon_dGamma  = dB[4,...]
          elif self.magGeomType == 'poloidal':
               dBclat_dBeta  = dB[1,...]
               dBlon_dBeta   = dB[2,...]
          elif self.magGeomType == 'pottor':
               dBclat_dAlpha = dB[1,...]
               dBclat_dGamma = dB[2,...]
               dBlon_dAlpha  = dB[3,...]
               dBlon_dGamma  = dB[4,...]
          elif self.magGeomType == 'potential':
               dBclat_dAlpha = dB[1,...]
               dBlon_dAlpha  = dB[2,...]
          
          #Convert from spherical coordinates to Cartesian coordinates
          if self.magGeomType == 'full':
               dBx_dAlpha = dBr_dAlpha*(sinClat*cosLon)[np.newaxis,:]
               dBy_dAlpha = dBr_dAlpha*(sinClat*sinLon)[np.newaxis,:]
               dBz_dAlpha = dBr_dAlpha*cosClat[np.newaxis,:]
               dBx_dBeta  = dBclat_dBeta*(cosClat*cosLon)[np.newaxis,:] \
                            - dBlon_dBeta*sinLon[np.newaxis,:]
               dBy_dBeta  = dBclat_dBeta*(cosClat*sinLon)[np.newaxis,:] \
                            + dBlon_dBeta*cosLon[np.newaxis,:]
               dBz_dBeta  = -dBclat_dBeta*sinClat[np.newaxis,:]
               dBx_dGamma = dBclat_dGamma*(cosClat*cosLon)[np.newaxis,:] \
                            - dBlon_dGamma*sinLon[np.newaxis,:]
               dBy_dGamma = dBclat_dGamma*(cosClat*sinLon)[np.newaxis,:] \
                            + dBlon_dGamma*cosLon[np.newaxis,:]
               dBz_dGamma = -dBclat_dGamma*sinClat[np.newaxis,:]
          elif self.magGeomType == 'poloidal':
               dBx_dAlpha = dBr_dAlpha*(sinClat*cosLon)[np.newaxis,:]
               dBy_dAlpha = dBr_dAlpha*(sinClat*sinLon)[np.newaxis,:]
               dBz_dAlpha = dBr_dAlpha*cosClat[np.newaxis,:]
               dBx_dBeta  = dBclat_dBeta*(cosClat*cosLon)[np.newaxis,:] \
                            - dBlon_dBeta*sinLon[np.newaxis,:]
               dBy_dBeta  = dBclat_dBeta*(cosClat*sinLon)[np.newaxis,:] \
                            + dBlon_dBeta*cosLon[np.newaxis,:]
               dBz_dBeta  = -dBclat_dBeta*sinClat[np.newaxis,:]
          elif self.magGeomType == 'pottor':
               dBx_dAlpha = dBr_dAlpha*(sinClat*cosLon)[np.newaxis,:] \
                            + dBclat_dAlpha*(cosClat*cosLon)[np.newaxis,:] \
                            - dBlon_dAlpha*sinLon[np.newaxis,:]
               dBy_dAlpha = dBr_dAlpha*(sinClat*sinLon)[np.newaxis,:] \
                            + dBclat_dAlpha*(cosClat*sinLon)[np.newaxis,:] \
                            + dBlon_dAlpha*cosLon[np.newaxis,:]
               dBz_dAlpha = dBr_dAlpha*cosClat[np.newaxis,:] \
                            - dBclat_dAlpha*sinClat[np.newaxis,:]
               dBx_dGamma = dBclat_dGamma*(cosClat*cosLon)[np.newaxis,:] \
                            - dBlon_dGamma*sinLon[np.newaxis,:]
               dBy_dGamma = dBclat_dGamma*(cosClat*sinLon)[np.newaxis,:] \
                            + dBlon_dGamma*cosLon[np.newaxis,:]
               dBz_dGamma = -dBclat_dGamma*sinClat[np.newaxis,:]
          elif self.magGeomType == 'potential':
               dBx_dAlpha = dBr_dAlpha*(sinClat*cosLon)[np.newaxis,:] \
                            + dBclat_dAlpha*(cosClat*cosLon)[np.newaxis,:] \
                            - dBlon_dAlpha*sinLon[np.newaxis,:]
               dBy_dAlpha = dBr_dAlpha*(sinClat*sinLon)[np.newaxis,:] \
                            + dBclat_dAlpha*(cosClat*sinLon)[np.newaxis,:] \
                            + dBlon_dAlpha*cosLon[np.newaxis,:]
               dBz_dAlpha = dBr_dAlpha*cosClat[np.newaxis,:] \
                            - dBclat_dAlpha*sinClat[np.newaxis,:]


          #dBcart has dimensions of (x-y-z, alpha-beta-gamma, harmonic coefficient, surface element)
          #Joining the arrays here is rather slow, a single concatenation 
          #follows by reshaping is an attempt to be as efficient as possible.
          if self.magGeomType == 'full':
               dBcart = np.concatenate((dBx_dAlpha, dBx_dBeta, dBx_dGamma, \
                                        dBy_dAlpha, dBy_dBeta, dBy_dGamma, \
                                        dBz_dAlpha, dBz_dBeta, dBz_dGamma))
               dBcart = np.reshape(dBcart, (3, 3, dB.shape[1], dB.shape[2]))
          elif self.magGeomType == 'poloidal':
               dBcart = np.concatenate((dBx_dAlpha, dBx_dBeta, \
                                        dBy_dAlpha, dBy_dBeta, \
                                        dBz_dAlpha, dBz_dBeta))
               dBcart = np.reshape(dBcart, (3, 2, dB.shape[1], dB.shape[2]))
          elif self.magGeomType == 'pottor':
               dBcart = np.concatenate((dBx_dAlpha, dBx_dGamma, \
                                        dBy_dAlpha, dBy_dGamma, \
                                        dBz_dAlpha, dBz_dGamma))
               dBcart = np.reshape(dBcart, (3, 2, dB.shape[1], dB.shape[2]))
          elif self.magGeomType == 'potential':
               dBcart = np.concatenate((dBx_dAlpha, dBy_dAlpha, dBz_dAlpha))
               dBcart = np.reshape(dBcart, (3, 1, dB.shape[1], dB.shape[2]))

          return dBcart


     def saveToFile(self, fName, compatibility = False):
          #Save the magnetic geometry, parameterized by the 
          #spherical harmonic coefficients, to the specified file.
          #Uses the same format as JDF's ZDI program.
          #Use compatibility = True for exact compatibility with Donati's format,
          #useful for other programs that take that as input
          #The 1st line is a comment. The 2nd line is number of l and m combinations,
          # the number of alpha-beta-gamma value types (always 3 here),
          # and a flag for the type of reconstruction (always -30 here)
          # Note: Donati's program actually uses the complex conjugate of the 
          # coefficients used here. For perfect agreement with programs 
          # based on that code, take the complex conjugate of these values and 
          # set the type of reconstruction flag to -3
          fOut = open(fName, 'w')
          if(compatibility == False):  #Similar format to JFD's
               fOut.write('ZDIpy: general poloidal plus toroidal field\n')
               fOut.write('%i %i %i\n' % (self.nTot, 3, -30))
               for i in range(self.nTot):
                    fOut.write('%2i %2i %14e %14e\n' % (self.l[i],self.m[i], \
                              np.real(self.alpha[i]), np.imag(self.alpha[i])))
               fOut.write('\n')
               for i in range(self.nTot):
                    fOut.write('%2i %2i %14e %14e\n' % (self.l[i],self.m[i], \
                              np.real(self.beta[i]), np.imag(self.beta[i])))
               fOut.write('\n')
               for i in range(self.nTot):
                    fOut.write('%2i %2i %14e %14e\n' % (self.l[i],self.m[i], \
                              np.real(self.gamma[i]), np.imag(self.gamma[i])))
               fOut.write('\n')
          #Identical format to JFD's, compatible with programs using that as input
          elif(compatibility == True):
               fOut.write('General poloidal plus toroidal field\n')
               fOut.write('%i %i %i\n' % (self.nTot, 3, -3))
               for i in range(self.nTot):
                    fOut.write('%2i %2i %13e %13e\n' % (self.l[i],self.m[i], \
                              np.real(self.alpha[i]), -np.imag(self.alpha[i])))
               fOut.write('\n')
               for i in range(self.nTot):
                    fOut.write('%2i %2i %13e %13e\n' % (self.l[i],self.m[i], \
                              np.real(self.beta[i]), -np.imag(self.beta[i])))
               fOut.write('\n')
               for i in range(self.nTot):
                    fOut.write('%2i %2i %13e %13e\n' % (self.l[i],self.m[i], \
                              np.real(self.gamma[i]), -np.imag(self.gamma[i])))
               fOut.write('\n')
          else: print('File writing error, unknown type')


def dLegendre_dTheta(vl, vm, cosTheta):
     #assumes you are passing cos(theta), 
     #returns dP(l,m)(cos(theta))/dtheta, where P(l,m) is the associated Legendre polynomial.
     #identity taken from mathworld.wolfram.com  Find a better reference?
    
     Plm = scipy.special.lpmv(vm, vl, cosTheta)
     Pl2m = scipy.special.lpmv(vm, vl - 1., cosTheta)
     dPdTh = (vl*cosTheta*Plm - (vl+vm)*Pl2m)/np.sqrt(1.-cosTheta**2)
     # if (cosTheta >= 1.) | (cosTheta <= -1.):
     #      print('ERROR in dLegendre_dTheta: value of cosTheta >= 1 or <= 1. Setting to 0.')
     #      dPdTh = 0.

     # alternately use dPdTheta formulation from mappot_rect_inc1.2.c
     # Plmplus1 = scipy.special.lpmv(vm+1, vl, cosTheta)
     # dPdTh2 = cosTheta/np.sqrt(1. - cosTheta**2)*vm*Plm + Plmplus1

     return dPdTh


def magSphHarmoicsFromFile(fname, lmax=0, verbose=1):
     #Read in magnetic spherical harmonic coefficients from a file, 
     # and return the magSphHarmoics object with those values.
     #Assumes J-F Donati's output coefficient file format.

     inFile = open(fname,'r')
    
     comment = inFile.readline()
     line = inFile.readline()
     nValues = int(line.split()[0])
     nBlocks = int(line.split()[1])
     #nPotential is a ZDI(Donati) flag for the type of magnetic field used,
     # here it is used to check if we are reading ZDI coefficients output by Donati, 
     # which need to be treated differently to work with the equations for Br, Bclat and Blong
     nPotential = int(line.split()[2])  

     #calculate the number of l values for the given total number of values
     #(i.e. solve nTot = l*(l+1)/2+l for l; => roots in l of: 0 = l**2 + 3*l - 2*nTot )
     nl = int(0.5*(np.sqrt(9+8*nValues)-3))

     #initialize the set of spherical harmonic coefficients
     if(lmax == 0):
          lmax = nl
     mSphHar = magSphHarmoics(lmax)

     #read in and save the coefficients
     for i in range(nValues):
          line = inFile.readline()
          if(i < mSphHar.nTot):
               #reading l and m is redundant, if l and m are ordered properly
               mSphHar.l[i] = int(line.split()[0])  
               mSphHar.m[i] = int(line.split()[1])
               mSphHar.alpha[i] = complex(float(line.split()[2]), float(line.split()[3]))
    
     inFile.readline()
     for i in range(nValues):
          line = inFile.readline()
          if(i < mSphHar.nTot):
               mSphHar.beta[i] = complex(float(line.split()[2]), float(line.split()[3]))
    
     inFile.readline()
     for i in range(nValues):
          line = inFile.readline()
          if(i < mSphHar.nTot):
               mSphHar.gamma[i] = complex(float(line.split()[2]), float(line.split()[3]))

     #In JF Donati's ZDI code, alpha beta and gamma are read in as real and imaginary parts, 
     # but the imaginary part is treated as -ve of the imaginary part in calculating Br Btheta and Bphi.
     # Specifically, in the terms like Br = alpha*Y = alpha*P_lm(theta)*exp(i*m*phi) that code uses:
     # real(alpha)*real(Y) + imag(alpha)*imag(Y)
     # = real(alpha)*P_lm(theta)*cos(m*phi) + imag(alpha)**P_lm(theta)*sin(m*phi)
     #where as this should actually be:
     # real(alpha)*real(Y) - imag(alpha)*imag(Y)    
     #  (The real part of the product of two complex numbers)
     # = real(alpha)*P_lm(theta)*cos(m*phi) - imag(alpha)**P_lm(theta)*sin(m*phi)
     #Consequently we take the complex conjugate of alpha beta and gamma 
     # if we are using coefficients output by Donati's ZDI code (in potential = -3 mode). 
     if (nPotential == -3):
          mSphHar.alpha = np.conj(mSphHar.alpha)
          mSphHar.beta = np.conj(mSphHar.beta)
          mSphHar.gamma = np.conj(mSphHar.gamma)
          if(verbose == 1):
               print('Treating spherical harmonics in Donati\'s "-3" ZDI fashion')
        
     return mSphHar


def magSphHarmoicsFromMean(lMax, Binit):
     #Misleading name/not fully fully functional (the mean field is not actually Binit)
     #need some description of the distribution of power in harmonics 
     #(e.g. proportional to 1/l, or lmax-l)

     mSphHar = magSphHarmoics(lMax)

     coeffBinit = Binit/float(mSphHar.nTot)
     for i in range(mSphHar.nTot):
          mSphHar.alpha[i] = coeffBinit*(1.+1.j)
          mSphHar.beta[i] = coeffBinit*(1.+1.j)
          mSphHar.gamma[i] = coeffBinit*(1.+1.j)

     return mSphHar


def SetupMagSphHarmoics(sGrid, initMagFromFile, initMagGeomFile, lMax, magGeomType='full', verbose=1):
     #initialize the magnetic geometry spherical harmonics from a file of coefficients, or to a constant value (of 0).
     #Here the magGeomType flag is only used to limit what derivatives are calculated.
     #The alpha beta and gamma values all still exist and are used to calculate 
     #magnetic vectors (even if some are not free parameters in the fit).
     if (initMagFromFile == 1):
          magGeom = magSphHarmoicsFromFile(initMagGeomFile, lMax, verbose)
     else:
          magGeom = magSphHarmoicsFromMean(lMax, 0.0)
     
     magGeom.setMagGeomType(magGeomType)
     #Save the stellar grid into magnetic geometry, and calculate spherical harmonics
     magGeom.initMagGeom(sGrid.clat, sGrid.long)

     return magGeom
