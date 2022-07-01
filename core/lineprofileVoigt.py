#Contains functions for generating the model line profile.
#Mostly intended for a Voigt profile in the weak field approximation.
import numpy as np

#The convolution with an instrumental profile can either be done explicitly,
#once disk integrated profiles have been calculated, or it can be included in
#the local line profile if the local profile is a Voigt (or a Gaussain).
#Including the instrumental profile in the local line profile is more efficient 
#and accurate, but may not be valid for more sophisticated line models.
explicitConvolution = False

class lineData:
     def __init__(self, inFileName, instRes = -1.):
          #Read in model line profile data from inFileName, and store it as part of the lineData object
          #lines beginning with a # are ignored (i.e. treated as comments)
          #Warning: currently the local line profile only uses the first (non-comment) line of line data
          # that should be pretty easy to expand later for multi-line profiles/spectra 
          self.wl0 = np.array([])
          self.str = np.array([])
          self.widthGauss = np.array([])
          self.widthLorentz = np.array([])
          self.g = np.array([])
          self.limbDark = np.array([])
          self.gravDark = np.array([])
          self.numLines = 0
          self.instRes = instRes  #The instrumental resolution R

          inFile = open(inFileName, 'r')
          for line in inFile:
               #check for comments (ignoring white-space)
               if (line.strip()[0] != '#'):
                    self.numLines += 1
                    #central wavelength of the line (in nm)
                    self.wl0 = np.append(self.wl0, [float(line.split()[0])])
                    #line strength (depth) of the line
                    self.str = np.append(self.str, [float(line.split()[1])])
                    #line gaussian Doppler width (sqrt(2)*sigma ) in velocity units
                    self.widthGauss = np.append(self.widthGauss, [float(line.split()[2])])
                    #line Lorenzian width factor = gamma / (sqrt(2)*sigma) = gamma/widthGauss
                    #where Lorenzian = gamma/(pi*(x^2 + gamma^2))  (and gamma is 1/2*FWHM of a Lorentzian)
                    self.widthLorentz = np.append(self.widthLorentz, [float(line.split()[3])])
                    #effective Lande g factor for the line
                    self.g = np.append(self.g, [float(line.split()[4])])
                    #limb darkening coefficient for the star
                    self.limbDark = np.append(self.limbDark, [float(line.split()[5])])
                    #gravity darkening coefficient for the star,
                    #only used for oblate stars, optionally can
                    #be set to 0 for no gravity darkening
                    try:
                         self.gravDark = np.append(self.gravDark, [float(line.split()[6])])
                    except ValueError: #If the value is missing, set it to 0
                         self.gravDark = np.append(self.gravDark, [0.0])
                    except IndexError:
                         self.gravDark = np.append(self.gravDark, [0.0])
                         


######################################
def limbDarkening(coeffEta, angle):
     #calculate the limb darkening using the supplied limb darkening coefficient 
     #at the given angle of the line-of-sight to the normal of the surface element
     #Based in the linear limb darkening law from Gray 2005, eq. 17.11
     limbD = 1. - coeffEta + coeffEta*np.cos(angle)
     return limbD
        

######################################
######################################

class localProfileAndDeriv:
     def __init__(self, lineData, numPts, wlGrid):

          #calculate a local line profile for a surface element
          #used later for disk integration to get a full observed line profile
          #the line is calculated in wavelength units here, and shifted to velocity later if needed
          
          c = 2.99792458e5 #speed of light in km/s

          self.wl = wlGrid
          self.Q = np.zeros(numPts) #not currently used
          self.U = np.zeros(numPts) #not currently used

          #Combine the local Gaussian and instrumental widths
          #into one Gaussian width, if necessary.
          if explicitConvolution == False and lineData.instRes > 0.0:
               #The instrumental resolution is a FWHM in lambda/delta lamba (= c/dv)
               #the local gaussain width is a velocity in sqrt(2)*sigma
               #velInstRes = c / lineData.instRes / (2.*np.sqrt(np.log(2.)))
               velInstRes = c / lineData.instRes * 0.6005612043932249
               widthGauss = np.sqrt(lineData.widthGauss[0]**2 + velInstRes**2)
               #Preserve the Lorentzian width relative to the local Gaussian width
               widthLorentz = lineData.widthLorentz[0]*(lineData.widthGauss[0]/widthGauss)
               #Preserve the line strength relative to the local Gaussian area
               lineStr = lineData.str[0]*(lineData.widthGauss[0]/widthGauss)
          elif explicitConvolution == False and lineData.instRes < 0.0:
               print('Warning: convolution with an instrumental profile may not be performed')
               widthGauss = lineData.widthGauss[0]
               widthLorentz = lineData.widthLorentz[0]
               lineStr = lineData.str[0]
          else:
               widthGauss = lineData.widthGauss[0]
               widthLorentz = lineData.widthLorentz[0]
               lineStr = lineData.str[0]
          
          #calculate some quantities that should be independent of brightness and magnetic field.

          # The form of the Voigt profile function used here is taken from
          # Humlicek 1982, J. Quant. Spec. Rad. Trans., vol 27, 437. It is
          # claimed to be accurate to 10^-4 relatively everywhere. The real 
          # part of w4 is the Voigt profile, and the imaginary part is twice the 
          # Faraday-Voigt function. 
          
          w4 = np.zeros(self.wl.shape, dtype=complex)

          widthGaussWl = (widthGauss/c*lineData.wl0[0])
          wlGaussNorm = (lineData.wl0[0] - self.wl)/widthGaussWl  #distance from line center in gaussian widths
          
          #Note on interpretation of the Lorentzian width: e.g. from Gray eq. 11.47,
          #total gamma, normalized by Gaussian width *4pi, but in frequency.
          #Since the gamma damping coefficients are in frequency, and in the opacity profile gamma
          #is divided by 4pi (for an angular frequency, e.g. eq. 11.7).
          #widthLorentz = lineData.widthLorentz[0]
          
          z = widthLorentz - 1j*wlGaussNorm
          zz = z*z
          s = np.abs(wlGaussNorm) + widthLorentz

          con1 = np.where(s >= 15.0)
          zt = z[con1]
          w4[con1] = 0.56418958355*zt/(0.5 + zt*zt)
          
          con2 = np.where((s >= 5.5)&(s < 15.0))
          zt = z[con2]
          zzt = zz[con2]
          w4[con2] = zt*(1.4104739589 + 0.56418958355*zzt) / \
                     ((3.0 + zzt)*zzt + 0.7499999999)
          
          con3 = np.where( (widthLorentz >= 0.195*np.abs(wlGaussNorm) - 0.176)&(s < 5.5))
          zt = z[con3]
          w4[con3] = (16.4954955 + zt*(20.2093334 + \
                    zt*(11.9648172 + zt*(3.77898687 + \
                    zt*0.564223565))))/(16.4954955 + \
                    zt*(38.8236274 + zt*(39.2712051 + \
                    zt*(21.6927370 + zt*(6.69939801 + zt)))))
          
          con4 = np.where(w4 == 0.+0j) 
          zt = z[con4]
          zzt = zz[con4]
          w4[con4] = np.exp(zzt) - zt*(36183.30536 - zzt* \
                    (3321.990492 - zzt*(1540.786893 - zzt* \
                    (219.0312964 - zzt*(35.76682780 - zzt* \
                    (1.320521697 - zzt*0.5641900381))))))/ \
                    (32066.59372 - zzt*(24322.84021 - zzt* \
                    (9022.227659 - zzt*(2186.181081 - zzt* \
                    (364.2190727 - zzt*(61.57036588 - zzt* \
                    (1.841438936 - zzt)))))))

          #From Landi & Landofi 2005, eq 5.58
          #dHdv = (2.*(-v*w4.real + anm*w4.imag))
          #so dH/dLambda = (2.*(-v*w4.real + anm*w4.imag))/widthGaussWl  [since v = wlGaussNorm]
          dVoigtdWl = (lineStr/widthGaussWl*2.)*(-wlGaussNorm*w4.real + widthLorentz*w4.imag)

          self.Iunscaled = 1.0 - lineStr*w4.real

          #The constant e*/(4*pi*m_e*c^2) = -4.6686E-12 in cgs units, for wavelength in nm.
          gCoeff = -4.6686E-12*lineData.wl0[0]**2*lineData.g[0]  
          self.dIdWlG = gCoeff*dVoigtdWl  
          
          
     def updateProfDeriv(self, lineData, Blos, dBlos_d, brightMap, numPts, wlGrid, surfaceScale, calcDI, calcDV):
          # Calculates the actual line profiles, and updates them based on changes to the brightness map and magnetic map.  
          # Other changed parameters will generally require re-initializing localProfileAndDeriv.  

          #Includes limb darkening and projected area (could include other effects too)
          contin = surfaceScale*brightMap.bright
          self.I = contin[np.newaxis,:]*self.Iunscaled
          self.V = (Blos*contin)[np.newaxis,:]*self.dIdWlG

          self.Ic = np.sum(contin)  #since Ic is the same for all points across the line
          invSumIc0 = 1./np.sum(self.Ic)

          #sum over surface elements and normalize by continuum
          self.I = np.sum(self.I, axis=1)*invSumIc0
          self.V = np.sum(self.V, axis=1)*invSumIc0          

          #derivative of V wrt alpha, beta, gamma coefficients
          #Disk integrating (summing over disk elements) is more efficient if done here.
          #The major limitation here seems to be efficiently handling memory management.
          if (calcDV == 1):
               # Using einsum  is similar speed or faster but uses less code.
               # This ordering of indices in the einsum is slightly faster, by 20-30%, but I'm not sure why.
               # The important part seems to be not putting k last in the returned matrix
               # It presumably accesses memory more contiguously, but it's not obvious why. 
               dIdWlGcont = self.dIdWlG*contin[np.newaxis,:]
               self.dVsum = np.einsum('ijl,kl->ikj', dBlos_d, dIdWlGcont)               
               self.dVsum = np.swapaxes(self.dVsum, 1, 2)
               self.dVsum *= invSumIc0
          else:
               self.dVsum = 0

          #derivative of I wrt brightness for each surface element.
          #doing a bit of algebra to simplify this, we get
          if (calcDI == 1):
               self.dIcsum = (surfaceScale*invSumIc0)[:,np.newaxis]*(self.Iunscaled.T - self.I[np.newaxis,:])
          else:
               self.dIcsum = 0

          #Brightness derivatives should impact V too
          if ((calcDI == 1) & (calcDV == 1)):
               self.dVdBright = (surfaceScale*invSumIc0)[:,np.newaxis]*((Blos[np.newaxis,:]*self.dIdWlG).T - self.V[np.newaxis,:])
          else:
               self.dVdBright = 0 


     def dopplerShift(self, velShift):
          #Doppler shift the spectrum (+velocity away from the observer)
          c = 2.99792458e5 #speed of light in km/s
          self.wl = self.wl + self.wl*velShift/c
        

######################################
######################################
class diskIntProfAndDeriv:
     def __init__(self, visibleGrid, vMagCart, dMagCart, brightMap, lData, velEq, wlGrid, calcDI, calcDV):

          c = 2.99792458e5 #speed of light in km/s
          
          self.wl = wlGrid #use the observation's wavelength/velocity grid
          self.numPts = wlGrid.shape[0]
          self.wlStart = wlGrid[0]
          self.wlEnd = wlGrid[-1]

          self.IIc = np.zeros(self.numPts)
          self.QIc = np.zeros(self.numPts)
          self.UIc = np.zeros(self.numPts)
          self.VIc = np.zeros(self.numPts)

          #get the rotational velocity of the cell projected 
          # along the line of sight (+ve away from the observer)
          vrCell = velEq*visibleGrid.velRotProj
          #inverse of the Doppler shift, for each surface element's wavelength grid
          self.wlCells = np.outer(self.wl, 1./(1. + vrCell/c))

          self.prof = localProfileAndDeriv(lData, self.numPts, self.wlCells)
          self.updateIntProfDeriv(visibleGrid, vMagCart, dMagCart, brightMap,
                                  lData, calcDI, calcDV)
          

     def updateIntProfDeriv(self, visibleGrid, vMagCart, dMagCart, brightMap, lData, calcDI, calcDV):
          
          self.calcDI = calcDI
          self.calcDV = calcDV

          vView = visibleGrid.vViewCart

          #Get the line of sight component of B: |B|cos(theta), 
          # where theta is the angle between B and the direction of propagation
          projBlos = np.einsum('ij,ij->j', vView, vMagCart) #a fancy way of doing a row-wise dot product
          if (calcDV == 1):
               #order of indices in dMagCart is: x-y-z, alpha-beta-gamma, spherical harmonic, surface element
               dBlos_d = np.einsum('il,ijkl->jkl', vView, dMagCart) #row-wise dot product
          else:
               dBlos_d = 0
          
          #Find the limb darkening for this cell (or array of cells)
          limbDark = limbDarkening(lData.limbDark, visibleGrid.viewAngle)
          
          #Get the gravity darkening for this cell, for oblate stars
          #(returns 1 for spherical stars)
          gDark = visibleGrid.gravityDarkening(lData.gravDark)
          
          #Use this to include limb darkening, gravity darkening, and surface areas
          #in the local profile continuum flux scaling.
          surfaceScale = limbDark*gDark*visibleGrid.projArea*visibleGrid.visible

          self.prof.updateProfDeriv(lData, projBlos, dBlos_d, brightMap,
                                    self.numPts, self.wlCells, surfaceScale,
                                    calcDI, calcDV)

          #sum over surface elements and normalization by continuum now done in localProfileAndDeriv
          self.IIc = self.prof.I
          self.VIc = self.prof.V
          self.dVIc = self.prof.dVsum
          self.dIIc = self.prof.dIcsum 
          self.dVdBri = self.prof.dVdBright


     def convolveIGnumpy(self, fwhm):
          #convolve the calling spectrum with a Gaussian instrumental profile
          # with a resolution (unitless R) of FWHM.
          # Assumes that the calling spectrum is on an evenly spaced grid!
          #Note: same results as convolveInstGauss() but faster!

          #If the convolution with an instrumental profile is incorporated
          #into the local line profile widths, no calculations are needed.
          if explicitConvolution == False:
               return
          
          wlCenter = (self.wlStart+self.wlEnd)/2.
          wlGstep = (self.wlEnd - self.wlStart)/(self.numPts-1)
          fwhmWl = wlCenter/fwhm
          wlStartG = -3.*fwhmWl
          wlEndG = 3.*fwhmWl
          numPtsG = 2*int(wlEndG/wlGstep)+1

          #If resolution is high enough this will just introduce errors, so skip it.
          if (fwhmWl < wlGstep):
               #print('skipping convolution: FWHM ({:}) < pixle size ({:})'.format(fwhmWl, wlGstep))
               return

          #Check if there is an uneven spacing in wavelength grid
          #(probably uneven pixel sizes in the LSD profiles)
          flagUneven = False
          steps = self.wl[1:-1] - self.wl[0:-2]
          if not np.allclose(steps, wlGstep, rtol=1e-3):
               flagUneven = True
               print('Error: uneven spacing in wavelength pixels. '
                     +'Convolution with the instrumental profile may produce errors.')
          
          #Generate the Gaussian instrumental profile, normalized to unity
          wlG = np.linspace(wlStartG, wlEndG, numPtsG)
          profG = (0.939437/fwhmWl)*np.exp(-2.772589*((wlG/fwhmWl)**2))
          norm = np.sum(profG)
          profG = profG/norm
          
          #pad the arrays by repeating the first and last points an extra numPtsG/2 times 
          # on either end.  Otherwise the convolution routine will automatically pad 
          # the arrays with 0, or it will reject numPtsG/2 points on either end
          tmpI = np.append(self.IIc, np.repeat(self.IIc[self.numPts-1], numPtsG//2))
          tmpI = np.insert(tmpI, 0, np.repeat(self.IIc[0], numPtsG//2))
          tmpQ = np.append(self.QIc, np.repeat(self.QIc[self.numPts-1], numPtsG//2))
          tmpQ = np.insert(tmpQ, 0, np.repeat(self.QIc[0], numPtsG//2))
          tmpU = np.append(self.UIc, np.repeat(self.UIc[self.numPts-1], numPtsG//2))
          tmpU = np.insert(tmpU, 0, np.repeat(self.UIc[0], numPtsG//2))
          tmpV = np.append(self.VIc, np.repeat(self.VIc[self.numPts-1], numPtsG//2))
          tmpV = np.insert(tmpV, 0, np.repeat(self.VIc[0], numPtsG//2))

          #perform a discreet convolution
          #scipy.signal.fftconvolve is actually a bit slower here,
          # since the arrays involved are small.  
          # (An FFT convolution would be better for large spectra)
          convI = np.convolve(tmpI, profG, mode='valid')
          convQ = np.convolve(tmpQ, profG, mode='valid')
          convU = np.convolve(tmpU, profG, mode='valid')
          convV = np.convolve(tmpV, profG, mode='valid')

          self.IIc = convI
          self.QIc = convQ
          self.UIc = convU
          self.VIc = convV

          #If requested, do the convolution for the Stokes I derivatives wrt the brightness map parameters
          if (self.calcDI == 1):
               numPtsPad = numPtsG//2
               tmpdIHead = np.tile(self.dIIc[:,0:1], (1,numPtsPad))
               tmpdIFoot = np.tile(self.dIIc[:,self.numPts-1:self.numPts], (1,numPtsPad))
               tmpdI = np.concatenate((tmpdIHead, self.dIIc, tmpdIFoot), axis=1)
               
               #The np.convolve routine requires 1D arrays.
               #For speed, flatten the input dIIc (tmpdI) array, then do the convolution all at once
               #rather than iterating over the numSurfaceElments axis of the array.
               #Since the array has been padded by numPtsG/2 points on either end, 
               #the convolution won't blur adjacent columns of the array together, 
               #we just discard the padding after the convolution (as we would do anyway)
               tmpFlatdX = np.ravel(tmpdI)
               convdI = np.convolve(tmpFlatdX, profG, mode='same')
               self.dIIc = np.reshape(convdI, tmpdI.shape)[:,numPtsPad:-numPtsPad]

          #If requested, do the convolution for the Stokes V derivatives wrt magnetic coefficients
          if (self.calcDV == 1):
               numPtsPad = numPtsG//2
               tmpdVHead = np.tile(self.dVIc[:,:,0:1], (1,1,numPtsPad))
               tmpdVFoot = np.tile(self.dVIc[:,:,self.numPts-1:self.numPts], (1,1,numPtsPad))
               tmpdV = np.concatenate((tmpdVHead, self.dVIc, tmpdVFoot), axis=2)
               
               #the np.convolve routine requires 1D arrays
               tmpFlatdX = np.ravel(tmpdV)
               flatConvdV = np.convolve(tmpFlatdX, profG, mode='same')
               self.dVIc = np.reshape(flatConvdV, tmpdV.shape)[:,:,numPtsPad:-numPtsPad]

          #If fitting I and V, and needing derivatives of V wrt brightness pixels
          if ((self.calcDI == 1) & (self.calcDV == 1)):
               numPtsPad = numPtsG//2
               tmpdVHead = np.tile(self.dVdBri[:,0:1], (1,numPtsPad))
               tmpdVFoot = np.tile(self.dVdBri[:,self.numPts-1:self.numPts], (1,numPtsPad))
               tmpdV = np.concatenate((tmpdVHead, self.dVdBri, tmpdVFoot), axis=1)
               
               #the np.convolve routine requires 1D arrays
               tmpFlatdX = np.ravel(tmpdV)
               flatConvdV = np.convolve(tmpFlatdX, profG, mode='same')
               self.dVdBri = np.reshape(flatConvdV, tmpdV.shape)[:,numPtsPad:-numPtsPad]              

          return

######################################

def getAllProfDiriv(par, listGridView, vecMagCart, dMagCart0, briMap, lineData, wlSynSet):
     #calculate the line profiles and derivatives for each observed phase.
     #return the results as a list.  Intended for initialization and setup.
     nObs = 0
     setSynSpec = []
     for phase in par.cycleList:
          spec = diskIntProfAndDeriv(listGridView[nObs], vecMagCart, dMagCart0,
                                     briMap, lineData, par.velEq,
                                     wlSynSet[nObs], 0, par.calcDV)
          spec.convolveIGnumpy(par.instrumentRes)
          setSynSpec += [spec]
          nObs += 1
     return setSynSpec


def calcSynEW(spec):
     #calculate an equivalent width for a synthetic I profile
     #equivWidApprox = (np.sum(1.-spec.IIc))*(spec.wl[1]-spec.wl[0])
     equivWidApprox = 0.
     for i in range(spec.wl.shape[0]-1):
          equivWidApprox += (1.-spec.IIc[i])*(spec.wl[i+1]-spec.wl[i])
     equivWidApprox += (1.-spec.IIc[-1])*(spec.wl[-1]-spec.wl[-2])
     return equivWidApprox
