#Contains functions defining the numerical grid that makes up the model
#stellar surface, and for converting to different coordinates or viewing angles.

from math import pi, sin, cos, acos
import numpy as np

class starGrid:
    def __init__(self, numPoints, period = 0.0, mass = 0.0, radiusEq = 0.0, verbose=1):
        #Generate a grid of roughly square surface elements on a sphere, with roughly the same size
        #The coordinates used are colatitude, longitude, and radius.  
        # longitude increases in the right-handed sense, running from 0 to 2*pi
        # colatitude increases away from the 'north' rotational pole (+ve pol in the right handed sense), running from 0 to pi
        # Generally colatitude is referred to as theta and longitude is phi.
        # Set the stellar surface at radius = 1, for a spherical star.
        # Input: numPoints is the number of points in colatitude 
        # (or rings of surface elements in colatitude).

        #number of points in colatitude
        self.numPtsClat = numPoints
        #number of points in longitude at the equator
        self.numPtsLongEq = 2*self.numPtsClat
        #number of points in longitude for each colatitude strip
        self.numPtsLong_Clat = np.zeros(self.numPtsClat, dtype=int)

        #Total number of points in longitude and latitude 
        #(i.e. the total number of stellar surface elements)
        self.numPoints = 0
        for i in range(self.numPtsClat):
            _clat = pi*((float(i)+0.5)/float(self.numPtsClat))
            #for approximately equal area surface elements,
            #use lnumPtsLong of longitudinal points at this colatitude
            lnumPtsLong = int(round(sin(_clat)*float(self.numPtsLongEq)))
            self.numPtsLong_Clat[i] = lnumPtsLong
            self.numPoints += lnumPtsLong

        self.clat = np.zeros(self.numPoints)
        self.long = np.zeros(self.numPoints)
        self.radius = np.zeros(self.numPoints)
        self.dClat = np.zeros(self.numPoints)
        self.dLong = np.zeros(self.numPoints)
        self.dRadius = np.zeros(self.numPoints)
        self.gdark_local = None

        #Setup the grid in colatitude and longitude
        n = 0
        for i in range(self.numPtsClat):
            _clat = pi*((float(i)+0.5)/float(self.numPtsClat))
            _dClat = pi*(1.0/float(self.numPtsClat))
            lnumPtsLong = int(self.numPtsLong_Clat[i])

            for j in range (lnumPtsLong):
                self.clat[n] = _clat
                self.dClat[n] = _dClat

                _long = 2.*pi*((float(j)+0.5)/float(lnumPtsLong))
                self.long[n] = _long
                _dLong = 2.*pi*(1.0/float(lnumPtsLong))
                self.dLong[n] = _dLong

                n += 1

        #Set up parameters for a sphere, then overwrite below if necessary
        #For a sphere, radius is always 1.
        self.radius[:] = 1.0
        self.dRadius[:] = 0.0
        self.fracOmegaCrit = 0.0
                
        #For oblate stars, update the radius
        #(and fraction of breakup angular velocity)
        #update the cell dimensions in radius (angular dimensions are the same)
        if (mass > 0.0 and radiusEq > 0.0):
            self.radius, self.fracOmegaCrit = self.calcRclatOblate(
                period, mass, radiusEq, verbose)
            #Evaluate the span in radius of the corners of the surface elements
            self.dRadius = self.getOblateRadiusStep(period, mass, radiusEq)
        #get the surface area of the cells, now includes oblateness if needed
        self.area = self.GetSurfaceArea()
        
        if(verbose == 1):
            print("initiated a {:} point stellar grid  ( {:} clat, {:} long_equator)".format(self.numPoints, self.numPtsClat, self.numPtsLongEq))


    def GetCellCorners(self, i):
        #Return the 4 corners of this surface element, in colatitude, longitude and radius
        long1 = self.long[i] - 0.5*self.dLong[i]
        long2 = self.long[i] + 0.5*self.dLong[i]
        clat1 = self.clat[i] - 0.5*self.dClat[i]
        clat2 = self.clat[i] + 0.5*self.dClat[i]
        #For possibly oblate stars with r(clat)
        r1 = self.radius[i] - 0.5*self.dRadius[i]
        r2 = self.radius[i] + 0.5*self.dRadius[i]
        return np.array([[long1, clat1, r1], [long2, clat1, r1], [long1, clat2, r2], [long2, clat2, r2]])

    def GetCartesianCells(self):
        #Return the position of all cells in Cartesian coordinates
        x = self.radius*np.sin(self.clat)*np.cos(self.long)
        y = self.radius*np.sin(self.clat)*np.sin(self.long)
        z = self.radius*np.cos(self.clat)
        return np.array([x,y,z])
        
    def GetCartesianCellCorners(self, i):
        #Returns the 4 corners of the cell, in Cartesian coordinates
        #(mostly useful for plotting)
        cartCor = np.zeros((4,3))
        cor = self.GetCellCorners(i)
        #loop over the 4 corners of the cell
        for j in range(4):
            #and convert from spherical to Cartesian coordinates
            x = cor[j,2]*np.sin(cor[j,1])*np.cos(cor[j,0])
            y = cor[j,2]*np.sin(cor[j,1])*np.sin(cor[j,0])
            z = cor[j,2]*np.cos(cor[j,1])
            cartCor[j,0] = x
            cartCor[j,1] = y
            cartCor[j,2] = z
        return cartCor

    def GetSurfaceArea(self):
        #Get the areas of all surface elements
        area = np.zeros(self.numPoints)
        #For an oblate star:
        if self.fracOmegaCrit > 0.:
            #We want the surface integral using the radius in calcRclatOblate, 
            #in spherical coordinates.  The definite surface integral is:
            #int_{phi1}^{phi2}[ int_{theta1}^{theta2}[ 1/cos(gamma) r^2 sin(theta) dtheta] dphi]
            #where gamma is the angle between the surface unit normal and
            #the radial unit vector at this point. 
            #And r(theta) is (Tassoul 1978, Collins 1963, 1965, 1966):
            #r(theta) = 3/(w sin(theta)) cos((pi + arccos(w sin(theta)))/3)
            #where w the ratio of Omega/Omega_crit.
            #The dot product of the surface normal and radial unit vector gives cos(gamma).
            #The normal to a surface defined implicitly by f(x,y,z) = const
            #is grad f(x,y,z).  The surface of the star is an equipotential,
            #i.e. where Phi = -GM/R - 1/2 Omega^2 R^2 sin(theta)^2 = const.
            #So we can get the normal using grad (-GM/R - 1/2 Omega^2 R^2 sin(theta)^2),
            #where Omega = w sqrt(8/27 GM/Rp^3) with Rp the polar radius
            #Recall that grad Psi = (dPsi/dr)r^ + (dPsi/dtheta)/r theta^
            #                       + (dPsi/dphi)/(r sin(theta)) phi^
            #with unit vectors r^ theta^ phi^.
            #Then n = grad Phi = [GM/R^2 - Omega^2 R sin(theta)^2]r^
            #                  + [-Omega^2 R sin(theta) cos(theta)]theta^
            #         = GM/Rp^2 ([-8/27 w^2 r sin(theta)^2 + 1/r^2]r^
            #                  + [-8/27 w^2 r sin(theta) cos(theta)]theta^)
            #Here we use r normalized by the polar radius (r = R/Rp).
            #This vector n is then normalized and the dot product with r^ taken, giving:
            #cos(gamma) = (-8/27 w^2 r sin(theta)^2 + 1/r^2)
            #             /sqrt([-8/27 w^2 r sin(theta)^2 + 1/r^2]^2
            #                   + [-8/27 w^2 r sin(theta) cos(theta)]^2)
            #
            #Then the exact surface area should be:
            #int_{phi1}^{phi2}[ int_{theta1}^{theta2}[1/cos(gamma) 9/(w^2 sin(theta)) cos^2((pi + arccos(w sin(theta)))/3) dtheta] dphi]
            #which simplifies to a bit:
            #9/w^2 (phi2-phi1) int_{theta1}^{theta2} 1/(cos(gamma)sin(theta)) cos^2((pi + arccos(w sin(theta)))/3) dtheta
            #Lacking a good analytic expression for the integral,
            #we can use a numerical approach.
            #The function to integrate:
            def fThetaP(theta, wo):
                dS =  9./wo**2/np.sin(theta)*(
                    np.cos((np.pi + np.arccos(wo*np.sin(theta)))/3.))**2
                #Use the gradiant of the potential for a surface normal
                #Similar to Cranmer 1996 eq 4.20 with eqs 4.11 & 4.12
                rs = 3./(wo*np.sin(theta)) * np.cos((np.pi 
                                + np.arccos(wo*np.sin(theta)))/3)
                gradPhi_r = 1./rs**2 - 8./27.*rs*(wo*np.sin(theta))**2
                gradPhi_theta =  -8./27.*rs*wo**2*np.sin(theta)*np.cos(theta)
                #Normalize the vector for a unit normal, then dot with r unit vector
                cosGamma = gradPhi_r/np.sqrt(gradPhi_r**2 + gradPhi_theta**2)
                
                thetaP = dS/cosGamma
                return thetaP
            
            from scipy.integrate import quad
            #The scipy.integrate.quad function seems to be adequate for a 
            #definite integral, at least when it doesn't vary too radically,
            #as with this small integration range.
            #This is slower than it could be, but should <= a few 0.01 sec.
            #Integrating with my own trapezoidal rule approximation, 
            #sampled on 1000 points, gives a very good agreement
            #(and is about as fast if I pass an array to the funciton).
            for i in range(self.numPoints):
                if self.clat[i] != self.clat[i-1]:
                    intTheta, errTheta = quad(fThetaP,
                                              self.clat[i] - 0.5*self.dClat[i],
                                              self.clat[i] + 0.5*self.dClat[i],
                                              args=(self.fracOmegaCrit))
                area[i] = self.dLong[i]*intTheta

            ##The rough quadrilateral approximation, in cartesian coords.
            ##This is not realy accurate enough, but gets closer for a fine 
            ##stellar grid.  Calculating this using a 10x10 grid of sub-pixels
            ##improves the accuracy, but is too slow.
            #areaQuad = np.zeros_like(area)
            #for i in range(self.numPoints):
            #    corn = self.GetCartesianCellCorners(i)
            #    #An area for a quadralateral can be calculated from 1/2*|AC x BD|,
            #    #for vectors AC, BD from corner A to C and B to D.
            #    #The corners in the array are oriented as:
            #    # 0(A)  1(B)
            #    # 2(D)  3(C)
            #    areaQuad[i] = 0.5*np.linalg.norm(np.cross(corn[3]-corn[0],
            #                                              corn[2]-corn[1]))
            ##print('area/pi:  num_int', np.sum(area)/np.pi,
            ##      'quad_approx', np.sum(areaQuad)/np.pi)

        else:  #For a sperical star:
            #exact surface element area calculation, assuming a sphere
            area = self.dLong*(np.cos(self.clat - 0.5*self.dClat)
                                    - np.cos(self.clat + 0.5*self.dClat))
        return area
                


    def GetDistRotAxis(self):
        #Get the distance of the cells to the rotation axis
        #Not currently used!!!
        return self.radius*np.sin(self.clat)

    def GetCartesianRotVel(self):
        #Return the rotational velocity as a fraction of the equatorial rotational velocity, in Cartesian coordinates
        #Done for all cells.  Assuming solid body rotation!
        #This is just d(position)/d(long) converted to Cartesian coordinates.
        #Or equivalently, the velocity along the unit vectors r^, theta^, phi^, 
        #with v_r=0, v_theta=0, and v_phi=sin(theta)=distance from the rotation axis, 
        #then converted into Cartesian coordinates.

        velX = np.negative(self.radius*np.sin(self.clat)*np.sin(self.long))
        velY = self.radius*np.sin(self.clat)*np.cos(self.long)
        velZ = np.zeros(len(self.clat))
        #for left handed rotation:
        #velX = self.radius*np.sin(self.clat)*np.sin(self.long)
        #velY = np.negative(self.radius*np.sin(self.clat)*np.cos(self.long))
        velVec = np.array([velX, velY, velZ])

        #Normalize by equitorial radius, for oblate stars,
        #so that this can be multiplied by equitorial velocity (from vsini)
        #iEquator is expected to be int(self.numPoints:2), but in case of unexpected grids:
        iEquator = np.argmin(np.abs(self.clat-0.5*np.pi)) 
        velVec = velVec/self.radius[iEquator]
        
        return velVec

    def getCartesianNormals(self):
        #Get the surface normals for each surface element in Cartesian coordinates

        #If the geometry is a sphere,
        #the line from origin to the cell center is the normal to the surface
        if self.fracOmegaCrit <= 0.0:
            vNormals = self.GetCartesianCells()
            lenThisPoint = np.sqrt(vNormals[0,:]**2 + vNormals[1,:]**2 + vNormals[2,:]**2)
            vNormals /= lenThisPoint

        #If the star is oblate, the surface normals are more complex
        else:
            wo = self.fracOmegaCrit
            sr = self.radius
            sclat = self.clat
            slong = self.long
            ##As a resonable numerical approximation, use the surface element
            ##corners to construct two vectors tangential to the surface.
            ##Then their cross product to get a vector normal to the surface.
            ##This needs more array operations to be efficent!
            #vNormals1 = np.zeros((3, self.numPoints))
            #for i in range(self.numPoints):
            #    aThisCorner = self.GetCartesianCellCorners(i)
            #    tangent1 = (aThisCorner[1] - aThisCorner[0])
            #    tangent2 = (aThisCorner[0] - aThisCorner[2])
            #    if i <= 3: #protect against the north pole
            #        tangent1 = (aThisCorner[3] - aThisCorner[2])
            #    normal = np.cross(tangent1, tangent2)
            #    normal /= np.linalg.norm(normal, axis=0) #normalize for unit normal
            #    vNormals1[:,i] = normal
                
            ##Alternately use an analytic expression for surface normals,
            ##for an oblate rapidly rotating star.
            ##To derive this, take the eq. for a Roach surface and convert it to 
            ##cartesian coordiantes, using the usual conversion with r=r(theta).
            ##That gives f(theta, phi) = [fx(theta,phi), fy(theta, phi), fz(theta, phi)].
            ##Then calculate two tangents to the surface using the partial derivatives
            ##df/dtheta and df/dphi. The cross product of two tangents gives
            ##a surface normal.  Then normalize the surface normal to get a unit normal.
            #dRdTheta = (1.0/np.tan(sclat)
            #            /np.sqrt(1.-(wo*np.sin(sclat))**2)
            #            *np.sin((np.pi+np.arccos(wo*np.sin(sclat)))/3.)
            #            - 3./(wo*np.tan(sclat)*np.sin(sclat))
            #            *np.cos((np.pi+np.arccos(wo*np.sin(sclat)))/3.) )
            #tanCart1 = np.vstack((dRdTheta*np.sin(sclat)*np.cos(slong)
            #                     + sr*np.cos(sclat)*np.cos(slong),
            #                     dRdTheta*np.sin(sclat)*np.sin(slong)
            #                     + sr*np.cos(sclat)*np.sin(slong),
            #                     dRdTheta*np.cos(sclat) - sr*np.sin(sclat) ))
            #tanCart2 = np.vstack((-sr*np.sin(sclat)*np.sin(slong),
            #                     sr*np.sin(sclat)*np.cos(slong),
            #                     np.zeros_like(slong) ))
            #anaNorm = np.cross(tanCart1, tanCart2, axis=0)
            #vNormals = anaNorm/np.linalg.norm(anaNorm, axis=0)

            #Another analytic alternative: the normal to a surface defined implicitly
            #by f(x,y,z) = const is grad f(x,y,z).  The surface of our star is
            #an equipotential, i.e. where Phi = -G*M/r - 1/2*Omega^2*r^2*sin(theta)^2 = const.
            #So we can use grad (-G*M/r - 1/2*Omega^2*r^2*sin(theta)^2),
            #where Omega = w*sqrt(8/27*G*M/Rp^3) with Rp the polar radius.
            #Note grad Psi = dPsi/dr*r^ + dPsi/dtheta*1/r*theta^ + dPsi/dphi*1/(r*sin(theta))*phi^
            #and we convert to r normalized by the polar radius (r_n = r/Rp).
            #Then neglecting a constant factor of G*M/Rp^2 we get:
            gradPot = np.array([-8./27.*wo**2*sr*np.sin(sclat)**2 + 1./sr**2,
                                -8./27.*wo**2*sr*np.sin(sclat)*np.cos(sclat),
                                np.zeros_like(sclat)])
            #Normalize the vector
            gpotnorm = gradPot/np.sqrt(gradPot[0]**2+gradPot[1]**2) #phi component = 0
            #And finally convert to catesian coordinates
            #Convert a unit vector from spherical to rectangular coordinates, with a matrix
            R = np.array([[np.sin(sclat)*np.cos(slong), np.cos(sclat)*np.cos(slong), -np.sin(slong)],
                          [np.sin(sclat)*np.sin(slong), np.cos(sclat)*np.sin(slong), np.cos(slong)],
                          [np.cos(sclat) , -np.sin(sclat), np.zeros_like(sclat)]])
            #matrix dot product, but leave the last axis (surface elements) alone
            vNormals = np.einsum('ijk,jk->ik', R, gpotnorm)
            #This is a somewhat more efficient calculation, requiring fewer trig
            #function evaluations.
            
        return vNormals

            
    def calcRclatOblate(self, period, mass, radius, verbose=1):
        # calculate the radius at specific colatitude (clat) using a Roche model
        #(see Tassoul 1978, Collins 1963, 1965, 1966)
        #Implementation from Tianqi Cang
        G = 6.67408e-11  # gravitational constant (m3*kg-1*s-2)
        Msun = 1.98892e30  # solar mass (kg)
        Rsun = 6.955e8  # solar radius (m)
        # rotational rate (rad/s), converting the period from days to seconds
        Omega = 2*np.pi/(period*86400.)
        # breakup angular velocity 
        Omega_break =  np.sqrt((8./27.)*G*(mass*Msun)/(radius*Rsun)**3)
        # wo is the fraction of breakup angular velocity
        wo = Omega/Omega_break
        if wo > 1.0: #Catch unphysical stellar paramters
            print(('Error: star above breakup velocity (Omega/Omega_break {:}), \n' 
                  + '   for given mass {:} Msun, radius {:} Rsun, and period {:} d').format(
                      wo, mass, radius, period))
            print('Assuming breakup velocity for geometry calculations')
            wo = 1.0
        elif wo < 1e-8: #Catch numerical underflow or divide by zero
            wo = 1e-8
        obl = (3./wo)*cos((np.pi + acos(wo))/3.)
        if verbose:
            print('The Oblateness of the star: Req/Rp = {:} ({:} of breakup)'.format(
                obl, wo))
        x_clat = 3./(wo*np.sin(self.clat)) \
                 *np.cos((np.pi + np.arccos(wo*np.sin(self.clat)))/3)
        
        return x_clat, wo

    def getOblateRadiusStep(self, period, mass, radiusEq):
        #Evaluate the span in radius of the corners of this surface element
        #between the clat+dClat/2 corner and clat-dClat/2 corner for an oblate star
        if (mass > 0.0 and radiusEq > 0.0):
            _refClat = self.clat
            #Temporarily overwriting self.clat is probably a bit dangerous!
            self.clat = _refClat + 0.499999999*self.dClat  #avoid the exact pole
            radiusP, tmp = self.calcRclatOblate(period, mass, radiusEq, verbose=False)
            self.clat = _refClat - 0.499999999*self.dClat  #avoid the exact pole (div by 0)
            radiusM, tmp = self.calcRclatOblate(period, mass, radiusEq, verbose=False)
            self.clat = _refClat #reset self.clat to the correct centre values!
            dRadius = radiusP - radiusM
        else:
            dRadius = np.zeros_like(self.clat)
        return dRadius

    def gravityDarkening(self, gravDarkCoeff): 
        #Calculate the local gravitational darkening,
        #this should be 1 at the pole, decreasing towards the equator.
        #For spherical stars this is just 1.0.
        
        bCalc = True
        if (self.gdark_local is not None):
            if (self.gravDarkCoeff == gravDarkCoeff):
                #If values already exist and they were for the correct
                #gravity darkening coefficent, just reuse them.
                gdark_local = self.gdark_local
                bCalc = False
        if bCalc:
            #If we need to calculate the value
            wo = self.fracOmegaCrit
            if (wo > 0.0) and (gravDarkCoeff > 0.0):
                gr = -1./self.radius**2 + 8./27.*self.radius*(wo*np.sin(self.clat))**2
                gtheta = 8./27.*self.radius*np.sin(self.clat)*np.cos(self.clat)*wo**2
                gdark_local = (np.sqrt(gr**2 + gtheta**2))**gravDarkCoeff
            else:
                gdark_local = np.ones_like(self.clat)
            #Save the local darkening and the gravity darkening coefficient
            self.gravDarkCoeff = gravDarkCoeff
            self.gdark_local = gdark_local
            
        return gdark_local


    

class visibleGrid:
    def __init__(self, starGrid, inclination, cycleAtClat, period, dOmega):
        #Computes angles between surface element normals and the line of sight,
        #flags cells as visible, computes projected areas, 
        #and projected rotation velocities, normalized to v_eq*sin(i) = 1.
        #
        # Note: this uses a coordinate system aligned with the stellar rotation 
        # axis, and defined at rotation cycle 0.  The coordinate system is
        # co-rotating with the equator of the star.  
        # The magnetic (and brightness) map is defined at rotation cycle 0.  
        #
        # Here we use a varying view angle for each surface element,
        # at each rotation phase. 
        #
        # Added the impact of differential (non-solid body) rotation on the
        # rotational velocity (as a function of colatitude).  
        # This could be done elsewhere to avoid doing the calculation once
        # per observation, and this requires period & dOmega, which is adds
        # a little complexity.

        #Save a reference to the stellar grid
        self.starGrid = starGrid

        #Setup the viewing angle, as a vector in Cartesian coordinates
        viewClat = inclination
        viewLong = (-cycleAtClat)*2.*np.pi
        # using right handed rotation and right handed longitude
        #
        #From JFD's equation:
        #viewLong = (-phase)*2.*np.pi
        #viewLong += phase*2.*np.pi*(pBeta + pGamma*(np.cos(starGrid.clat))**2)
        # where: beta = 1 - Omega_eq/Omega0 = 1 - P0/P_eq
        #        gamma = dOmega/Omega0 = dOmega*P0/(2*pi)
        #        and P0 is an arbitrary reference period (Omega0 is a reference angular frequency)
        # or:    P_eq = P0/(1 - beta)
        #        dOmega = gamma*(2*pi)/P0

        viewX = 1.*np.sin(viewClat)*np.cos(viewLong)
        viewY = 1.*np.sin(viewClat)*np.sin(viewLong)
        viewZ = 1.*np.cos(viewClat)*np.ones(viewLong.shape)
        vView = np.array([viewX, viewY, viewZ])
        lenView = np.linalg.norm(vView, axis=0)
        #normalized vector for the line of sight, in Cartesian coordinates
        self.vViewCart = vView/lenView


        #Find the angles between the surface element normals and the line of sight
        #first get the vector normal to the surface element
        vNormals = starGrid.getCartesianNormals()
        #and then use a dot product to get cos of the angle between vectors
        dotprod = np.einsum('ij,ij->j', vNormals, vView) #a fancy way of doing a row-wise dot product
        _cosViewAngle = dotprod/(lenView)
        self.viewAngle = np.arccos(_cosViewAngle)

        #set the visible cells
        #visible if the angle between the line of sight and the cell's normal is <90 degrees
        self.visible = np.zeros(starGrid.numPoints, dtype=int)
        for i in range(starGrid.numPoints):
            if (self.viewAngle[i] > 0.) & (self.viewAngle[i] < pi*0.5):
                self.visible[i] = 1
            else:
                self.visible[i] = 0

        #Find the area of the cell projected on the disk of the star.
        # (negative values should not be visible)
        #This is an approximation, which maybe should be made exact!
        self.projArea = starGrid.area*_cosViewAngle

        #Find the rotation velocity, normalized to v_eq=1.
        vVel = starGrid.GetCartesianRotVel()
        #Include the change in the velocity profile due to non-solid body rotation
        velDiffRot = calcVelDiffrotFactor(period, dOmega, starGrid.clat)
        vVel = vVel*velDiffRot
        #Rotation velocity projected on the line of sight, normalized to v_eq=1.
        #Using the +ve direction going away from the observer, 
        # so the view vector needs to be -ve of normal.
        self.velRotProj = np.einsum('ij,ij->j', -self.vViewCart, vVel) #row-wise dot product

        #Create an alias of the gravity darkening function,
        #for ease of use in the lineprofileVoigt.py function diskIntProfAndDeriv
        self.gravityDarkening = self.starGrid.gravityDarkening

def calcOmegaClat(period, dOmega, clat):
    #Calculate rotational angular frequency (Omega) as a function of colatitude on the star,
    #due to differential rotation. 
    #Uses the solar-like law:  Omega(lat) = Omega_eq - dOmega*sin^2(lat)
    # => Omega(clat) = Omega_eq - dOmega*cos^2(clat)
    omega_eq = 2.*np.pi/period
    omega_clat = omega_eq - dOmega*(np.cos(clat))**2
    return omega_clat

def getCyclesClat(period, dOmega, jDates, jDateRef, clat):
    #get the rotation cycle (phase) for each surface element, 
    # using the differential rotation law in calcOmegaClat()
    omega_clat = calcOmegaClat(period, dOmega, clat)
    
    period_clat = 2.*np.pi/omega_clat
    cycleAtClat = (jDates[:,np.newaxis] - jDateRef)/period_clat[np.newaxis,:]

    return cycleAtClat

def getListGridView(par, sGrid):
    #Get projected, rotated, stellar geometry parameters for each observed phase
    #Differential rotation calculation: 
    #rotation cycle for each surface element, for each observation
    cyclesAtClat = getCyclesClat(par.period, par.dOmega, par.jDates, par.jDateRef, sGrid.clat)
    
    nObs = 0
    listGridView = []
    for jDate in par.jDates:
        #Calculate rotation phase dependent geometric quantities
        sGridView = visibleGrid(sGrid, par.incRad, cyclesAtClat[nObs,:], par.period, par.dOmega)
        listGridView += [sGridView]
        nObs += 1

    return listGridView

def calcVelDiffrotFactor(period, dOmega, clat):
    #Include the impact of differential rotation on the rotation velocity, as a function of colatitude.
    #Simply use the fractional differential rotation for the fractional velocity change relative to Veq.
    omega_eq = 2.*np.pi/period
    omega_clat = calcOmegaClat(period, dOmega, clat)
    norm_clat = omega_clat/omega_eq
    
    return norm_clat
