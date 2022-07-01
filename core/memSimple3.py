# Maximum entropy method (MEM) image reconstruction, using the algorithm of
# J. Skilling and R.K. Bryan, (1984, MNRAS 211, 111).
#
# This routine iteratively searches for the maximum entropy image.
# Each call to this routine makes one iteration towards the best-fit maximum 
# entropy image, so this should be called in a loop from the main program.
# The routine roughly follows the notation of Skilling and Bryan,  
# but with more verbose names. The comments with equation references 
# refer to their 1984 paper.
#

import numpy as np
from scipy import linalg

def mem_iter(n1, n2, ntot, Img, Data, Fmodel, sig2, Resp, weights, defImg, defIpm, ffIMax, targetAim, fixEntropy):
    """
    Maximum entropy method (MEM) image reconstruction, using the algorithm
    of Skilling & Bryan (1984, MNRAS 211, 111).
    This allows for different forms of entropy to be applied to different parts of
    the image vector, for simultainous magnetic and brightness mapping.

    :param n1: use standard image entropy (+ve, no upper limit) for the first n1 elements of Img.
    :param n2: use filling factor entropy (+ve, limiting upper value) for the n1:n2 elements of Img.
      Filling factor entropy follows A. Collier Cameron (1992, in Surface Inhomogeneities
      on Late-Type Stars, p. 33) and Y.C. Unruh & A. Collier Cameron (1995, MNRAS, 273, 1) Eq 4.
    :param ntot: total number of elements in the image vector Img.
      Positive/negative (magnetic) entropy (allows -ve & +ve, trends towards 0, no upper/lower limits)
      is used for the n2:ntot elements.  This form of entropy is from 
      Hobson & Lasenby (1998, MNRAS, 298, 905) (e.g. Eq 8)
    :param Img: array of current image values (ntot long), corresponds to f in Skilling and Bryan (Eq 1).
    :param Data: array of data values the model is fit to (Eq 2).
    :param Fmodel: array of model, i.e. simulated data, calculated from the current input image Img.
      (Same dimensions as Data, and for the same points.)  For ZDI this is the model spectrum.
    :param sig2: array standard deviations (1 sigma errors) squared, for data values in Data.
      (Same dimensions as Data.)
    :param Resp: response matrix, derivatives of the model (Fmodel) with respect to the image (Img).
      That is, each element of the Resp is defined as R(k,j) = dF(k)/dI(j).
      (Dimensions of [Data.shape, Img.shape])
    :param weights: array of additional weights, which are multiplied with the entropy terms
      for each image pixel.  (Same Dimension as Img)
      This is debatable from a theoretical stand point, since it would be more elegant
      and more in keeping with Bayesian statistics to include this in the default values
      for each pixel.  That would include per-pixel weights properly as a prior.  However in testing
      a simple multiplicative value was effective, and is more consistent with other ZDI codes. 
    :param defImg: default image intensity (Eq 6), for standard entropy.
      Used for the Img[0:n1] part of the image.  For filling factor entropy (Img[n1:n2] part of the image)
      this is also used as the default value (e.g. m in Eq 4 of Unruh & Collier Cameron (1995)).
    :param defIpm: "default" value for the positive/negative (magnetic) entropy.
      Actually controls the slope/change in slope in the entropy curve, the values will default to 0.
      See Hobson & Lasenby (1998), this for their eq 8,  m_f and m_g, when m_f = m_g = defIpm.
      Applies to Img[n2:nmax]
    :param ffIMax: maximum image value allowed for filling factor entropy as in, e.g.,
      Eq 4 of Unruh & Collier Cameron (1995) but replacing their 1 with this value
      (i.e. not limiting the 'filling factor' to strictly 1).
      Applies to Img[n1:n2]
    :param targetAim: the desired chi-squared (Eq 4) value to converge to,
      or the desired entropy to converge to.
    :param fixEntropy: A flag for whether to fit to target chi^2 or target entropy.
      Normaly this routine would fit to a target chi^2 level, but if this == 1
      the routine will fit to target entropy level.

    :return: entropy, chi2, test, Img, entStand, entFF, entMag
      * entropy: The entropy (Eq 6) of the input image.
      * chi2: The chi-squared (Eq 4) value for the input data and model.
      * test: The test statistic of Skilling and Bryan (Eq 37) for the input image, model, and data.
              This matches the returned entropy and chi2.
      * Img: Updated array of image values.
      * entStand: The entropy for the standard entropy part of the image.
      * entFF: The entropy for the 'filling factor' part of the image
      * entMag: The entropy for the positive/negative (magnetic) part of the image
    """

    #Set the maximum number of search directions used
    #Skilling and Bryan recommend 3-6 (Sect 3.6 near eq 20) (mostly tested with 6).
    #More search directions may aid for more rapid convergence,
    #but at the cost of most computationally intensive iterations
    #particularly in the searchDir routine, and a bit more memory.
    #  In one test fitting brightness maxDir=10-12 was optimal.
    #  similarly for the same star fitting magnetic maxDir=10-12 was optimal
    #  (over-fitting converged more efficiently with even higher maxDir)
    maxDir = 10
    
    #L_fac limits the step size of the iteration in parameter space.
    #Similar to limiting to a fractional change in parameters.
    #Uses Eq 17, where l_o^2 = L_fac*sum(I)
    L_fac = 0.3

    #for S (entropy) and C (chi^2).
    C0, gradC = get_c_gradc(Data, Fmodel, sig2, Resp)
    S0, gradS, gradgradS, fsupi, Itot, entStand, entFF, entMag = \
        get_s_grads(n1, n2, ntot, Img, weights, defImg, defIpm, ffIMax)
    test = get_test(gradC, gradS)

    #Calculate the normalized search directions.
    edir, nedir, gamma = searchDir(ntot, maxDir, Resp, sig2, fsupi, gradC, gradS, gradgradS)

    #Calculate some values needed by the control subroutine. 
    Cmu, Smu = getCmuSmu(gradC, gradS, edir)
    L02 = getL0squared(L_fac, Itot)
    alphaMin = getAlphaMin(gamma)
    
    if (fixEntropy == 1):
        #The modified fuctions needed for fitting to target entropy
        import core.memSaim3 as saim
        Saim = targetAim
        Saimq = saim.getSaimQuad(Saim, S0, gamma, Smu)
        
        #Run a modified version of Skilling and Bryan's 'control' procedure
        xq = saim.control(S0, gamma, Cmu, Smu, Saimq, L02, alphaMin)
    else:
        chiAim = targetAim
        Caimq = getCaimQuad(chiAim, C0, gamma, Cmu)
        
        #Run Skilling and Bryan's 'control' procedure (e.g. Fig 3)
        #This calculates the coefficients x for a step forward
        #in each of the search directions. 
        xq = control(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin)
    
    #Update the image vector along the search directions.
    Img = updateImg(xq, edir, Img, n1, n2, ntot, ffIMax)

    entropy = S0
    chi2 = C0

    return entropy, chi2, test, Img, entStand, entFF, entMag


def get_c_gradc(Data, Fmodel, sig2, Resp):
    #Finds the current chi^2, and gradient in chi^2

    #the current 'constraint statistic' (usually chi^2) for this iteration (Eq 4). 
    C0 = np.sum((Fmodel - Data)**2/sig2)
    #Finds the gradient of C (Eq 8). (C is the 'constraint statistic' usually chi^2)
    gradC = 2.0*np.sum(Resp.T*(Fmodel - Data)/sig2, axis=1)
    #While the second derivative matrix grad(grad(C)) is useful, it is also
    #potentially a large array and at least requires a slow dot product
    #grad(grad(C)) = sum_k(2*R_kj*R_ki/sigma^2_k) = 2*R^T_kj.[I/sigma^2]_kk.R_ki (from Eq 8)
    #fortunately we can get around this by using dot products with 
    #1D matrices with R below, saving computation time and memory.

    return C0, gradC


def get_s_grads(n1, n2, ntot, Img, weights, defImg, defIpm, maxIff):
    #Finds the current, entropy, gradient in entropy,
    #and some additional entropy/image based quantities.
    
    gradS = np.zeros(ntot)
    gradgradS = np.zeros(ntot)
    fsupi  = np.zeros(ntot)
    #fsupi = f^i = -1/g_ii = -1/gradgradS from eq 18 and following paragraph

    # Calculate the current image entropy S (S0) (Eq 6), gradient of S (gradS) (Eq 7),
    # and diagonal elements of the second derivative matrix of S (gradgradS) (Eq 7) (off diagonal elements are zero). 

    #Entropy for brightness image
    gradS[0:n1] = weights[0:n1]*(np.log(defImg) - np.log(Img[0:n1]))  #(Eq 7a)
    gradgradS[0:n1] = -weights[0:n1]/Img[0:n1]       #(Eq 7b)
    fsupi[0:n1] = -1.0/gradgradS[0:n1]  # fsupi # f^i from eq 18
    #entStand = np.sum(-weights[0:n1]*(Img[0:n1]*(np.log(Img[0:n1]/defImg) - 1.0))) #(eq 6)
    entStand = np.sum(-weights[0:n1]*(Img[0:n1]*(np.log(Img[0:n1]/defImg) - 1.0) + defImg)) #normalized entropy
    Itot = np.sum(weights[0:n1]*Img[0:n1])

    #Entropy for filling factors (image with limited brightness)
    #from Collier Cameron (1992), Unruh & Collier Cameron (1995)  (Eq 4.)
    gradS[n1:n2]  = weights[n1:n2]*( np.log(defImg/Img[n1:n2]) - np.log((maxIff - defImg)/(maxIff - Img[n1:n2])) )
    gradgradS[n1:n2] = -weights[n1:n2]*maxIff/(Img[n1:n2]*(maxIff-Img[n1:n2]))
    fsupi[n1:n2] = -1.0/gradgradS[n1:n2]
    entFF = np.sum(-weights[n1:n2]*( Img[n1:n2]*np.log(Img[n1:n2]/defImg) + (maxIff - Img[n1:n2])*np.log((maxIff - Img[n1:n2])/(maxIff - defImg)) ))
    Itot += np.sum(weights[n1:n2]*Img[n1:n2])

    #Entropy for magnetic spherical harmonic coefficients (positive and negative values)
    #from Hobson & Lasenby (1998) (Eq. 8)
    tmpPsi = np.sqrt(Img[n2:ntot]**2 + 4.0*defIpm**2)
    gradS[n2:ntot] = -weights[n2:ntot]*np.log((tmpPsi + Img[n2:ntot])/(2.0*defIpm))
    gradgradS[n2:ntot] = -weights[n2:ntot]/tmpPsi
    fsupi[n2:ntot]  = -1.0 / gradgradS[n2:ntot]
    entMag = np.sum( weights[n2:ntot]*(tmpPsi - 2.0*defIpm - Img[n2:ntot]*np.log((tmpPsi+Img[n2:ntot])/(2.0*defIpm))) )
    Itot += np.sum(weights[n2:ntot]*np.maximum(np.abs(Img[n2:ntot]), defIpm))
    
    S0 = entStand + entFF + entMag

    return S0, gradS, gradgradS, fsupi, Itot, entStand, entFF, entMag


def searchDir(ntot, maxDir, Resp, sig2, fsupi, gradC, gradS, gradgradS):
    #Find nedir (up to maxDir) linearly independent search directions 
    #Start finding possible search directions
    # with Eqs 20, using Eq 18 and the paragraphs after it.
    # edir  = search directions (n fitting parameters, maxDir search directions)
    edir = np.zeros((ntot, maxDir))

    #Finds f(grad(C)) Then normalize the search direction. 
    #"1st search direction"  e_1 = f(grad(C)) = f^i dC/df^i
    #and here  gradC = grad(C) = dChi^2/df
    #and  fsupi = -1/grad(grad(entropy)) = -1/g_ii = f^i (Eq 18 and following 2 paragraphs)
    edir[:,0] = gradC*fsupi

    #sumNorm is used to normalize this search direction
    sumNorm = np.dot(edir[:,0], edir[:,0])
    if (sumNorm > 0.0):
        sumNorm = np.sqrt(sumNorm)
        err_gradCis0 = 0
    else:   #catch cases where chi^2 is unconstrained (grad(chi^2) = 0)
        sumNorm = 1.0
        err_gradCis0 = 1
    edir[:,0] /= sumNorm

    if (maxDir == 1): return  #if only using 1 search direction

    #Finds f(grad(S)) Then normalise the search direction. 
    #"2nd search direction"
    
    #2nd search direction e_2 = f(grad(S)) = f^i grad(S)
    #and here: gradS = grad(S) = dEntropy/dI
    #and  fsupi = -1/grad(grad(entropy)) = -1/g_ii = f^i (Eq 18 and following 2 paragraphs)
    edir[:,1] = gradS*fsupi

    sumNorm = np.dot(edir[:,1], edir[:,1])
    if (sumNorm > 0.0):
        sumNorm = np.sqrt(sumNorm)
        err_gradSis0 = 0
    else:  #catch cases where entropy is unconstrained (grad(entropy) = 0)
        sumNorm = 1.0
        err_gradSis0 = 1
    edir[:,1] /= sumNorm

    if ((err_gradCis0 == 1) & (err_gradSis0 ==1)):
        print("Error: f(gradS) and f(gradC) are both zero, problem not constrained")
        import sys
        sys.exit()
        
    #Loop over the remaining (maxDir-2) search directions.
    #This can be dome somewhat recursively since the array alternates between
    #terms based on chi^2 and entropy, and higher terms are just using
    #further f(grad(grad(C)))*x as Eqs 20
    
    #find the remaining up to maxDir (typically 4) search directions (eq 20)
    # e_3 = f(grad(grad(C))).f(grad(C)), e_4 = f(grad(grad(C))).f(grad(S))
    # e_5 = f(grad(grad(C))).f(grad(grad(C))).f(grad(C)), e_6 = f(grad(grad(C))).f(grad(grad(C)).f(grad(S))
    # generally: e_n = f(grad(grad(C))*e_(n-2)
    # For grad(grad(C)), start from Eq 8 we have:
    #      grad(C) = sum_k(R_kj*2*(F_k-D_k)/sigma^2_k) for response matrix R
    #  ->  grad(grad(C)) = sum_k(2*R_kj*R_ki/sigma^2_k) since dF_k/df_i = R_ki
    # Then for: f(grad(grad(C))) = f^i * 2*sum_k(R_kj*R_ki/sigma^2_k)
    #          with f^i = fsupi = -1/grad(grad(S))
    #          = 2*f^i * sum_k( R_ki*R_kj/sigma^2_k ) )  [dimensions of _ij]
    #   f(grad(grad(C))).f(grad(C)) = 2*f^i * sum_k( R_ki*R_kj/sigma^2_k ) ) . f(grad(C))^j
    #          = 2* f^i * sum_j (sum_k( R_ki*R_kj/sigma^2_k ) ) * f(grad(C))^j)
    #          = 2* sum_k ( f^i*R_ki/sigma^2_k * sum_j(*R_kj * f(grad(C))^j))
    #          = 2*f^i* (R_ki/sigma^2_k).(R_kj.f(grad(C))^j)
    #            this form keeps the math simple and avoids having to i,j,k all at once
    #as tensors = 2*f^i . R_ki^T . [I/sigma^2]_k^k . R_kj . f(grad(C))^j
    for i in range(2,maxDir):
        #These two dots are slow, since they are (nobs x nfitpars).(nfitpars)
        tempDot = np.dot(Resp, edir[:,i-2])
        edir[:,i] = 2.0*fsupi*np.dot(Resp.T/sig2, tempDot)

        sumNorm = np.dot(edir[:,i], edir[:,i])        
        if (sumNorm > 0.0):
            edir[:,i] /= np.sqrt(sumNorm)


    #Now diagonalize the search subspace and construct quadratic models
    edir, nedir, gamma  = diagDir(edir, maxDir, gradgradS, sig2, Resp)
    
    return edir, nedir, gamma


def diagDir(edir, nedir, gradgradS, sig2, Resp):

    # Diagonalize the search space as described in and Section 3.7.1. 
    # Find the current useful (i.e. linearly independent) search directions
    # among the set of e (edir) directions.  

    #First calculate the metric tensor g (eq 24b)
    #Note: this "g" = g_mu,nu is different from the g_i,j in the search and calc_grad routines.  
    #This metric is set up using the search directions e as the basis
    # and is defined by g_mu,nu = e^T_mu * e_nu (where ^T is the transpose)
    #The previous metric was diagonal and was:
    # g_i,j = 1/f^i (j=i) = -grad(grad(S)) = 1./fsupi [for the S=I*log(I) entropy]

    #from eq 24 (g_mu,nu = e^T_mu*e_nu)
    # (use -grad(grad(S)) as g_ij to make e_mu^T covarient,
    # multiply by the metric as the index lowering operation)
    g = np.dot((-gradgradS*edir.T), edir)
    
    #Diagonalize g. (Eigenvalues (gamma) are diagonals of the 
    #'diagonalized' matrix, with eigenvectors as the new basis)
    #The routine linalg.eigh is typically faster than linalg.eig, 
    #however linalg.eigh requires a symmetric or Hermitian matrix. 
    gamma, eiVec = linalg.eigh(g)
    
    #If any of the eigenvalues of g are small then throw out that direction
    #This protects against linear dependence in the search directions.
    #(Sect 3.7.1 par 1)   The cutoff value here is tested but arbitrary.
    maxGamma = np.max(np.abs(gamma))
    iok = gamma > 1e-8*maxGamma

    #Transform/project the search directions onto the new basis for diagonalized g
    edir = np.dot(edir, eiVec[:,iok])/np.sqrt(gamma[iok])
    nedir = edir.shape[1]
    
    #Here we also re-scaled the search directions edir so that the metric is Cartesian.
    #(in the tensor sense of Cartesian, so covariant and contravariant tensors are the same,
    # in that case the metric g_ij is the Kronecker delta_ij, roughly analogous to
    # the identity matrix.  Since the metric here is g_mu,nu = e^T_mu.e_nu,
    # and in the projected basis g_mu,nu is diagonal with g_mu,mu = gamma_mu
    # we can re-normalize e_mu by 1/sqrt(gamma_mu)  )
    
    #Now calculate M (held in matrix MM) [Eq 24d].
    # M_mu,nu = e^T_mu.grad(grad(C)).e_nu
    #   and: grad(grad(C)) = sum_k(2*R_kj*R_ki/sigma^2_k) ~~ 2*R^T_kj.[I/sigma^2]_kk.R_ki
    # we want to avoid having to do R^T.R since that is relatively slow, so some algebra:
    # M_mu,nu = e^T_mu,j.(2*R^T_kj.[I/sigma^2]_kk.R_ki).e_nu,i
    #         = 2*(R_kj.e_mu,j)^T.[I/sigma^2]_kk.(R_ki.e_nu,i)  #since B^T.A^T=(A.B)^T
    #This dot is slow, since it is (nobs x nfitpars).(nfitpars x nedir)
    tmpRe = np.dot(Resp,edir)
    MM = 2.0*np.dot((tmpRe.T/sig2), tmpRe)
    
    #Diagonalize M (eigenvalues of M are the diagonal of the diagonalized
    #matrix, by definition, with eigenvectors as the new basis)
    gammaM, eiVec = linalg.eigh(MM)

    #Transform/project the search directions onto the new basis that makes M diagonal.  
    edir = np.dot(edir, eiVec)

    return edir, nedir, gammaM


def getCmuSmu(gradC, gradS, edir):
    #Calculate Smu and Cmu (eqs 24a and 24c), as the projections
    #of gradC and gradS on to the new basis of search directions.  
    #For us in the quadratic models of ~C and ~S (see eqs 27).
    Cmu = np.dot(gradC, edir)
    Smu = np.dot(gradS, edir)
    return Cmu, Smu

def getL0squared(L_fac, Itot):
    #Find the limit on the allowed step size in parameter space for this
    #iteration L02 (paragraph above eq 28, actually l_0^2).
    # L_fac is input to the mem function, L_fac = 0.3 is an ok default
    # (typically use L_fac in 0.1 to 0.5 range) 
    L02 = L_fac*Itot
    return L02

def getAlphaMin(gamma):
    #Find alpha min (paragraph below eq 32). 
    #typically alphaMin = 0 for positive grad(grad(C)),
    # otherwise = max(-gamma(mu))
    alphaMin = 0.0
    minGamma = np.min(gamma)
    if(minGamma < 0.0):
        alphaMin = -minGamma
    return alphaMin 

def getCaimQuad(chiAim, C0, gamma, Cmu):
    #Find a good target value of C for this iteration
    #First find the minimum of the quadratic approximation to C, Cminq (eq 28). 
    # (derived from eq 27 ~C(x), the quadratic approximation for C,
    #  take x_mu which minimizes the eq, then substitute it 
    #  back into the equation to get ~C_min)
    Cminq = C0 - 0.5*np.sum(Cmu*Cmu/gamma)

    #Find the quadratic approximation for chiAim for this step, Caimq (Eq 29).  
    #Skilling and Bryan chose the constants 0.667 and 0.333. 
    #Alternate reasonable constants are 0.8 and 0.2
    Caimq = 0.66666667*Cminq + 0.33333333*C0
    if (chiAim > Caimq):
        Caimq = chiAim
    return Caimq


#For the control subroutine:
#Chop alpha or P down towards the lower limit
def chopDown(alpha, alphaLow):
    alphaHigh = alpha
    alpha = 0.5*(alphaLow + alpha)
    return alpha, alphaHigh

#For the control subroutine:
#Chop alpha or P up towards infinity
def chopUp(alpha, alphaHigh):
    alphaLow = alpha
    if (alphaHigh > 0.0):
        alpha = 0.5*(alphaHigh + alpha)
    else:
        alpha = 2.0*alpha + 0.1
    return alpha, alphaLow

def control(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin):    
    #Implements the control procedure schematically shown in Fig. 3
    #and described in Sections 3.7.2 and 3.7.3. 

    #convergence tolerance for alpha and P
    #if the code is stalling at a not quite good enough test,
    #then decreasing this may occasionally help (but may be slower)
    convTol = 1e-5
    
    #allow chi^2 to relax slightly if at the target value
    if (C0 < Caimq*1.001):  
        Caimr = Caimq*1.001
    else:
        Caimr = Caimq

    P = Plow = 0.0
    Phigh = 0.0
    Pfinished = 0
    itera = 0
    iterP = 0
    #Loop for P chop, search for P such that Caimq <= Cq < Cqp <= C0
    #and try to minimize P.
    while (Pfinished == 0):
        alphaLow = alphaMin
        alphaHigh = -1.0
        alpha = alphaMin + 1.0
        afinished = 0
        #Loop for the alpha chop, search for alpha such that Caimq <= Cq < Cqp <= C0
        while (afinished == 0):
            asuccess = 0
            #Calculate x_mu, ~C (Cq), ~Cp (C_qp), and l^2
            #from: C (Eq 27b, from 23b), l^2 (Eq 27c), x (Eq 34), Cp (Eq 35).
            xqp = (alpha*Smu - Cmu)/(P + gamma + alpha)
            Cq = C0 + np.dot(Cmu, xqp) + 0.5*np.einsum('i,i,i', gamma,xqp,xqp)
            Cqp = C0 + np.dot(Cmu,xqp) + 0.5*np.einsum('i,i,i', P+gamma,xqp,xqp)
            L2 = np.dot(xqp, xqp)
            
            #If next (quadratic approx) chi^2 is larger (and significantly above the target)
            #(there are some cases where chi^2 is slightly worse, but only 
            # slightly above target, and it seems to be more efficient 
            # or stable to allow those)
            if ((Cqp > C0) & (Cqp > Caimr)): 
                #chop alpha down towards alphaMin (alpha_min) (use less entropy)
                alpha, alphaHigh = chopDown(alpha, alphaLow)
            #if chi^2 is better than the target
            #maybe just elif ((Cq < Caimq)):,
            #depends on if the case where C0 < Cq < Caimq is 'good enough'
            elif ((Cq < Caimq) & (Cq < C0)):
                #chop alpha up towards infinity (use more entropy)
                alpha, alphaLow = chopUp(alpha, alphaHigh) 
            #the step is too large redirect alpha chop towards Cqp = C0
            elif (L2 > L02):
                #if  Cqp is better than C0, chop alpha up
                #(use more entropy) (increases Cpq towards C0)
                if (Cqp < C0):
                    alpha, alphaLow = chopUp(alpha, alphaHigh) 
                #otherwise chop alpha down (use less entropy)
                #(decrease Cpq towards C0)
                else:
                    alpha, alphaHigh = chopDown(alpha, alphaLow)
            #if C in desired range, call this step an improvement in alpha.
            # Caimq <= Cq & Cqp <= C0
            # Caimq <= Cq & Cpq <= Caimr
            #    C0 <= Cq & Cpq <= Caimr
            # (generally Cq <= Cqp for positive P, and P should be >=0)
            else:  
                asuccess = 1
                ##S&B flowchart: and chop alpha down
                #alpha, alphaHigh = chopDown(alpha, alphaLow)
                #It may be more efficient in rare cases to chop alpha
                #so that Cq goes towards Caimq (mostly this will be down)
                if(Cq < Caimq):
                    alpha, alphaLow = chopUp(alpha, alphaHigh) 
                else:
                    alpha, alphaHigh = chopDown(alpha, alphaLow)
            
            #If alphaHigh ~= alphaLow we've converged on an alpha
            if ( (alphaHigh > 0.0) & (abs(alphaHigh - alphaLow) < (convTol*alphaHigh + 1.0E-10)) ):
                afinished = 1
            #halt if alpha is badly unconstrained
            if (alpha > 1.0E20):
                afinished = 1
            itera += 1
        #End of alpha chop loop        
        
        #if we did not find a good enough alpha, increase P
        if (asuccess != 1):  
            P, Plow = chopUp(P, Phigh)
        #if we found a good enough alpha, decrease P (if 0 will stay at 0)
        else:
            #If we have a converged P and good enough alpha
            if ( (P == 0.0) | (abs(Phigh - Plow) < convTol*Phigh + 1.0E-10) ):
                Pfinished = 1
            else:
                P, Phigh = chopDown(P, Plow)

        #Check for cases with no suitable P. Generally this 
        #should not happened, and there is an an error elsewhere.
        if ((asuccess == 0) & (P > 1.0E20)):
            Pfinished = 1               
            print("P chop blow up encountered.")
        iterP += 1
    # End of P chop loop
    #print('nP', iterP, 'P', P, 'na', itera, 'a', alpha)

    #return the final (successful) x
    return xqp


def updateImg(xq, edir, Img, n1, n2, ntot, maxIff):
    #Update the image array (eq 25).
    # Image(new) = I + deltaI = I + x^nu*e_nu
    Img += np.inner(xq, edir)
    
    #Protect against stray negative image values.
    #(only for the first n1 image elements. For brightness with regular entropy)
    intMod = np.where(Img[:n1] <= 0.0)
    Img[intMod] = 1.0E-6
    
    #(for the next n1-n2 elements, protect against negative or too large values. For filling factor entropy)
    intMod = np.where(Img[n1:n2] <= 0.0)
    Img[intMod] = 1.0E-6*maxIff
    intMod = np.where(Img[n1:n2] >= maxIff)
    Img[intMod] = maxIff*(1.0-1.0E-6)
    #elements between n2 and ntot can have any value (usually the SHD coefficients for ZDI)
    return Img


def get_test(gradC, gradS):
    #Calculate Skilling and Bryan's test parameter (eq 37).
    #(how anti-parallel the gradients in entropy and chi^2 are.)
    mag_gradS = np.sqrt(np.sum(gradS**2))
    mag_gradC = np.sqrt(np.sum(gradC**2))
    if (mag_gradS == 0.0):
        inv_mag_gradS = 0.
    else:
        inv_mag_gradS = 1./mag_gradS
    if (mag_gradC == 0.0):
        inv_mag_gradC = 0.
    else:
        inv_mag_gradC = 1./ mag_gradC
    test = 0.5*np.sum( (gradS*inv_mag_gradS - gradC*inv_mag_gradC)**2 )
    
    return test



#Convert the observed data, and uncertainties, into 1D format used by mem_iter
def packDataVector(obsSet, fitBri, fitMag):

    Data = np.empty(0)
    sig2 = np.empty(0)
    if (fitBri == 1):
        for tmpObs in obsSet:
            Data = np.append(Data, tmpObs.specI)
            sig2 = np.append(sig2, tmpObs.specIsig**2)
    if (fitMag == 1):
        for tmpObs in obsSet:
            Data = np.append(Data, tmpObs.specV)
            sig2 = np.append(sig2, tmpObs.specVsig**2)

    return Data, sig2

def packResponseMatrix(setSynSpec, nDataTot, npBriMap, magGeom, magGeomType, fitBri, fitMag):

    #Setup the array for I in brightness
    if (fitBri == 1):
        nDataUsed = 0
        allModeldI = np.zeros([nDataTot, npBriMap])
        for spec in setSynSpec:
            #save the derivatives of I wrt brightness
            allModeldI[nDataUsed:nDataUsed+spec.numPts,:] = spec.dIIc.T
            nDataUsed += spec.numPts

    #Setup the array for V in magnetic coefficients
    if (fitMag == 1):
        #A little sanity check:
        if magGeom.magGeomType != magGeomType:
            print('ERROR: miss-match in magGeomType flags!')
            import sys
            sys.exit()
        
        nDataUsed = 0
        nMagCoeff = 0
        if magGeom.magGeomType == 'full':
            nMagCoeff = 3*2*magGeom.nTot
        elif magGeom.magGeomType == 'poloidal' or magGeom.magGeomType == 'pottor':
            nMagCoeff = 2*2*magGeom.nTot
        elif magGeom.magGeomType == 'potential':
            nMagCoeff = 1*2*magGeom.nTot
        allModeldV_lin = np.zeros((nDataTot, nMagCoeff))
        
        for spec in setSynSpec:
            #save the derivatives of V wrt magnetic coefficients, for the used coefficients
            nDataUsedNext = nDataUsed + spec.numPts
            if magGeom.magGeomType == 'full':
                for a in range(3):  #loop over alpha, beta, gamma
                    allModeldV_lin[nDataUsed:nDataUsedNext,
                                   a*2*magGeom.nTot : (a*2+1)*magGeom.nTot] \
                                = np.real(spec.dVIc[a,:,:]).T
                    allModeldV_lin[nDataUsed:nDataUsedNext,
                                   (a*2+1)*magGeom.nTot : (a*2+2)*magGeom.nTot] \
                                = np.imag(spec.dVIc[a,:,:]).T
            elif magGeom.magGeomType == 'poloidal' or magGeom.magGeomType == 'pottor':
                for a in range(2):
                    allModeldV_lin[nDataUsed:nDataUsedNext, 
                                   a*2*magGeom.nTot : (a*2+1)*magGeom.nTot] \
                                = np.real(spec.dVIc[a,:,:]).T
                    allModeldV_lin[nDataUsed:nDataUsedNext,
                                   (a*2+1)*magGeom.nTot : (a*2+2)*magGeom.nTot] \
                                = np.imag(spec.dVIc[a,:,:]).T
            elif magGeom.magGeomType == 'potential':
                allModeldV_lin[nDataUsed:nDataUsedNext, 0 : magGeom.nTot] \
                    = np.real(spec.dVIc[0,:,:]).T
                allModeldV_lin[nDataUsed:nDataUsedNext, magGeom.nTot : 2*magGeom.nTot] \
                    = np.imag(spec.dVIc[0,:,:]).T
            
            nDataUsed += spec.numPts

    #Setup the array for V in brightness changes
    if ((fitBri == 1) & (fitMag == 1)):
        nDataUsed = 0
        allModeldVdBri = np.zeros([nDataTot, npBriMap])
        for spec in setSynSpec:
            #save the derivatives of V wrt brightness
            allModeldVdBri[nDataUsed:nDataUsed+spec.numPts,:] = spec.dVdBri.T
            nDataUsed += spec.numPts
    
    #concatenate the two sets of derivatives (or just return one set)
    if ((fitBri == 1) & (fitMag == 1)):
        
        allModeldIdV = np.zeros((nDataTot*2, npBriMap + nMagCoeff))
        #dI by dBrightness
        allModeldIdV[0:nDataTot, 0:npBriMap] = allModeldI
        #dI by dMagnetic
        #allModeldIdV[0:nDataTot, npBriMap:npBriMap+nMagCoeff] = 0.
        #dV by dBrightness
        allModeldIdV[nDataTot:nDataTot*2, 0:npBriMap] = allModeldVdBri
        #dV by dMagnetic
        allModeldIdV[nDataTot:nDataTot*2, npBriMap:npBriMap+nMagCoeff] = allModeldV_lin
    elif (fitBri == 1):
        allModeldIdV = allModeldI
    elif (fitMag == 1):
        allModeldIdV = allModeldV_lin
    else:
        print('ERROR: got no fittable parameters in packResponseMatrix()')
    
    return allModeldIdV

# Merge the set of observed spectra (I and/or V) into one 1D array
def packModelVector(setSynSpec, fitBri, fitMag):
    allModelI = np.empty(0)
    allModelV = np.empty(0)
    if (fitBri == 1):
        for spec in setSynSpec:
            allModelI = np.append(allModelI, spec.IIc)
    if (fitMag == 1):
        for spec in setSynSpec:
            allModelV = np.append(allModelV, spec.VIc)

    allModelIV = np.concatenate((allModelI, allModelV))

    return allModelIV


#Convert the set of model (fitting) parameters into a 1D "image" vector for passing to mem_iter
def packImageVector(briMap, magGeom, magGeomType, fitBri, fitMag):
    #A little sanity check:
    if magGeom.magGeomType != magGeomType:
        print('ERROR: miss-match in magGeomType flags!')
        import sys
        sys.exit()
    
    Image = np.empty(0)
    if (fitBri == 1):
        Image = briMap.bright
    #All three alpha beta and gamma exist in the magnetic geometry object,
    #but here we only use the ones that are fit, then in upackImageVector
    #the coefficients that are not fit are set based on the type of geometry assumed.
    if (fitMag == 1):
        if magGeomType == 'full':
            Image = np.concatenate([Image, 
                        np.real(magGeom.alpha), np.imag(magGeom.alpha), 
                        np.real(magGeom.beta), np.imag(magGeom.beta),   
                        np.real(magGeom.gamma), np.imag(magGeom.gamma)]  )
        elif magGeomType == 'poloidal':
            Image = np.concatenate([Image, 
                        np.real(magGeom.alpha), np.imag(magGeom.alpha), 
                        np.real(magGeom.beta), np.imag(magGeom.beta)]  )
        elif magGeomType == 'pottor':
            Image = np.concatenate([Image, 
                        np.real(magGeom.alpha), np.imag(magGeom.alpha), 
                        np.real(magGeom.gamma), np.imag(magGeom.gamma)]  )
        elif magGeomType == 'potential':
            Image = np.concatenate([Image, 
                        np.real(magGeom.alpha), np.imag(magGeom.alpha)]  )

    return Image


#Convert a 1D "image" vector from mem_iter back into the normal format for storing parameters used elsewhere.
def unpackImageVector(Image, briMap, magGeom, magGeomType, fitBri, fitMag):
    #A little sanity check:
    if magGeom.magGeomType != magGeomType:
        print('ERROR: miss-match in second magGeomType flags!')
        import sys
        sys.exit()

    npBriMap = briMap.bright.shape[0]
    
    endBri = 0
    if (fitBri == 1):
        briMap.bright[:] = Image[0:npBriMap]
        endBri = npBriMap

    #If only some magnetic geometry coefficients are free 
    #(due to an assumed type of geometry)
    #then calculate the fixed values from the free values.
    if (fitMag == 1):
        if magGeomType == 'full':
            magGeom.alpha = Image[endBri:endBri+magGeom.nTot] + \
                            Image[endBri+magGeom.nTot:endBri+magGeom.nTot*2]*1.j
            magGeom.beta  = Image[endBri+magGeom.nTot*2:endBri+magGeom.nTot*3] + \
                            Image[endBri+magGeom.nTot*3:endBri+magGeom.nTot*4]*1.j
            magGeom.gamma = Image[endBri+magGeom.nTot*4:endBri+magGeom.nTot*5] + \
                            Image[endBri+magGeom.nTot*5:endBri+magGeom.nTot*6]*1.j
        elif magGeomType == 'poloidal':
            magGeom.alpha = Image[endBri:endBri+magGeom.nTot] + \
                            Image[endBri+magGeom.nTot:endBri+magGeom.nTot*2]*1.j
            magGeom.beta  = Image[endBri+magGeom.nTot*2:endBri+magGeom.nTot*3] + \
                            Image[endBri+magGeom.nTot*3:endBri+magGeom.nTot*4]*1.j
            magGeom.gamma[:] = 0. + 0.j
        elif magGeomType == 'pottor':
            magGeom.alpha = Image[endBri:endBri+magGeom.nTot] + \
                            Image[endBri+magGeom.nTot:endBri+magGeom.nTot*2]*1.j
            magGeom.beta  = Image[endBri:endBri+magGeom.nTot] + \
                            Image[endBri+magGeom.nTot:endBri+magGeom.nTot*2]*1.j
            magGeom.gamma = Image[endBri+magGeom.nTot*2:endBri+magGeom.nTot*3] + \
                            Image[endBri+magGeom.nTot*3:endBri+magGeom.nTot*4]*1.j
        elif magGeomType == 'potential':
            magGeom.alpha = Image[endBri:endBri+magGeom.nTot] + \
                            Image[endBri+magGeom.nTot:endBri+magGeom.nTot*2]*1.j
            magGeom.beta  = Image[endBri:endBri+magGeom.nTot] + \
                            Image[endBri+magGeom.nTot:endBri+magGeom.nTot*2]*1.j
            magGeom.gamma[:] = 0. + 0.j

    return


class constantsMEM:
    #Hold some control constants for the mem_iter routine
    def __init__(self, par, briMap, magGeom, nDataTot):
        #setup control constants for input to mem_iter
        self.npBriMap = 0
        self.npMagGeom = 0
        self.nDataTotIV = 0
        if (par.fitBri == 1):
            self.npBriMap = briMap.bright.shape[0]
            self.nDataTotIV += nDataTot
        if (par.fitMag == 1):
            if magGeom.magGeomType == 'full':
                self.npMagGeom = magGeom.nTot*6
            elif magGeom.magGeomType == 'poloidal'or magGeom.magGeomType == 'pottor':
                self.npMagGeom = magGeom.nTot*4
            elif magGeom.magGeomType == 'potential':
                self.npMagGeom = magGeom.nTot*2
            else:
                self.npMagGeom = magGeom.nTot*6
            self.nDataTotIV += nDataTot
        self.n1Model = self.npBriMap
        self.n2Model = self.n1Model
        if (par.fEntropyBright == 2):
            #If using "filling factor" entropy (limits brightness to >0, <1)
            self.n1Model = 0
            self.n2Model = self.npBriMap
        self.nModelTot = self.n2Model + self.npMagGeom


# Weight applied to the entropy terms (and their derivatives) in mem_iter
# typically it is set to the l order of the harmonic coefficient
def setEntropyWeights(par, magGeom, sGrid):
    weightEntropy = 0
    
    #weightEntropyV = np.ones(magGeom.nTot*6)  #alternately use no weighting (everything = 1)
    if magGeom.magGeomType == 'poloidal':
        weightEntropyV = np.tile(magGeom.l, 4)
    if magGeom.magGeomType == 'pottor':
        weightEntropyV = np.tile(magGeom.l, 4)
    if magGeom.magGeomType == 'potential':
        weightEntropyV = np.tile(magGeom.l, 2)
    else:
        weightEntropyV = np.tile(magGeom.l, 6)
    
    #Allow for a difference in relative scaling of entropy between
    # magnetic field and brightness.
    weightEntropyI = sGrid.area*par.brightEntScale
    if(par.fitBri == 1):
        weightEntropy = weightEntropyI
    if(par.fitMag == 1):
        weightEntropy = weightEntropyV
    if((par.fitMag == 1) & (par.fitBri == 1)):
        weightEntropy = np.concatenate((weightEntropyI,weightEntropyV))
    return weightEntropy
