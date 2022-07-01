#The modified functions needed to run the maximum entropy method fitting (mem_*),
#but fitting with the constraint that entropy is >= Saim, rather than chi^2 <= chiAim.
#Most functions are the same (they don't care about the target),
#however a new control routine is needed, and we need the constant Saimq rather than Caimq

import numpy as np

def getSaimQuad(Saim, S0, gamma, Smu):
    #Find a good target value of S for this iteration
    #First find the maximum of the quadratic approximation to S, Smaxq (similar to Eqn 28)
    # (derived from eq 27 ~S(x), the quadratic approximation for S,
    #  take x_mu which maximizes the eq, then substitute it 
    #  back into the equation to get ~S_max)
    # [S_max = S0 + 0.5*Smu*Smu]
    Smaxq = S0 + 0.5*np.sum(Smu*Smu)

    #Find the quadratic approximation for Saim for this step, Saimq (like Eq 29).  
    #Skilling and Bryan chose the constants 0.667 and 0.333. 
    #Alternate reasonable constants are 0.8 and 0.2
    Saimq = 0.66666667*Smaxq + 0.33333333*S0
    if (Saim < Saimq):
        Saimq = Saim
    return Saimq

#Note, for target entropy possibly rephrase the problem to Q = S - lambda*C,
#rather than Q = alpha*S - C.  Then update lambda rather than alpha
#as the code goes, but that requires a bigger changes to the code...


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

def control(S0, gamma, Cmu, Smu, Saimq, L02, alphaMin):    
    #Implements the control procedure schematically shown in Fig. 3
    #and described in Sections 3.7.2 and 3.7.3.
    #But rephrased for entropy rather than chi^2

    #convergence tolerance for alpha and P
    #if the code is stalling at a not quite good enough test,
    #then decreasing this may occasionally help (but may be slower)
    convTol = 1e-5
    
    #allow entropy to relax slightly if at the target value
    if (S0 > Saimq*1.001):  
        Saimr = Saimq*1.001
    else:
        Saimr = Saimq

    P = Plow = 0.0
    Phigh = 0.0
    Pfinished = 0
    itera = 0
    iterP = 0
    #Loop for P chop, search for P such that Saimq >= Sq > Sqp >= S0
    #and try to minimize P.
    while (Pfinished == 0):
        alphaLow = alphaMin
        alphaHigh = -1.0
        alpha = alphaMin + 1.0
        afinished = 0
        #Loop for the alpha chop, search for alpha such that Saimq >= Sq > Sqp <= S0
        while (afinished == 0):
            asuccess = 0

            #Calculate x_mu (xqp), ~S (Sq), ~Sp (Sqp), and l^2 (L2)
            #from: xqp (Eq 34), L2 (Eq 27c), Sq (Eq 27b)
            xqp = (alpha*Smu - Cmu)/(P + gamma + alpha)
            L2 = np.dot(xqp, xqp)
            Sq = S0 + np.dot(Smu, xqp) - 0.5*np.dot(xqp, xqp)
            #For deriving Sqp, take eq 33 (Q = alpha*Sq - Cq - P*l^2)
            #substitute in Sq and l^2 from eq 27a and 27c, then group 
            #the terms together to make one term Sqp, including P in it.
            #Note, by my math, there is a difference of P vs 2*P between
            #eq 33 and eq 34 or 35, which also shows up here
            #(i.e. 2P from eq 33 => P here and elsewhere). 
            Sqp = S0 + np.dot(Smu, xqp) - 0.5*(1.+P/alpha)*np.dot(xqp, xqp)
            
            #If next (quadratic approx) entropy is smaller (and significantly
            #below the target) (there are some cases where entropy is slightly
            # worse, but only slightly below target, and it seems to be more 
            # efficient or stable to allow those)
            if ((Sqp < S0) & (Sqp < Saimr)): 
                #chop alpha up towards infinity (use more entropy)
                alpha, alphaLow = chopUp(alpha, alphaHigh) 
            #if entropy is larger than the target (and the last iteration)
            elif ((Sq > Saimq) & (Sq > S0)):
                #chop alpha down towards alphaMin (alpha_min) (use less entropy)
                alpha, alphaHigh = chopDown(alpha, alphaLow)
                                
            #the step is too large redirect alpha chop towards Sqp = S0
            elif (L2 > L02):
                #if (modified) entropy is better than S0, chop alpha down
                #(use less entropy) (decrease Spq towards S0)
                if (Sqp > S0):
                    alpha, alphaHigh = chopDown(alpha, alphaLow)
                #otherwise chop alpha up (use more entropy)
                #(increase Spq towards S0)
                else:
                    alpha, alphaLow = chopUp(alpha, alphaHigh) 
            #if S in desired range, call this step an improvement in alpha.
            # Saimq > Sq & Sqp > S0
            # Saimq > Sq & Spq > Saimr
            #    S0 > Sq & Spq > Saimr
            # (generally Sq >= Sqp for positive P, and P should be >=0)
            else:
                asuccess = 1
                #Ok range, still chop Sq towards target Saimq
                if(Sq < Saimq):
                    #if below target chop alpha up (weight to more entropy)
                    alpha, alphaLow = chopUp(alpha, alphaHigh)
                else:
                    #This is most important for the first few iterations
                    # of decreasing entropy where S0 > Sq > Saimq.
                    alpha, alphaHigh = chopDown(alpha, alphaLow)
            
            #If alphaHigh ~= alphaLow we've converged on an alpha
            if ( (alphaHigh > 0.0) & (abs(alphaHigh - alphaLow) < (convTol*alphaHigh + 1.0E-10)) ):
                afinished = 1
                
            #First iteration, if initialized to zero entropy,
            # just weight heavily for chi^2 fitting
            if (S0 >= 0.):
                alpha = alphaMin
                asuccess = 1
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
