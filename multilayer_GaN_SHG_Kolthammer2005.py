#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:30:00 2025

@author: quadrupole
"""

# Uses formalism of Kolthammer 2005 to calculate SHG from thin film of c-plane GaN on sapphire
# Additionally, allows for polarization-dependence calculations on input and output
# Ignore surface contributions; we assume they're << bulk contributions for GaN
    # because the film thickness is >> lambda/4pi = 86 nm (Boyd pg 120)
# Coherence length in GaN is lambda0/(4*|Delta(n)|) = 1.525 um
# As of 2025-09-08, the results of this code more-or-less match results of Miragliotta 1993

'''
Note that currently, there are minus signs applied to the x and y components 
of the pump field to account for the fact that a wave incident from phi=0 has
negative x and y components. However, this changes the results (especially
alpha- and theta- dependence), which shouldn't be the case if SHG from GaN is 
azmuthally-independent. '
'''

import numpy as np
import matplotlib.pyplot as plt 

c = 3e8 # Speed of light (m/s) 
#epsilon0 = 8.854e-12 # Permittivity of free space (F/m) 

# relative dielectric constant of layers at SHG frequency Omega and pump frequency omega
# Code assumes 3 layers 
air = {'540 nm':1.00027804**2, '1080 nm':1.00027394**2}
GaN = {'540 nm':2.4194**2, '1080 nm':2.3322**2}
sapp = {'540 nm':1.7711**2, '1080 nm':1.7542**2}
epsilon_Omega = np.array([sapp['540 nm'], GaN['540 nm'], air['540 nm']]) # 540 nm 
epsilon_omega = np.array([sapp['1080 nm'], GaN['1080 nm'], air['1080 nm']]) # 1080 nm
# =============================================================================
# epsilon_Omega = np.array([air['540 nm'], GaN['540 nm'], sapp['540 nm']]) # 540 nm 
# epsilon_omega = np.array([air['1080 nm'], GaN['1080 nm'], sapp['1080 nm']]) # 1080 nm
# =============================================================================


def ER(omega, theta, d, e0, pol):
    # omega is pump frequency
    # theta is polar angle of incident light, measured from normal, in radians 
    # d is thickness of GaN layer
    # e0 is a 2-element dict giving the magnitude of the pump field in s/p coordinates;
    # its assumed that phi=0, such that an s-pol pump field is in the y direction
    # note that this assumption only works because SHG from GaN is azimuthally invariant
    # Returns the magnitude of the [pol]-pol SHG field in reflection geometry 

    n = 2  # Harmonic number

    kappa = omega/c * np.sin(theta)
    K = n * kappa
    # The following four are arrays, indexed by j (layer number)
    W = np.sqrt((n*omega/c)**2 * epsilon_Omega - K**2)
    w = np.sqrt((omega/c)**2 * epsilon_omega - kappa**2)
    ts = np.array([2*w[0]/(w[0]+w[1]), 2*w[1]/(w[1]+w[2])])
    rs = np.array([(w[0]-w[1])/(w[0]+w[1]), (w[1]-w[2])/(w[1]+w[2])])
    tp = np.array([2*np.sqrt(epsilon_omega[0]*epsilon_omega[1])*w[0] / (epsilon_omega[1]*w[0] + epsilon_omega[0]*w[1]),
                   2*np.sqrt(epsilon_omega[1]*epsilon_omega[2])*w[1] / (epsilon_omega[2]*w[1] + epsilon_omega[1]*w[2])])
    rp = np.array([(epsilon_omega[1]*w[0] - epsilon_omega[0]*w[1]) / (epsilon_omega[1]*w[0] + epsilon_omega[0]*w[1]),
                   (epsilon_omega[2]*w[1] - epsilon_omega[1]*w[2]) / (epsilon_omega[2]*w[1] + epsilon_omega[1]*w[2])])

    def mlayer_s(j):
        return np.array([[np.exp(1j*w[j]*d), 0], [0, np.exp(-1j*w[j]*d)]])

    def minterface_s(j):
        return 1/ts[j] * np.array([[1, rs[j]], [rs[j], 1]])
    
    def mlayer_p(j): # Identical to mlayer_s 
        return np.array([[np.exp(1j*w[j]*d), 0], [0, np.exp(-1j*w[j]*d)]])

    def minterface_p(j): # Different from minterface_s 
        return 1/tp[j] * np.array([[1, rp[j]], [rp[j], 1]])

    def e(m, z):
        # Returns a 2x1 vector containing the sum of m e+ fields and (n-m) e- fields 
        # at depth z in the GaN layer (measured from air)
        a_s = minterface_s(0) @ mlayer_s(1) @ minterface_s(1)
        a_p = minterface_p(0) @ mlayer_p(1) @ minterface_p(1) 
        e_plus = {}
        e_minus = {}
        #z = d # for testing 
        m_layer_z = np.array([[np.exp(1j*w[1]*z), 0], [0, np.exp(-1j*w[1]*z)]])
        
        e_plus['s'] = (m_layer_z @ minterface_s(1) @ np.array([0, e0['s']/a_s[1, 1]]))[0]
        e_minus['s'] = (m_layer_z @ minterface_s(1) @ np.array([0, e0['s']/a_s[1, 1]]))[1]        
        e_plus['p'] = (m_layer_z @ minterface_p(1) @ np.array([0, e0['p']/a_p[1, 1]]))[0]
        e_minus['p'] = (m_layer_z @ minterface_p(1) @ np.array([0, e0['p']/a_p[1, 1]]))[1] 
        #print("es+ = " + str(e_plus['s']) + "\nep+ = " + str(e_plus['p']) + "\nes- = " + str(e_minus['s']) + "\nep- = " + str(e_minus['p']))
        return {'s' : m*e_plus['s'] + (n-m)*e_minus['s'], 'p' : m*e_plus['p'] + (n-m)*e_minus['p']}
    
    def q(m, j):
        return (2*m-n)*w[j]
    
    def Ps(m, j, z): # Nonlinear polarization equation, with chi2 = 1
        if j == 1: 
            ey = -e(m, z)['s'] 
            ez = kappa**2/(kappa**2 + w[j]**2)*e(m, z)['p'] 
            #print("Ps = " + str(2*ey*ez))
            return 2*ey*ez 
        else: return 0 # assumes SHG only happens in the GaN layer 
    
    def Pkappa(m, j, z): # Nonlinear polarization equation, with chi2 = 1
        if j == 1: 
            ex = -w[j]**2/(kappa**2 + w[j]**2)*e(m,z)['p'] 
            ez = kappa**2/(kappa**2 + w[j]**2)*e(m,z)['p'] 
            #print("Pkappa = " +str(2*ex*ez))
            return -2*ex*ez  # Equivalent to Px (minus sign because kappa direction flips upon reflection)
        else: return 0 # assumes SHG only happens in the GaN layer 
    
    def Pz(m, j, z): # Nonlinear polarization equation, with chi2 = 1
        if j == 1: 
            ex = -w[j]**2/(kappa**2 + w[j]**2)*e(m,z)['p'] 
            ey = e(m, z)['s'] 
            ez = kappa**2/(kappa**2 + w[j]**2)*e(m,z)['p'] 
            #print("Pz = " + str((ex**2+ey**2) - 2*ez**2))
            return (ex**2+ey**2) - 2*ez**2 
        else: return 0 # assumes SHG only happens in the GaN layer 
        
    def As(m, j, z):
        Cj = -4*np.pi / (epsilon_Omega[j] * (W[j]**2 - q(m,j)**2))
        #print("As = " + str(Cj*epsilon_Omega[j]*(n*omega/c)**2 * Ps(m, j, z)))
        return Cj*epsilon_Omega[j]*(n*omega/c)**2 * Ps(m,j,z)

    def Az(m, j, z):
        Cj = -4*np.pi / (epsilon_Omega[j] * (W[j]**2 - q(m,j)**2))
        #print("Az = " + str(Cj*((epsilon_Omega[j]*(n*omega/c)**2 - q(m,j)**2) * Pz(m,j, z) - q(m,j)*n*kappa*Pkappa(m,j, z) )))
        return Cj*((epsilon_Omega[j]*(n*omega/c)**2 - q(m,j)**2) * Pz(m,j,z) - q(m,j)*n*kappa*Pkappa(m,j,z) )
    
    def Akappa(m, j, z):
        Cj = -4*np.pi / (epsilon_Omega[j] * (W[j]**2 - q(m,j)**2)) 
        #print("Akappa = " + str(Cj*(W[j]**2 * Pkappa(m,j) - q(m,j)*n*kappa*Pz(m,j))))
        return Cj*(W[j]**2 * Pkappa(m,j,z) - q(m,j)*n*kappa*Pz(m,j,z))
    
    def Ss(sign, j, z):
        total = 0
        for m in np.arange(n+1):
            # Ignore surface contributions; we assume they're << bulk contributions for GaN
            # because the film thickness is >> lambda/4pi (Boyd pg 120)
            total += As(m,j+1,z)*(1+sign*q(m,j+1)/W[j]) - As(m,j,z)*(1+sign*q(m,j)/W[j]) 
        #print("Ss = " + str(1/2*total))
        return 1/2 * total 
    
    def Sp(sign, j, z):
        total = 0
        for m in np.arange(n+1):
            # Ignore surface contributions; we assume they're << bulk contributions for GaN
            # because the film thickness is >> lambda/4pi (Boyd pg 120)
            total += n*kappa*(Az(m,j+1,z)-Az(m,j,z)) - Akappa(m,j+1,z)*(q(m,j+1)+sign*epsilon_Omega[j]*(n*omega/c)**2/W[j]) + Akappa(m,j,z)*(q(m,j)+sign*epsilon_Omega[j]*(n*omega/c)**2/W[j]) 
        #print("Sp = " + str(1/(2*np.sqrt(epsilon_Omega[j])*(n*omega/c)**2) * total ))
        return 1/(2*np.sqrt(epsilon_Omega[j])*(n*omega/c)) * total 

    def Mslayer(j):
        return np.array([[np.exp(1j*W[j]*d), 0, 0], [0, np.exp(-1j*W[j]*d), 0], [0, 0, 1]])

    def Mplayer(j):
        return np.array([[np.exp(1j*W[j]*d), 0, 0], [0, np.exp(-1j*W[j]*d), 0], [0, 0, 1]])

    def Msinterface(j, z):
        return np.array([[1/ts[j], rs[j]/ts[j], Ss(+1, j, z)], [rs[j]/ts[j], 1/ts[j], Ss(-1, j, z)], [0, 0, 1]])

    def Mpinterface(j, z):
        return np.array([[1/tp[j], rp[j]/tp[j], Sp(+1, j, z)], [rp[j]/tp[j], 1/tp[j], Sp(-1, j, z)], [0, 0, 1]])
    
    
    # Calculate ER from each depth in the GaN
    # Sum the resultant SHG fields and return the result 
    E_total = 0
    ZI = 1000 # This takes a while, but gives more accurate results 
    #ZI = 200 #for fast and dirty computations
    for zi in np.array([d/2]): #np.linspace(0,d, ZI): #
        z = zi 
        # Calculate matrix M
        # The @ operator is shorthand for np.matmul
        if pol == 's':
            M = Msinterface(0,z) @ Mslayer(1) @ Msinterface(1,z)
        elif pol == 'p':
            M = Mpinterface(0,z) @ Mplayer(1) @ Mpinterface(1,z)
        #print(M) 
        E_total += M[0,2] - M[1,2]/M[1,1] 
    
    return E_total 
    #return M[0,2] - M[1,2]/M[1,1] 

def ET(omega, theta, d, e0, pol):
    # omega is pump frequency
    # theta is polar angle of incident light, measured from normal
    # d is thickness of GaN layer
    # e is a 2-element dict giving the magnitude of the pump field in s/p coordinates;
    # its assumed that phi=0, such that an s-pol pump field is in the y direction
    # note that this assumption only works because SHG from GaN is azimuthally invariant
    # Returns the magnitude of the [pol]-pol SHG field in transmission geometry 

    n = 2  # Harmonic number

    kappa = omega/c * np.sin(theta)
    K = n * kappa
    # The following four are arrays, indexed by j (layer number)
    W = np.sqrt((n*omega/c)**2 * epsilon_Omega - K**2)
    w = np.sqrt((omega/c)**2 * epsilon_omega - kappa**2)
    ts = np.array([2*w[0]/(w[0]+w[1]), 2*w[1]/(w[1]+w[2])])
    rs = np.array([(w[0]-w[1])/(w[0]+w[1]), (w[1]-w[2])/(w[1]+w[2])])
    tp = np.array([2*np.sqrt(epsilon_omega[0]*epsilon_omega[1])*w[0] / (epsilon_omega[1]*w[0] + epsilon_omega[0]*w[1]),
                   2*np.sqrt(epsilon_omega[1]*epsilon_omega[2])*w[1] / (epsilon_omega[2]*w[1] + epsilon_omega[1]*w[2])])
    rp = np.array([(epsilon_omega[1]*w[0] - epsilon_omega[0]*w[1]) / (epsilon_omega[1]*w[0] + epsilon_omega[0]*w[1]),
                   (epsilon_omega[2]*w[1] - epsilon_omega[1]*w[2]) / (epsilon_omega[2]*w[1] + epsilon_omega[1]*w[2])])

    def mlayer_s(j):
        return np.array([[np.exp(1j*w[j]*d), 0], [0, np.exp(-1j*w[j]*d)]])

    def minterface_s(j):
        return 1/ts[j] * np.array([[1, rs[j]], [rs[j], 1]])
    
    def mlayer_p(j): # Identical to mlayer_s 
        return np.array([[np.exp(1j*w[j]*d), 0], [0, np.exp(-1j*w[j]*d)]])

    def minterface_p(j): # Different from minterface_s 
        return 1/tp[j] * np.array([[1, rp[j]], [rp[j], 1]])

    def e(m):
        # Returns a 2x1 vector containing the sum of m e+ fields and (n-m) e- fields in the GaN layer
        a_s = minterface_s(0) @ mlayer_s(1) @ minterface_s(1)
        a_p = minterface_p(0) @ mlayer_p(1) @ minterface_p(1) 
        e_plus = {}
        e_minus = {}
        e_plus['s'] = (minterface_s(1) @ np.array([0, e0['s']/a_s[1, 1]]))[0]
        e_minus['s'] = (minterface_s(1) @ np.array([0, e0['s']/a_s[1, 1]]))[1]
        e_plus['p'] = (minterface_p(1) @ np.array([0, e0['p']/a_p[1, 1]]))[0]
        e_minus['p'] = (minterface_p(1) @ np.array([0, e0['p']/a_p[1, 1]]))[1] 
        #print("es+ = " + str(e_plus['s']) + "\nep+ = " + str(e_plus['p']) + "\nes- = " + str(e_minus['s']) + "\nep- = " + str(e_minus['p']))
        return {'s' : m*e_plus['s'] + (n-m)*e_minus['s'], 'p' : m*e_plus['p'] + (n-m)*e_minus['p']}
    
    def q(m, j):
        return (2*m-n)*w[j]
    
    def Ps(m, j): # Nonlinear polarization equation, with chi2 = 1
        if j == 1: 
            ey = -e(m)['s'] 
            ez = kappa**2/(kappa**2 + w[j]**2)*e(m)['p'] 
            #print("Ps = " + str(2*ey*ez))
            return 2*ey*ez 
        else: return 0 # assumes SHG only happens in the GaN layer 
    
    def Pkappa(m, j): # Nonlinear polarization equation, with chi2 = 1
        if j == 1: 
            ex = -w[j]**2/(kappa**2 + w[j]**2)*e(m)['p'] 
            ez = kappa**2/(kappa**2 + w[j]**2)*e(m)['p'] 
            #print("Pkappa = " +str(2*ex*ez))
            return -2*ex*ez  # Equivalent to Px (no minus sign because kappa doesn't flip upon transmission)
        else: return 0 # assumes SHG only happens in the GaN layer 
    
    def Pz(m, j): # Nonlinear polarization equation, with chi2 = 1
        if j == 1: 
            ex = -w[j]**2/(kappa**2 + w[j]**2)*e(m)['p'] 
            ey = -e(m)['s'] 
            ez = kappa**2/(kappa**2 + w[j]**2)*e(m)['p'] 
            #print("Pz = " + str((ex**2+ey**2) - 2*ez**2))
            return (ex**2+ey**2) - 2*ez**2 
        else: return 0 # assumes SHG only happens in the GaN layer 
        
    def As(m, j):
        Cj = -4*np.pi / (epsilon_Omega[j] * (W[j]**2 - q(m,j)**2))
        #print("As = " + str(Cj*epsilon_Omega[j]*(n*omega/c)**2 * Ps(m, j)))
        return Cj*epsilon_Omega[j]*(n*omega/c)**2 * Ps(m, j)

    def Az(m, j):
        Cj = -4*np.pi / (epsilon_Omega[j] * (W[j]**2 - q(m,j)**2))
        #print("Az = " + str(Cj*((epsilon_Omega[j]*(n*omega/c)**2 - q(m,j)**2) * Pz(m,j) - q(m,j)*n*kappa*Pkappa(m,j) )))
        return Cj*((epsilon_Omega[j]*(n*omega/c)**2 - q(m,j)**2) * Pz(m,j) - q(m,j)*n*kappa*Pkappa(m,j) )
    
    def Akappa(m, j):
        Cj = -4*np.pi / (epsilon_Omega[j] * (W[j]**2 - q(m,j)**2)) 
        #print("Akappa = " + str(Cj*(W[j]**2 * Pkappa(m,j) - q(m,j)*n*kappa*Pz(m,j))))
        return Cj*(W[j]**2 * Pkappa(m,j) - q(m,j)*n*kappa*Pz(m,j))
    
    def Ss(sign, j):
        total = 0
        for m in np.arange(n+1):
            # Ignore surface contributions; we assume they're << bulk contributions for GaN
            # because the film thickness is >> lambda/4pi (Boyd pg 120)
            total += As(m,j+1)*(1+sign*q(m,j+1)/W[j]) - As(m,j)*(1+sign*q(m,j)/W[j]) 
        #print("Ss = " + str(1/2*total))
        return 1/2 * total 
    
    def Sp(sign, j):
        total = 0
        for m in np.arange(n+1):
            # Ignore surface contributions; we assume they're << bulk contributions for GaN
            # because the film thickness is >> lambda/4pi (Boyd pg 120)
            total += n*kappa*(Az(m,j+1)-Az(m,j)) - Akappa(m,j+1)*(q(m,j+1)+sign*epsilon_Omega[j]*(n*omega/c)**2/W[j]) + Akappa(m,j)*(q(m,j)+sign*epsilon_Omega[j]*(n*omega/c)**2/W[j]) 
        #print("Sp = " + str(1/(2*np.sqrt(epsilon_Omega[j])*(n*omega/c)**2) * total ))
        return 1/(2*np.sqrt(epsilon_Omega[j])*(n*omega/c)) * total 

    def Mslayer(j):
        return np.array([[np.exp(1j*W[j]*d), 0, 0], [0, np.exp(-1j*W[j]*d), 0], [0, 0, 1]])

    def Mplayer(j):
        return np.array([[np.exp(1j*W[j]*d), 0, 0], [0, np.exp(-1j*W[j]*d), 0], [0, 0, 1]])

    def Msinterface(j):
        return np.array([[1/ts[j], rs[j]/ts[j], Ss(+1, j)], [rs[j]/ts[j], 1/ts[j], Ss(-1, j)], [0, 0, 1]])

    def Mpinterface(j):
        return np.array([[1/tp[j], rp[j]/tp[j], Sp(+1, j)], [rp[j]/tp[j], 1/tp[j], Sp(-1, j)], [0, 0, 1]])
    
    # Calculate ER from each depth in the GaN
    # Sum the resultant SHG fields and return the result 
    E_total = 0
    ZI = 1000 # This takes a while, but gives more accurate results 
    #ZI = 200 #for fast and dirty computations
    for zi in np.linspace(0,d, ZI):
        z = zi 
        # Calculate matrix M
        # The @ operator is shorthand for np.matmul
        if pol == 's':
            M = Msinterface(0,z) @ Mslayer(1) @ Msinterface(1,z)
        elif pol == 'p':
            M = Mpinterface(0,z) @ Mplayer(1) @ Mpinterface(1,z)
        #print(M) 
        E_total += -M[1,2]/M[1,1] 
        'You will want to double-check the above line against Kolthammer before trusting this code'
    
    return E_total 

# =============================================================================
#     # Calculate matrix M
#     # The @ operator is shorthand for np.matmul
#     if pol == 's':
#         M = Msinterface(0) @ Mslayer(1) @ Msinterface(1)
#     elif pol == 'p':
#         M = Mpinterface(0) @ Mplayer(1) @ Mpinterface(1)
#     #print(M) 
#     # Calculate ET
#     return -M[1,2]/M[1,1] 
# =============================================================================


# Plotting and testing 
angle_range = np.linspace(0, 4*np.pi/2, 100)
omega = 2*np.pi * c/1.080e-6
theta = np.arcsin(0.15/1.7711)
d = 1.80e-6
SHGs = np.array([])
SHGp = np.array([]) 
SHG50 = np.array([]) 

#testing
#a = 0 # Pump pol 
b = 0 # analyzer pol 
theta = np.arcsin(0.15/1.7711) 
SHGs = np.append(SHGs, np.abs(np.cos(b)*ER(omega,theta,d,{'s':1,'p':0},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':1,'p':0},'s'))**2)
SHGp = np.append(SHGp, np.abs(np.cos(b)*ER(omega,theta,d,{'s':0,'p':1},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':0,'p':1},'s'))**2)
  
print("s-pol SHG = " + str(SHGs))
print("p-pol SHG = " + str(SHGp))

# =============================================================================
# for a in angle_range:
#     power_dependence = 1 # (1/2 + np.sin(a)/2) 
#     SHGs = np.append(SHGs, np.abs(ER(omega, theta, d, {'s' : power_dependence*np.cos(a), 'p' : power_dependence*np.sin(a)},'s'))**2)
#     SHGp = np.append(SHGp, np.abs(ER(omega, theta, d, {'s' : power_dependence*np.cos(a), 'p' : power_dependence*np.sin(a)},'p'))**2)
# plt.plot(angle_range, SHGs, 'blue', label='s-pol analyzer ')
# plt.plot(angle_range, SHGp, 'orange', label='p-pol analyzer')
# plt.legend() 
# plt.title('Pump-polarization dependence of SHG \n from 1.8-um GaN thin film. \n λ=1080 nm, $k_x$=0.5')
# plt.xlabel('Pump polarization \n (0 = s-pol, $\pi$/2 = p-pol)')
# plt.show() 
# =============================================================================

# =============================================================================
# input_mirror_p = 0.98299 # ThorLabs BB1-E03 
# input_mirror_s = 0.99579 # ThorLabs DMSP900 
# dichroic_s = 0.99381 
# dichroic_p = 0.98683
# input_field_s = np.sqrt(1 * input_mirror_s * input_mirror_p * input_mirror_s * input_mirror_s * dichroic_p)
# input_field_p = np.sqrt(1 * input_mirror_p * input_mirror_s * input_mirror_p * input_mirror_p * dichroic_s)
# =============================================================================
# =============================================================================
# input_field_s = 1
# input_field_p = 1
# for b in angle_range:
#     # b is analyzer angle, going from ppol tp spol 
#     SHGs = np.append(SHGs, np.abs(np.cos(b)*ER(omega,theta,d,{'s':input_field_s,'p':0},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':input_field_s,'p':0},'s'))**2)
#     SHGp = np.append(SHGp, np.abs(np.cos(b)*ER(omega,theta,d,{'s':0,'p':input_field_p},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':0,'p':input_field_p},'s'))**2)
#     SHG50 = np.append(SHG50, np.abs(np.cos(b)*ER(omega,theta,d,{'s':input_field_s/np.sqrt(2),'p':input_field_p/np.sqrt(2)},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':input_field_s/np.sqrt(2),'p':input_field_p/np.sqrt(2)},'s'))**2)
# 
# plt.plot(np.rad2deg(angle_range), SHGs, label='s-pump')
# plt.plot(np.rad2deg(angle_range), SHGp, label='p-pump')
# #plt.plot(np.rad2deg(angle_range), SHG50, label='50/50-pump')
# plt.legend() 
# plt.title('Polarization analysis of SHG from thin film \n Pump @ 1080 nm, $k_\parallel / k_0 = 0.15$')
# plt.xlabel('Analyer \n (p-pol = 0, s-pol = $\pi$/2)')
# plt.show() 
# =============================================================================
# =============================================================================
# th_range = np.linspace(0, np.arcsin(1.3/1.7711), 100)
# for th in th_range:
#     # th is incident angle, measured from surface normal 
#     SHGs = np.append(SHGs, np.abs(ER(omega,th,d,{'s':1,'p':0},'p'))**2)
#     SHGp = np.append(SHGp, np.abs(ER(omega,th,d,{'s':0,'p':1},'p'))**2) 
# 
# plt.plot(np.sin(th_range)*1.7711, SHGs, label='s-pump')
# plt.plot(np.sin(th_range)*1.7711, SHGp, label='p-pump') 
# plt.legend()
# 'Θ'
# plt.title('$k_\parallel$-dependence of p-pol SHG')
# plt.xlabel('$k_\parallel / k_0$')
# plt.show() 
# =============================================================================

# =============================================================================
# # Testing what ZI is required for convergence 
# ratio = np.array([]) 
# ZI_range = np.arange(3,200,10)
# for j in [500]: #[146]: 
#     # ZI = 100 yields 6.42
#     # ZI = 146 yields a ratio of 5.496 
#     # ZI=1000 yields a ratio of 4.325, which is ~halfway between the lowest (0.1) and highest (8.5) possible ratios, so we'll call that convergence 
#     # ZI=2000 yields a ratio of 4.252 
#     ZI = j
#     b=0
#     SHGs = np.abs(np.cos(b)*ER(omega,theta,d,{'s':1,'p':0},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':1,'p':0},'s'))**2
#     SHGp = np.abs(np.cos(b)*ER(omega,theta,d,{'s':0,'p':1},'p'))**2 + np.abs(np.sin(b)*ER(omega,theta,d,{'s':0,'p':1},'s'))**2
#     ratio = np.append(ratio, SHGp/SHGs)
#     print(ratio) 
# 
# plt.plot(ZI_range, ratio) 
# plt.show() 
# =============================================================================
