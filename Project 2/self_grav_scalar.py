import numpy as np
import pylab as pl
from elliptics_solver import *
import sys

#set parameters for simulation
N = 256
delta_r = 1./N
delta_t = 0.001
courant = delta_t / delta_r
timesteps = 50
epsilon = 0.

correction_weight = 1.
GEOM_COUPLING     = True
PSI_EVOL          = False

#define grid
R     = 50. 
amp   = 0.02
r_0   = 20.
delta = 5.

r_grid = np.linspace(delta_r, R, N)


#initialize arrays
phi = np.zeros((timesteps, N))
xi  = np.zeros((timesteps, N)) 
Pi  = np.zeros((timesteps, N))

alpha =  np.ones((timesteps, N))
beta  = np.zeros((timesteps, N))
psi   =  np.ones((timesteps, N))

#define initial data for scalar field
phi[0, :] = amp * np.exp(-(r_grid-r_0)**2./delta**2.)

#xi[0, :] = -2.*(r_grid-r_0)/delta**2. * amp*np.exp(-(r_grid-r_0)**2./delta**2.)

#Note: need to set xi as the numerical derivative of phi;
#using an analytical derivative gives incorrect result
for i in range(N):
	if(i == 0):
		xi[0, i] = (-phi[0, i+2] + 4.*phi[0, i+1] - 3.*phi[0, i])/(2.*delta_r)
	elif(i == N-1):
		xi[0, i] = (phi[0, i-2] - 4.*phi[0, i-1] + 3.*phi[0, i])/(2.*delta_r)
	else:
		xi[0, i] = (phi[0, i+1] - phi[0, i-1])/(2.*delta_r)


#need to set up "approximately" ingoing initial data; do so using the same prescription
#as we used in Project 1, which is identical except except a->\psi^2 
#we are asked to define this condition only in terms of phi, xi, r; using approximation in example code
Pi[0, :] = xi[0, :] #ingoing initial data
#don't do anything -> Pi = 0 -> half inward half outward


if(GEOM_COUPLING == True):
	#need to solve elliptics here for initial timestep
	#first set initial values of f_n
	f_n = np.zeros(3*N)
	f_n[0:N]     = psi[0, :] 
	f_n[N:2*N]   = beta[0, :]
	f_n[2*N:3*N] = alpha[0, :]
	f_n = solve_elliptics_first_ts(f_n, xi[0, :], Pi[0, :], r_grid, correction_weight=correction_weight)
	#now set psi, beta, alpha with solution to elliptics
#	psi[0, :]   = f_n[0:N]
        psi[0, :]   = np.abs(f_n[0:N]) #TODO: testing this; REMOVE LATER
        beta[0, :]  = f_n[N:2*N]
	alpha[0, :] = f_n[2*N:3*N]


if(PSI_EVOL == True):
    A = np.zeros((3*N, 3*N))
    B = np.zeros((3*N, 3*N))
else:
    A = np.zeros((2*N, 2*N))
    B = np.zeros((2*N, 2*N))

#populate the matrix at timestep n
def populate_matrices(n):

    if(PSI_EVOL == False):
        #define matrix A
        for i in range(N):
            if(i == 0):
        	A[i, :] = [1 if j==i else 0 for j in range(2*N)]
            elif(i == N-1): #xi BC
        	A[i, :] = [1. + delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                          else delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                          else delta_t/2. * (1./(2.*delta_r)) if j==i-2
        		  else 0 for j in range(2*N)]
            else:
        	A[i, :] = [1.   -delta_t/2.* (1.*(beta[n+1][i+1]-beta[n+1][i-1])/(2.*delta_r)) if j==i
        		   else -delta_t/2.* ( beta[n+1][i]/(2.*delta_r) ) if j==i+1
        		   else -delta_t/2.* (-beta[n+1][i]/(2.*delta_r) ) if j==i-1
        		   else -delta_t/2.* ( 1./psi[n+1][i%N]**2. * (alpha[n+1][i%N+1]-alpha[n+1][i%N-1])/(2.*delta_r) 
        				      -2.*alpha[n+1][i%N]/psi[n+1][i%N]**3.*(psi[n+1][i%N+1]-psi[n+1][i%N-1])/(2.*delta_r) ) if j==N+i
        		   else -delta_t/2.* (alpha[n+1][i%N]/psi[n+1][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
        		   else -delta_t/2.*(-alpha[n+1][i%N]/psi[n+1][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
        		   else 0 for j in range(2*N)]
        
        
        for i in range(N, 2*N):
            if(i == N):
        	A[i, :] = [-3. if j==i
        		   else 4. if j==i+1
        		   else -1. if j==i+2
                           else 0 for j in range(2*N)]
            elif(i == 2*N-1): #Pi BC
                A[i, :] = [1. + delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                           else delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else delta_t/2. * (1./(2.*delta_r)) if j==i-2
        		   else 0 for j in range(2*N)]
            else:
        	A[i, :] = [1. - delta_t/2. * (2.*beta[n+1][i%N]/(3.*r_grid[i%N]) 
                                             + 1./3.*(beta[n+1][i%N+1]-beta[n+1][i%N-1])/(2.*delta_r)
        				     + 4.*beta[n+1][i%N]/(2.*psi[n+1][i%N])*(psi[n+1][i%N+1]-psi[n+1][i%N-1])/(2.*delta_r)) if j==i
        		   else -delta_t/2. * (beta[n+1][i%N]/(2.*delta_r)) if j==i+1
        		   else -delta_t/2. * (-beta[n+1][i%N]/(2.*delta_r)) if j==i-1
        		   else -delta_t/2. * (2.*alpha[n+1][i%N]/(r_grid[i%N]*psi[n+1][i%N]**2.)
        				       + 2.*alpha[n+1][i%N]*(psi[n+1][i%N+1]-psi[n+1][i%N-1])/(2.*delta_r*psi[n+1][i%N]**3.)
        				       + 1./psi[n+1][i%N]**2.*(alpha[n+1][i%N+1]-alpha[n+1][i%N-1])/(2.*delta_r)) if j==i-N
        		   else -delta_t/2. * (alpha[n+1][i%N]/psi[n+1][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
        		   else -delta_t/2. *(-alpha[n+1][i%N]/psi[n+1][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
        		   else 0 for j in range(2*N)]
        
        
        #define matrix B, now fully to second order accuracy
        for i in range(N):
            if(i == 0):
        	B[i, :] = [0 for j in range(2*N)]
            elif(i == 1): #ADDED KO_ODD_FW
        	B[i, :] = [1. + delta_t/2. * (1.*(beta[n][i+1]-beta[n][i-1])/(2.*delta_r)) - epsilon/(16.*1)*(6.-1.) if j==i
                           else delta_t/2.* ( beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2.* (-beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i-1
                           else - epsilon/(16.*1)*1 if j==i+2
                           else delta_t/2.* ( 1./psi[n][i%N]**2. * (alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n][i%N]/psi[n][i%N]**3.*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r) ) if j==N+i
                           else delta_t/2.* (alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else delta_t/2.*(-alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(2*N)]
            elif(i == N-2):#internal xi eqn without KO diss.
        	B[i, :] = [1. + delta_t/2. * (1.*(beta[n][i+1]-beta[n][i-1])/(2.*delta_r)) if j==i
                           else delta_t/2.* ( beta[n][i]/(2.*delta_r) )  if j==i+1
                           else delta_t/2.* (-beta[n][i]/(2.*delta_r) )  if j==i-1
                           else delta_t/2.* ( 1./psi[n][i%N]**2. * (alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n][i%N]/psi[n][i%N]**3.*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r) ) if j==N+i
                           else delta_t/2.* (alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else delta_t/2.*(-alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(2*N)]
            elif(i == N-1): #xi BC
                B[i, :] = [1.   -delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                           else -delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (1./(2.*delta_r)) if j==i-2
        		   else 0 for j in range(2*N)]
            else: #ADDED DISS_KO to this
        	B[i, :] = [1. + delta_t/2. * (1.*(beta[n][i+1]-beta[n][i-1])/(2.*delta_r)) - epsilon/(16.*1)*6. if j==i
                           else delta_t/2.* ( beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2.* (-beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i-1
        		   else - epsilon/(16.*1)*1. if j==i-2
        		   else - epsilon/(16.*1)*1 if j==i+2
                           else delta_t/2.* ( 1./psi[n][i%N]**2. * (alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n][i%N]/psi[n][i%N]**3.*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r) ) if j==N+i
                           else delta_t/2.* (alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else delta_t/2.*(-alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(2*N)]
        
        for i in range(N, 2*N):
            if(i == N):
        	B[i, :] = [3. if j==i
        		   else -4. if j==i+1
        		   else 1. if j==i+2
        		   else 0 for j in range(2*N)]
            elif(i == N+1): #ADDED KO_EVEN_FWD TO THIS
        	B[i, :] = [1. + delta_t/2. * (2.*beta[n][i%N]/(3.*r_grid[i%N])
                                             + 1./3.*(beta[n][i%N+1]-beta[n][i%N-1])/(2.*delta_r)
        				     + 4.*beta[n][i%N]/(2.*psi[n][i%N])*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r)) - epsilon/(16.*1.)*(6.+1.) if j==i
                           else delta_t/2. * (beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2. * (-beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i-1
                           else - epsilon/(16.*1.)*1. if j==i+2
                           else delta_t/2. * (2.*alpha[n][i%N]/(r_grid[i%N]*psi[n][i%N]**2.)
                                               + 2.*alpha[n][i%N]*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r*psi[n][i%N]**3.)
                                               + 1./psi[n][i%N]**2.*(alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)) if j==i-N
                           else delta_t/2. * (alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else delta_t/2. *(-alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(2*N)]
            elif(i == 2*N-2): #internal Pi equation without KO diss.
        	B[i, :] = [1. + delta_t/2. * (2.*beta[n][i%N]/(3.*r_grid[i%N])
                                             + 1./3.*(beta[n][i%N+1]-beta[n][i%N-1])/(2.*delta_r)
        				     + 4.*beta[n][i%N]/(2.*psi[n][i%N])*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r)) if j==i
                           else delta_t/2. * (beta[n][i%N]/(2.*delta_r))  if j==i+1
                           else delta_t/2. * (-beta[n][i%N]/(2.*delta_r)) if j==i-1
                           else delta_t/2. * (2.*alpha[n][i%N]/(r_grid[i%N]*psi[n][i%N]**2.)
                                               + 2.*alpha[n][i%N]*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r*psi[n][i%N]**3.)
                                               + 1./psi[n][i%N]**2.*(alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)) if j==i-N
                           else delta_t/2. * (alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else delta_t/2. *(-alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(2*N)]
            elif(i == 2*N-1): #Pi BC
                B[i, :] = [1.   -delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                           else -delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (1./(2.*delta_r)) if j==i-2
        		   else 0 for j in range(2*N)]
        
            else: #ADDED DISS_KO TO THIS
        	B[i, :] = [1. + delta_t/2. * (2.*beta[n][i%N]/(3.*r_grid[i%N]) 
                                             + 1./3.*(beta[n][i%N+1]-beta[n][i%N-1])/(2.*delta_r)
        				     + 4.*beta[n][i%N]/(2.*psi[n][i%N])*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r)) - epsilon/(16.*1)*6. if j==i
                           else delta_t/2. * (beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2. * (-beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i-1
        		   else - epsilon/(16.*1)*1. if j==i-2
        		   else - epsilon/(16.*1)*1. if j==i+2
                           else delta_t/2. * (2.*alpha[n][i%N]/(r_grid[i%N]*psi[n][i%N]**2.)
                                               + 2.*alpha[n][i%N]*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r*psi[n][i%N]**3.)
        				       + 1./psi[n][i%N]**2.*(alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)) if j==i-N
                           else delta_t/2. * (alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else delta_t/2. *(-alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(2*N)]

    elif(PSI_EVOL == True):
        #define matrix A
        for i in range(N):
            if(i == 0):
                A[i, :] = [1 if j==i else 0 for j in range(3*N)]
            elif(i == N-1): #xi BC
                A[i, :] = [1. + delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                          else delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                          else delta_t/2. * (1./(2.*delta_r)) if j==i-2
                          else 0 for j in range(3*N)]
            else:
                A[i, :] = [1.   -delta_t/2.* (1.*(beta[n+1][i+1]-beta[n+1][i-1])/(2.*delta_r)) if j==i
                           else -delta_t/2.* ( beta[n+1][i]/(2.*delta_r) ) if j==i+1
                           else -delta_t/2.* (-beta[n+1][i]/(2.*delta_r) ) if j==i-1
                           else -delta_t/2.* ( 1./psi[n+1][i%N]**2. * (alpha[n+1][i%N+1]-alpha[n+1][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n+1][i%N]/psi[n+1][i%N]**3.*(psi[n+1][i%N+1]-psi[n+1][i%N-1])/(2.*delta_r) ) if j==N+i
                           else -delta_t/2.* (alpha[n+1][i%N]/psi[n+1][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else -delta_t/2.*(-alpha[n+1][i%N]/psi[n+1][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(3*N)]


        for i in range(N, 2*N):
            if(i == N):
                A[i, :] = [-3. if j==i
                           else 4. if j==i+1
                           else -1. if j==i+2
                           else 0 for j in range(3*N)]
            elif(i == 2*N-1): #Pi BC
                A[i, :] = [1. + delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                           else delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else delta_t/2. * (1./(2.*delta_r)) if j==i-2
                           else 0 for j in range(3*N)]
            else:
                A[i, :] = [1. - delta_t/2. * (2.*beta[n+1][i%N]/(3.*r_grid[i%N])
                                             + 1./3.*(beta[n+1][i%N+1]-beta[n+1][i%N-1])/(2.*delta_r)
                                             + 4.*beta[n+1][i%N]/(2.*psi[n+1][i%N])*(psi[n+1][i%N+1]-psi[n+1][i%N-1])/(2.*delta_r)) if j==i
                           else -delta_t/2. * (beta[n+1][i%N]/(2.*delta_r)) if j==i+1
                           else -delta_t/2. * (-beta[n+1][i%N]/(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (2.*alpha[n+1][i%N]/(r_grid[i%N]*psi[n+1][i%N]**2.)
                                               + 2.*alpha[n+1][i%N]*(psi[n+1][i%N+1]-psi[n+1][i%N-1])/(2.*delta_r*psi[n+1][i%N]**3.)
                                               + 1./psi[n+1][i%N]**2.*(alpha[n+1][i%N+1]-alpha[n+1][i%N-1])/(2.*delta_r)) if j==i-N
                           else -delta_t/2. * (alpha[n+1][i%N]/psi[n+1][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else -delta_t/2. *(-alpha[n+1][i%N]/psi[n+1][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(3*N)]

        #define matrix B, now fully to second order accuracy
        for i in range(N):
            if(i == 0):
                B[i, :] = [0 for j in range(3*N)]
            elif(i == 1): #ADDED KO_ODD_FW
                B[i, :] = [1. + delta_t/2. * (1.*(beta[n][i+1]-beta[n][i-1])/(2.*delta_r)) - epsilon/(16.*1)*(6.-1.) if j==i
                           else delta_t/2.* ( beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2.* (-beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i-1
                           else - epsilon/(16.*1)*1 if j==i+2
                           else delta_t/2.* ( 1./psi[n][i%N]**2. * (alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n][i%N]/psi[n][i%N]**3.*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r) ) if j==N+i
                           else delta_t/2.* (alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else delta_t/2.*(-alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(3*N)]
            elif(i == N-2):#internal xi eqn without KO diss.
                B[i, :] = [1. + delta_t/2. * (1.*(beta[n][i+1]-beta[n][i-1])/(2.*delta_r)) if j==i
                           else delta_t/2.* ( beta[n][i]/(2.*delta_r) )  if j==i+1
                           else delta_t/2.* (-beta[n][i]/(2.*delta_r) )  if j==i-1
                           else delta_t/2.* ( 1./psi[n][i%N]**2. * (alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n][i%N]/psi[n][i%N]**3.*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r) ) if j==N+i
                           else delta_t/2.* (alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else delta_t/2.*(-alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(3*N)]
            elif(i == N-1): #xi BC
                B[i, :] = [1.   -delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                           else -delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (1./(2.*delta_r)) if j==i-2
                           else 0 for j in range(3*N)]
            else: #ADDED DISS_KO to this
                B[i, :] = [1. + delta_t/2. * (1.*(beta[n][i+1]-beta[n][i-1])/(2.*delta_r)) - epsilon/(16.*1)*6. if j==i
                           else delta_t/2.* ( beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2.* (-beta[n][i]/(2.*delta_r) ) - epsilon/(16.*1)*(-4.) if j==i-1
                           else - epsilon/(16.*1)*1. if j==i-2
                           else - epsilon/(16.*1)*1 if j==i+2
                           else delta_t/2.* ( 1./psi[n][i%N]**2. * (alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)
                                              -2.*alpha[n][i%N]/psi[n][i%N]**3.*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r) ) if j==N+i
                           else delta_t/2.* (alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i+1
                           else delta_t/2.*(-alpha[n][i%N]/psi[n][i%N]**2.*1./(2.*delta_r)) if j==N+i-1
                           else 0 for j in range(3*N)]

        for i in range(N, 2*N):
            if(i == N):
                B[i, :] = [3. if j==i
                           else -4. if j==i+1
                           else 1. if j==i+2
                           else 0 for j in range(3*N)]
            elif(i == N+1): #ADDED KO_EVEN_FWD TO THIS
                B[i, :] = [1. + delta_t/2. * (2.*beta[n][i%N]/(3.*r_grid[i%N])
                                             + 1./3.*(beta[n][i%N+1]-beta[n][i%N-1])/(2.*delta_r)
                                             + 4.*beta[n][i%N]/(2.*psi[n][i%N])*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r)) - epsilon/(16.*1.)*(6.+1.) if j==i
                           else delta_t/2. * (beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2. * (-beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i-1
                           else - epsilon/(16.*1.)*1. if j==i+2
                           else delta_t/2. * (2.*alpha[n][i%N]/(r_grid[i%N]*psi[n][i%N]**2.)
                                               + 2.*alpha[n][i%N]*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r*psi[n][i%N]**3.)
                                               + 1./psi[n][i%N]**2.*(alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)) if j==i-N
                           else delta_t/2. * (alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else delta_t/2. *(-alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(3*N)]
            elif(i == 2*N-2): #internal Pi equation without KO diss.
                B[i, :] = [1. + delta_t/2. * (2.*beta[n][i%N]/(3.*r_grid[i%N])
                                             + 1./3.*(beta[n][i%N+1]-beta[n][i%N-1])/(2.*delta_r)
                                             + 4.*beta[n][i%N]/(2.*psi[n][i%N])*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r)) if j==i
                           else delta_t/2. * (beta[n][i%N]/(2.*delta_r))  if j==i+1
                           else delta_t/2. * (-beta[n][i%N]/(2.*delta_r)) if j==i-1
                           else delta_t/2. * (2.*alpha[n][i%N]/(r_grid[i%N]*psi[n][i%N]**2.)
                                               + 2.*alpha[n][i%N]*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r*psi[n][i%N]**3.)
                                               + 1./psi[n][i%N]**2.*(alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)) if j==i-N
                           else delta_t/2. * (alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else delta_t/2. *(-alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(3*N)]
            elif(i == 2*N-1): #Pi BC
                B[i, :] = [1.   -delta_t/2. * (3./(2.*delta_r) + 1./r_grid[i%N]) if j==i
                           else -delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (1./(2.*delta_r)) if j==i-2
                           else 0 for j in range(3*N)]

            else: #ADDED DISS_KO TO THIS
                B[i, :] = [1. + delta_t/2. * (2.*beta[n][i%N]/(3.*r_grid[i%N])
                                             + 1./3.*(beta[n][i%N+1]-beta[n][i%N-1])/(2.*delta_r)
                                             + 4.*beta[n][i%N]/(2.*psi[n][i%N])*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r)) - epsilon/(16.*1)*6. if j==i
                           else delta_t/2. * (beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i+1
                           else delta_t/2. * (-beta[n][i%N]/(2.*delta_r)) - epsilon/(16.*1)*(-4.) if j==i-1
                           else - epsilon/(16.*1)*1. if j==i-2
                           else - epsilon/(16.*1)*1. if j==i+2
                           else delta_t/2. * (2.*alpha[n][i%N]/(r_grid[i%N]*psi[n][i%N]**2.)
                                               + 2.*alpha[n][i%N]*(psi[n][i%N+1]-psi[n][i%N-1])/(2.*delta_r*psi[n][i%N]**3.)
                                               + 1./psi[n][i%N]**2.*(alpha[n][i%N+1]-alpha[n][i%N-1])/(2.*delta_r)) if j==i-N
                           else delta_t/2. * (alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N+1
                           else delta_t/2. *(-alpha[n][i%N]/psi[n][i%N]**2. * 1./(2.*delta_r)) if j==i-N-1
                           else 0 for j in range(3*N)]



 #       if(PSI_EVOL == True):
        for i in range(2*N, 3*N):
                if(i == 2*N): #psi r=0 BC
                        A[i, :] = [-3. if j==i
                                         else 4. if j==i+1
                                         else -1. if j==i+2
                                         else 0 for j in range(3*N)]
                elif(i == 3*N-1): #psi r=R BC
                        A[i, :] = [3./(2.*delta_r) + 1./r_grid[i%N] if j==i
                                         else -4./(2.*delta_r) if j==i-1
        				 else 1./(2.*delta_r) if j==i-2
        				 else 0 for j in range(3*N)]
                else: #psi internal evolution equation
                        A[i, :] = [1. - delta_t/2. * ( beta[n+1][i%N]/(3.*r_grid[i%N]) 
                                                            + (beta[n+1][i%N+1]-beta[n+1][i%N-1])/(6. * 2. * delta_r) ) if j==i
                                         else -delta_t/2. * beta[n+1][i%N]/(2.*delta_r) if j==i+1
                                         else delta_t/2. * beta[n+1][i%N]/(2.*delta_r) if j==i-1
                                         else 0 for j in range(3*N)]
        for i in range(2*N, 3*N):
        	if(i == 2*N): #psi r=0 BC
        		B[i, :] = [3. if j==i
        			   else -4. if j==i+1
        			   else 1. if j==i+2
        			   else 0 for j in range(3*N)]
        	elif(i == 2*N+1): #psi internal eq. with DISS_EVEN_KO
        		B[i, :] = [1. + delta_t/2. * ( beta[n][i%N]/(3.*r_grid[i%N])
                                                            + (beta[n][i%N+1]-beta[n][i%N-1])/(6. * 2. * delta_r) ) if j==i
                                   else delta_t/2. * beta[n][i%N]/(2.*delta_r) if j==i+1
                                   else -delta_t/2. * beta[n][i%N]/(2.*delta_r) if j==i-1
                                   else 0 for j in range(3*N)]
        	elif(i == 3*N-2): #psi internal eq. with no KO dissipation
        		B[i, :] = [1. + delta_t/2. * ( beta[n][i%N]/(3.*r_grid[i%N])
                                                      + (beta[n][i%N+1]-beta[n][i%N-1])/(6. * 2. * delta_r) ) if j==i
                                   else delta_t/2. * beta[n][i%N]/(2.*delta_r) if j==i+1
                                   else -delta_t/2. * beta[n][i%N]/(2.*delta_r) if j==i-1
                                   else 0 for j in range(3*N)]
        	elif(i == 3*N-1): #psi r=R BC
#        		B[i, :] = [0 for j in range(3*N)]
                        B[i, :] = [-3./(2.*delta_r) - 1./r_grid[i%N] if j==i
                                         else 4./(2.*delta_r) if j==i-1
                                         else -1./(2.*delta_r) if j==i-2
                                         else 0 for j in range(3*N)]
        	else: #psi internal eq. with KO_DISS
        		B[i, :] = [1. + delta_t/2. * ( beta[n][i%N]/(3.*r_grid[i%N])
                                                      + (beta[n][i%N+1]-beta[n][i%N-1])/(6. * 2. * delta_r) ) if j==i
                                   else delta_t/2. * beta[n][i%N]/(2.*delta_r) if j==i+1
                                   else -delta_t/2. * beta[n][i%N]/(2.*delta_r) if j==i-1
                                   else 0 for j in range(3*N)]

	return 0



def update_u(timestep):

    if(PSI_EVOL == False):
        u = np.zeros(2*N)
    else:
        u = np.zeros(3*N)

    for i in range(2*N):
        if(i < N):
            u[i] = xi[timestep, i]
        else:
            u[i] = Pi[timestep, i-N]

    if(PSI_EVOL == True):
        for i in range(2*N, 3*N):
            u[i] = psi[timestep, i-2*N]


    return u

#update xi and Pi using the solved-for vector ans, which is u at timestep [timestep]
def update_r_s(ans, timestep):
    for i in range(2*N):
        if(i < N):
            xi[timestep, i] = ans[i]
        else:
            Pi[timestep, i-N] = ans[i]

    if(PSI_EVOL == True):
        for i in range(2*N, 3*N):
            psi[timestep, i-2*N] = ans[i]


    return 0

#n = 0
elliptic_res = np.zeros(3*N)
matter_res   = np.zeros(2*N)
xi_residual  = np.zeros(N)
Pi_residual  = np.zeros(N)

#initial guess for elliptics
psi[1, :]   = psi[0, :]
beta[1, :]  = beta[0, :]
alpha[1, :] = alpha[0, :]

#num_iter = 0
#max_num_iter = 8
#elliptics_tol = 1e-8
#
#normres = 10.

if(PSI_EVOL == True):
    #need to add constant term for psi r=R outer boundary
    c = np.zeros(3*N)
    c[3*N-1] = 2./r_grid[N-1]
else:
    c = np.zeros(2*N)

#TODO: modify this to solve elliptics to tolerance after each hyperbolic iteration
def solve_system(n):
	num_iter = 0
	max_num_iter = 40
	tol = 1e-8

	normres = 10. #TODO: make this the actual residual

	while(num_iter <= max_num_iter and normres > tol):
		num_iter += 1
	#for i in range(10):
		#we want to work through the first iteration of the coupled matter-elliptics solver
		#1.) elliptics, matter at n=0 are solved.
		#2.) we want to solve matter with a guess for elliptics at n=1
		populate_matrices(n)            #A has n+1, B has n
                u = update_u(n)                 #populate u with xi, Pi at n = 0
                bb = B.dot(u)                   #compute bb (all at n = 0)
		ans = np.linalg.solve(A, bb+c)  #solve for xi, Pi at n=1 with elliptics guess for n=1
		update_r_s(ans, n+1)            #update xi, Pi at n=1 with elliptics guess for n=1

#		if(PSI_EVOL == True):
#			bb = psi_RHS.dot(psi[n, :])
#			psi[n+1, :] = np.linalg.solve(psi_LHS, bb)

		if(GEOM_COUPLING == True):
			num_iter_elliptic = 0
			elliptic_res_norm = 10. #TODO: make this actual residual
			#do one Newton iteration to improve elliptics at future timestep n+1
			#TRYING ELLIPTICS TILL TOLERANCE
			while(num_iter_elliptic <= max_num_iter and elliptic_res_norm > tol):
				f_n, elliptic_res = Newton_iteration(xi, Pi, psi, beta, alpha, r_grid, n+1, delta_t, epsilon, correction_weight, PSI_EVOL)
				if(PSI_EVOL == False):
					psi[n+1, :]   = f_n[0:N] #THE WRONG PSI MAY BE IN RESIDUAL
#                                psi[n+1, :]   = f_n[0:N] 
                                beta[n+1, :]  = f_n[N:2*N]
				alpha[n+1, :] = f_n[2*N:3*N]
                                if(PSI_EVOL == False):
                                    elliptic_res_norm = np.amax(np.abs(elliptic_res))
                                else:
                                    elliptic_res_norm = np.amax(np.abs(elliptic_res[N:3*N]))

				num_iter_elliptic += 1
				print 'elliptic iteration:', num_iter_elliptic, 'elliptic res:', elliptic_res_norm
		else:
			elliptic_res = np.zeros(3*N)
	
		#confusion with n in below matter_residuals
		xi_residual, Pi_residual = matter_residuals(xi, Pi, psi, beta, alpha, r_grid, n, delta_t, epsilon)
		matter_res = np.append(xi_residual, Pi_residual)

                if(PSI_EVOL == False):
                    res = np.append(elliptic_res, matter_res)
                else:
                    res = np.append(elliptic_res[N:3*N], matter_res)
		normres = np.amax(np.abs(res))
#		print 'inf. norm of full residual:', np.amax(np.abs(elliptic_res)), np.amax(np.abs(matter_res)), normres
                print 'inf. norm of full residual:', normres

	return 0.

for n in range(timesteps-1):
        sys.stdout.flush() #flush output to SLURM .out file
	print '-----', n, '-----'
	solve_system(n)

#for n in range(1, timesteps):
#    populate_matrices(n-1) #TODO: not sure this works
#
#    u = update_u(n-1)
#
#    bb = B.dot(u)
#
#    ans = np.linalg.solve(A, bb)
#
#    update_r_s(ans, n)
#
#    print 'timestep n =', n
#
#    if(GEOM_COUPLING == True):
#	    #need to solve elliptics before populating CN matrices
#	    #first set initial values of f_n
#	    f_n = np.zeros(3*N)
#	    f_n[0:N]     = psi[n-1, :] #TODO: testing this
#	    f_n[N:2*N]   = beta[n-1, :]
#	    f_n[2*N:3*N] = alpha[n-1, :]
#	    f_n = solve_elliptics(f_n, xi[n, :], Pi[n, :], r_grid, correction_weight=correction_weight)
#	    #now set psi, beta, alpha with solution to elliptics
#	    psi[n, :]   = f_n[0:N]
#	    beta[n, :]  = f_n[N:2*N]
#	    alpha[n, :] = f_n[2*N:3*N]
#
#    for i in range(N):
#        #uses O(h^2) Crank-Nicolson time differencing
#    	phi[n, i] = ( phi[n-1, i] + 0.5 * delta_t * ((alpha[n][i]/psi[n][i]**2.*Pi[n, i] 
#                                                      + beta[n][i]*xi[n][i]) 
#                                                    + (alpha[n-1][i]/psi[n-1][i]**2.*Pi[n-1, i] 
#                                                       + beta[n-1][i]*xi[n-1, i])) )

for n in range(1, timesteps):
	for i in range(N):
        #uses O(h^2) Crank-Nicolson time differencing
		phi[n, i] = ( phi[n-1, i] + 0.5 * delta_t * ((alpha[n][i]/psi[n][i]**2.*Pi[n, i] 
		                                               + beta[n][i]*xi[n][i]) 
		                                             + (alpha[n-1][i]/psi[n-1][i]**2.*Pi[n-1, i] 
		                                                + beta[n-1][i]*xi[n-1, i])) )

print '-----computing mass aspect-----'
mass_aspect = np.zeros((timesteps, N)) 
#this computes the mass aspect function m(r,t)
for n in range(timesteps-1): #TODO: make this the same as sample code
	for j in range(N):
#		if(j == 0):
#			mass_aspect[n, j] = ( r_grid[j] * psi[n,j]**6./(18.*alpha[n,j]**2.)
#                                             *(r_grid[j]*(-3.*beta[n,j]+4.*beta[n,j+1]-beta[n,j+2])/(2.*delta_r)  
#                                               - beta[n,j] )**2.
#                                             - 2.*r_grid[j]**2.*(-3.*psi[n,j]+4.*psi[n,j+1]-psi[n,j+2])/(2.*delta_r)
#                                               *(psi[n,j] + r_grid[j]*(-3.*psi[n,j]+4.*psi[n,j+1]-psi[n,j+2])/(2.*delta_r) ) )
#		elif(j == N-1):
#			mass_aspect[n, j] = ( r_grid[j] * psi[n,j]**6./(18.*alpha[n,j]**2.)
#                                             *(r_grid[j]*(3.*beta[n,j]-4.*beta[n,j-1]+beta[n,j-2])/(2.*delta_r) 
#                                               - beta[n,j] )**2. 
#                                             - 2.*r_grid[j]**2.*(3.*psi[n,j]-4.*psi[n,j-1]+psi[n,j-2])/(2.*delta_r) 
#                                               *(psi[n,j] + r_grid[j]*(3.*psi[n,j]-4.*psi[n,j-1]+psi[n,j-2])/(2.*delta_r) ) )
		if(j != 0 and j != N-1):
			mass_aspect[n, j] = ( r_grid[j] * psi[n+1,j]**6./(18.*alpha[n+1,j]**2.)
					     *(r_grid[j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r) 
                                               - beta[n+1,j] )**2.
					     - 2.*r_grid[j]**2.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
					       *(psi[n+1,j] + r_grid[j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r) ) )
		elif(j == 0):
			mass_aspect[n, j] = mass_aspect[n, j+1]
		elif(j == N-1):
			mass_aspect[n, j] = mass_aspect[n, j-1]

print '-----saving datafiles-----'
np.savetxt('r_grid.txt', r_grid)
np.savetxt('phi.txt', phi)
np.savetxt('xi.txt', xi)
np.savetxt('Pi.txt', Pi)
np.savetxt('psi.txt', psi)
np.savetxt('beta.txt', beta)
np.savetxt('alpha.txt', alpha)
np.savetxt('mass_aspect.txt', mass_aspect)

print '-----computing residuals-----'
#phi_residual = np.zeros((timesteps, N))
#xi_residual  = np.zeros((timesteps, N))
#Pi_residual  = np.zeros((timesteps, N))

psi_residual    = np.zeros((timesteps, N))
psi_ev_residual = np.zeros((timesteps, N))
alpha_residual  = np.zeros((timesteps, N))
beta_residual   = np.zeros((timesteps, N))

for n in range(timesteps-1):
	f_n = np.zeros(3*N)
        f_n[0:N]     = psi[n, :]
        f_n[N:2*N]   = beta[n, :]
        f_n[2*N:3*N] = alpha[n, :]
        f_n = residual(f_n, xi[n, :], Pi[n, :], r_grid)
        #now set psi_residual, beta_residual, alpha_residual
        psi_residual[n, :]   = f_n[0:N]
        beta_residual[n, :]  = f_n[N:2*N]
        alpha_residual[n, :] = f_n[2*N:3*N]


#for n in range(timesteps-1):
#	for j in range(N):
#		if(j == 0):
#			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t 
#					       - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
#					       - 0.5*beta[n,j]*xi[n,j] 
#					       - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
#                                               - 0.5*beta[n+1,j]*xi[n+1,j]) 
#			xi_residual[n, j]  = 0.5*(xi[n, j] + xi[n+1,j])
#			Pi_residual[n, j]  = ( -Pi[n+1,j+2] + 4.*Pi[n+1,j+1] - 3.*Pi[n+1,j] 
#					       -Pi[n,j+2]   + 4.*Pi[n,j+1]   - 3.*Pi[n,j] )
#			psi_ev_residual[n, j] = ( -psi[n+1,j+2] + 4.*psi[n+1,j+1] - 3.*psi[n+1,j]
#                                               -psi[n,j+2]   + 4.*psi[n,j+1]   - 3.*psi[n,j] )
#		elif(j == 1):
#			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t
#                                               - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
#                                               - 0.5*beta[n,j]*xi[n,j]
#                                               - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
#                                               - 0.5*beta[n+1,j]*xi[n+1,j])
#                        xi_residual[n, j] = ( (xi[n+1,j]-xi[n,j])/delta_t
#                                            -0.5*(-2.*Pi[n,j]*alpha[n,j]/psi[n,j]**3. * (psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#                                                  + Pi[n,j]/psi[n,j]**2. * (alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
#                                                  + alpha[n,j]/psi[n,j]**2. * (Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
#                                                  + beta[n,j] * (xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
#                                                  + xi[n,j] * (beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#                                                  -2.*Pi[n+1,j]*alpha[n+1,j]/psi[n+1,j]**3. * (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                                  + Pi[n+1,j]/psi[n+1,j]**2. * (alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
#                                                  + alpha[n+1,j]/psi[n+1,j]**2. * (Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
#                                                  + beta[n+1,j] * (xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
#                                                  + xi[n+1,j] * (beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)    )
#                                                  +(epsilon/(16.*delta_t))*(-xi[n,j]-4*xi[n,j-1]+6*xi[n,j]-4*xi[n,j+1]+xi[n,j+2])
#                                                )
#                        Pi_residual[n, j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
#                                             -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
#					     -0.5*(4./2.)*beta[n,j]*Pi[n,j]/psi[n,j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#                                             -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#                                             -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
#                                             -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
#                                             -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
#                                             -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#                                             -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
#                                             -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
#					     -0.5*(4./2.)*beta[n+1,j]*Pi[n+1,j]/psi[n+1,j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
#                                             -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
#                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
#                                             +(epsilon/(16.*delta_t))*(Pi[n,j]-4*Pi[n,j-1]+6*Pi[n,j]-4*Pi[n,j+1]+Pi[n,j+2])
#                                            )
#                        psi_ev_residual[n,j] = ( (psi[n+1,j]-psi[n,j])/delta_t
#                                                -0.5*( beta[n,j]*(psi[n,j]/(3.*r_grid[j]) + (psi[n,j+1]-psi[n,j-1])/(2.*delta_r) )
#                                                      +psi[n,j]/6.*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#                                                      +beta[n+1,j]*(psi[n+1,j]/(3.*r_grid[j]) + (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r) )
#                                                      +psi[n+1,j]/6.*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
#                                                        )
#                                            )
#		elif(j == N-1):
#			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t  
#                                               - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
#                                               - 0.5*beta[n,j]*xi[n,j]
#                                               - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
#                                               - 0.5*beta[n+1,j]*xi[n+1,j])
#			xi_residual[n, j] = ( (xi[n+1,j]-xi[n,j])/delta_t 
#                                           + 0.5 * ( (3.*xi[n+1,j] - 4.*xi[n+1,j-1] + xi[n+1, j-2])/(2.*delta_r) 
#                                                    + xi[n+1,j]/r_grid[j]
#						    +(3.*xi[n,j] - 4.*xi[n,j-1] + xi[n, j-2])/(2.*delta_r)  
#						    + xi[n,j]/r_grid[j]) )
#			Pi_residual[n, j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
#                                           + 0.5 * ( (3.*Pi[n+1,j] - 4.*Pi[n+1,j-1] + Pi[n+1, j-2])/(2.*delta_r)
#                                                    + Pi[n+1,j]/r_grid[j]
#                                                    +(3.*Pi[n,j] - 4.*Pi[n,j-1] + Pi[n, j-2])/(2.*delta_r)
#                                                    + Pi[n,j]/r_grid[j]) )
#			psi_ev_residual[n,j] = ( (3.*psi[n,j]-4.*psi[n,j-1]+psi[n,j-2])/(2.*delta_r) + psi[n,j]/r_grid[j]
#					     +(3.*psi[n+1,j]-4.*psi[n+1,j-1]+psi[n+1,j-2])/(2.*delta_r) + psi[n+1,j]/r_grid[j] )
#		elif(j == N-2):
#			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t
#                                               - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
#                                               - 0.5*beta[n,j]*xi[n,j]
#                                               - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
#                                               - 0.5*beta[n+1,j]*xi[n+1,j])
#                        xi_residual[n, j] = ( (xi[n+1,j]-xi[n,j])/delta_t
#                                            -0.5*(-2.*Pi[n,j]*alpha[n,j]/psi[n,j]**3. * (psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#                                                  + Pi[n,j]/psi[n,j]**2. * (alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
#                                                  + alpha[n,j]/psi[n,j]**2. * (Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
#                                                  + beta[n,j] * (xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
#                                                  + xi[n,j] * (beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#                                                  -2.*Pi[n+1,j]*alpha[n+1,j]/psi[n+1,j]**3. * (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                                  + Pi[n+1,j]/psi[n+1,j]**2. * (alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
#                                                  + alpha[n+1,j]/psi[n+1,j]**2. * (Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
#                                                  + beta[n+1,j] * (xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
#                                                  + xi[n+1,j] * (beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)    )
#                                                )
#                        Pi_residual[n, j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
#                                             -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
#					     -0.5*(4./2.)*beta[n,j]*Pi[n,j]/psi[n,j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#                                             -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#                                             -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
#                                             -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
#                                             -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
#                                             -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#                                             -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
#                                             -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
#					     -0.5*(4./2.)*beta[n+1,j]*Pi[n+1,j]/psi[n+1,j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
#                                             -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
#                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
#                                            )
#                        psi_ev_residual[n,j] = ( (psi[n+1,j]-psi[n,j])/delta_t
#                                                -0.5*( beta[n,j]*(psi[n,j]/(3.*r_grid[j]) + (psi[n,j+1]-psi[n,j-1])/(2.*delta_r) )
#                                                      +psi[n,j]/6.*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#                                                      +beta[n+1,j]*(psi[n+1,j]/(3.*r_grid[j]) + (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r) )
#                                                      +psi[n+1,j]/6.*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
#                                                        )
#                                            ) 
#		else:
#			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t  
#                                               - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
#                                               - 0.5*beta[n,j]*xi[n,j]
#                                               - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
#                                               - 0.5*beta[n+1,j]*xi[n+1,j])
#			xi_residual[n, j] = ( (xi[n+1,j]-xi[n,j])/delta_t
#					    -0.5*(-2.*Pi[n,j]*alpha[n,j]/psi[n,j]**3. * (psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#						  + Pi[n,j]/psi[n,j]**2. * (alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
#						  + alpha[n,j]/psi[n,j]**2. * (Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
#						  + beta[n,j] * (xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
#						  + xi[n,j] * (beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#						  -2.*Pi[n+1,j]*alpha[n+1,j]/psi[n+1,j]**3. * (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)    
#                                                  + Pi[n+1,j]/psi[n+1,j]**2. * (alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)    
#                                                  + alpha[n+1,j]/psi[n+1,j]**2. * (Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)    
#                                                  + beta[n+1,j] * (xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)    
#                                                  + xi[n+1,j] * (beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)    )
#					    	  +(epsilon/(16.*delta_t))*(xi[n,j-2]-4*xi[n,j-1]+6*xi[n,j]-4*xi[n,j+1]+xi[n,j+2])
#						)
#			Pi_residual[n, j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
#					     -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
#					     -0.5*(4./2.)*beta[n,j]*Pi[n,j]/psi[n,j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#					     -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#					     -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
#					     -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
#					     -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
#					     -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
#					     -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
#					     -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
#					     -0.5*(4./2.)*beta[n+1,j]*Pi[n+1,j]/psi[n+1,j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
#                                             -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
#                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
#                                             -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
#					     +(epsilon/(16.*delta_t))*(Pi[n,j-2]-4*Pi[n,j-1]+6*Pi[n,j]-4*Pi[n,j+1]+Pi[n,j+2])
#					    )
#			psi_ev_residual[n,j] = ( (psi[n+1,j]-psi[n,j])/delta_t 
#						-0.5*( beta[n,j]*(psi[n,j]/(3.*r_grid[j]) + (psi[n,j+1]-psi[n,j-1])/(2.*delta_r) )
#						      +psi[n,j]/6.*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
#						      +beta[n+1,j]*(psi[n+1,j]/(3.*r_grid[j]) + (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r) )
#                                                      +psi[n+1,j]/6.*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)   
#							)
#					    )
#
#np.savetxt('phi_residual.txt', phi_residual)
#np.savetxt('xi_residual.txt', xi_residual)
#np.savetxt('Pi_residual.txt', Pi_residual) 
np.savetxt('psi_residual.txt', psi_residual)
np.savetxt('psi_ev_residual.txt', psi_ev_residual)
np.savetxt('beta_residual.txt', beta_residual)
np.savetxt('alpha_residual.txt', alpha_residual)

print '-----done-----'
