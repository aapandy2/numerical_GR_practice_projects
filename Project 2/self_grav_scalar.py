import numpy as np
from elliptics_solver import solve_elliptics, residual

#set parameters for simulation
N = 256
delta_r = 1./N
delta_t = 0.3/256.
courant = delta_t / delta_r
timesteps = 2048 
epsilon = 0.3

correction_weight = 1.
GEOM_COUPLING = True

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
	f_n = solve_elliptics(f_n, xi[0, :], Pi[0, :], r_grid, correction_weight=correction_weight)
	#now set psi, beta, alpha with solution to elliptics
	psi[0, :]   = f_n[0:N]
	beta[0, :]  = f_n[N:2*N]
	alpha[0, :] = f_n[2*N:3*N]


A = np.zeros((2*N, 2*N))
B = np.zeros((2*N, 2*N))

psi_LHS = np.zeros((N, N))
psi_RHS = np.zeros((N, N))
#populate the matrix at timestep n
def populate_matrices(n):
	#define matrix A
	for i in range(N):
	    if(i == 0):
		A[i, :] = [1 if j==i else 0 for j in range(2*N)]
	    elif(i == N-1): #xi BC
		A[i, :] = [1. + delta_t/2. * (3./(2.*delta_r) + 1./r_grid[j%N]) if j==i
			   else delta_t/2. * (-4./(2.*delta_r)) if j==i-1
			   else delta_t/2. * (1./(2.*delta_r)) if j==i-2
			   else 0 for j in range(2*N)]
	    else: #TODO: REMOVE ALL CONSTANTS FROM THIS AND ADD THEM TO OTHER SIDE; LHS (A) SHOULD BE CORRECT
	        A[i, :] = [     (-0.5*1./2.*beta[n+1][j%N+1]/delta_r + 0.5*1./2.*beta[n+1][j%N-1]/delta_r + 1./(delta_t)) if j==i 
			   else (0.5*1./2.*beta[n+1][j%N]/delta_r) if j==(i-1) 
			   else (-0.5*1./2.*beta[n+1][j%N]/delta_r) if j==(i+1) 
			   else (0.5*1./2.*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==(N+i-1)
			   else (-0.5*1./2.*alpha[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**2) + 0.5*1./2.*alpha[n+1][j%N-1]/(delta_r*psi[n+1][j%N]**2.) + 0.5*alpha[n+1][j%N]*psi[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**3.) - 0.5*alpha[n+1][j%N]*psi[n+1][j%N-1]/(delta_r*psi[n+1][j%N]**3.)) if j==(N+i) 
			   else (-0.5*1./2.*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2.)) if j==(N+i+1) 
			   else 0 for j in range(2*N)]
	
	for i in range(N, 2*N):
	    if(i == N):
		A[i, :] = [-3. if j==i
			   else 4. if j==i+1
			   else -1. if j==i+2
	                   else 0 for j in range(2*N)]
	    elif(i == 2*N-1): #Pi BC
                A[i, :] = [1. + delta_t/2. * (3./(2.*delta_r) + 1./r_grid[j%N]) if j==i
                           else delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else delta_t/2. * (1./(2.*delta_r)) if j==i-2
			   else 0 for j in range(2*N)]
	    else: #TODO: REMOVE ALL CONSTANTS FROM THIS AND ADD THEM TO OTHER SIDE: LHS(A) SHOULD BE CORRECT
	        A[i, :] = [     (0.5*-0.166666666666667*beta[n+1][j%N+1]*delta_r**(-1.00000000000000) - 0.5*-0.166666666666667*beta[n+1][j%N-1]*delta_r**(-1.00000000000000) - 0.5*0.666666666666667*beta[n+1][j%N]*r_grid[j%N]**(-1.00000000000000) + 1./(delta_t))  if j==i
	                   else (0.5*1.00000000000000*1./2.*beta[n+1][j%N]*delta_r**(-1.00000000000000))  if j==i-1
	                   else (-0.5*1./2.*beta[n+1][j%N]*delta_r**(-1.00000000000000))  if j==i+1
			   else (0.5*1.00000000000000*1./2.*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==i-N-1
	                   else (0.5*1.00000000000000*1./2.*alpha[n+1][j%N-1]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000) - 0.5*1.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N+1]/psi[n+1][j%N]**3. + 0.5*1.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N-1]/psi[n+1][j%N]**3. - 0.5*1./2.*alpha[n+1][j%N+1]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000) - 0.5*2.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)*r_grid[j%N]**(-1.00000000000000))  if j==i-N
	                   else (-0.5*1./2.*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==i-N+1
	                   else 0 for j in range(2*N)]
	
	#define matrix B, now fully to second order accuracy
	for i in range(N):
	    if(i == 0):
		B[i, :] = [0 for j in range(2*N)]
	    elif(i == 1): #ADDED KO_ODD_FW
		B[i, :] = [     (0.25*beta[n][j%N+1]/delta_r - 0.25*beta[n][j%N-1]/delta_r + 1./(delta_t) - epsilon/(16.*delta_t) * (6. - 1.))  if j==i
                           else (-0.25*beta[n][j%N]/delta_r + epsilon/(16.*delta_t) * 4.)  if j==(i-1)
                           else (0.25*beta[n][j%N]/delta_r + epsilon/(16.*delta_t) * 4.)  if j==(i+1)
			   else (-epsilon/(16.*delta_t) * 1.) if j==(i+2)
                           else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==(N+i-1)
                           else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2.) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2.) - 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3.) + 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3.))  if j==(N+i)
                           else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==(N+i+1)
                           else 0 for j in range(2*N)]
	    elif(i == N-2): #normal internal equations but without KO diss
		B[i, :] = [     (0.25*beta[n][j%N+1]/delta_r - 0.25*beta[n][j%N-1]/delta_r + 1./(delta_t) - 0./(16.*delta_t) * (6.))  if j==i
                           else (-0.25*beta[n][j%N]/delta_r - 0./(16.*delta_t) * (-4.))  if j==(i-1)
                           else (0.25*beta[n][j%N]/delta_r - 0./(16.*delta_t) * (-4.))  if j==(i+1)
                           else (- 0./(16.*delta_t) * (1.)) if j==(i-2)
                           else (- 0./(16.*delta_t) * (1.)) if j==(i+2)
                           else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==(N+i-1)
                           else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2.) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2.) - 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3.) + 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3.))  if j==(N+i)
                           else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==(N+i+1)
                           else 0 for j in range(2*N)]
	    elif(i == N-1): #xi BC
                B[i, :] = [1. - delta_t/2. * (3./(2.*delta_r) + 1./r_grid[j%N]) if j==i
                           else -delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (1./(2.*delta_r)) if j==i-2
			   else 0 for j in range(2*N)]
	    else: #ADDED DISS_KO to this
	        B[i, :] = [     (0.25*beta[n][j%N+1]/delta_r - 0.25*beta[n][j%N-1]/delta_r + 1./(delta_t) - epsilon/(16.*delta_t) * (6.))  if j==i 
			   else (-0.25*beta[n][j%N]/delta_r - epsilon/(16.*delta_t) * (-4.))  if j==(i-1) 
			   else (0.25*beta[n][j%N]/delta_r - epsilon/(16.*delta_t) * (-4.))  if j==(i+1) 
			   else (- epsilon/(16.*delta_t) * (1.)) if j==(i-2)
			   else (- epsilon/(16.*delta_t) * (1.)) if j==(i+2)
			   else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==(N+i-1) 
			   else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2.) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2.) - 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3.) + 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3.))  if j==(N+i)
			   else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==(N+i+1) 
			   else 0 for j in range(2*N)]
	
	for i in range(N, 2*N):
	    if(i == N):
		B[i, :] = [3. if j==i
			   else -4. if j==i+1
			   else 1. if j==i+2
			   else 0 for j in range(2*N)]
	    elif(i == N+1): #ADDED KO_EVEN_FWD TO THIS
		B[i, :] = [     (0.0833333333333333*beta[n][j%N+1]/delta_r - 0.0833333333333333*beta[n][j%N-1]/delta_r + 0.333333333333333*beta[n][j%N]/r_grid[j%N] + 1./(delta_t) - epsilon/(16.*delta_t) * (1. + 6.))  if j==i
                           else (-0.25*beta[n][j%N]/delta_r - epsilon/(16.*delta_t) * (-4.))  if j==i-1
                           else (0.25*beta[n][j%N]/delta_r - epsilon/(16.*delta_t) * (-4.))  if j==i+1
			   else (- epsilon/(16.*delta_t) * 1.) if j==i+2
                           else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==i-N-1
                           else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2.) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2) + 1.0*alpha[n][j%N]/(psi[n][j%N]**2.*r_grid[j%N]) + 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3.) - 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3.))  if j==i-N
                           else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==i-N+1
                           else 0 for j in range(2*N)]
	    elif(i == 2*N-2): #normal internal equations except without DISS_KO
		B[i, :] = [     (0.0833333333333333*beta[n][j%N+1]/delta_r - 0.0833333333333333*beta[n][j%N-1]/delta_r + 0.333333333333333*beta[n][j%N]/r_grid[j%N] + 1./(delta_t) - 0./(16.*delta_t) * (6.))  if j==i
                           else (-0.25*beta[n][j%N]/delta_r - 0./(16.*delta_t) * (-4.))  if j==i-1
                           else (0.25*beta[n][j%N]/delta_r - 0./(16.*delta_t) * (-4.))  if j==i+1
                           else (- 0./(16.*delta_t) * (1.)) if j==(i-2)
                           else (- 0./(16.*delta_t) * (1.)) if j==(i+2)
                           else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==i-N-1
                           else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2) + 1.0*alpha[n][j%N]/(psi[n][j%N]**2.*r_grid[j%N]) + 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3.) - 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3.))  if j==i-N
                           else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==i-N+1
                           else 0 for j in range(2*N)]
	    elif(i == 2*N-1): #Pi BC
                B[i, :] = [1. - delta_t/2. * (3./(2.*delta_r) + 1./r_grid[j%N]) if j==i
                           else -delta_t/2. * (-4./(2.*delta_r)) if j==i-1
                           else -delta_t/2. * (1./(2.*delta_r)) if j==i-2
			   else 0 for j in range(2*N)]

	    else: #ADDED DISS_KO TO THIS
	        B[i, :] = [     (0.0833333333333333*beta[n][j%N+1]/delta_r - 0.0833333333333333*beta[n][j%N-1]/delta_r + 0.333333333333333*beta[n][j%N]/r_grid[j%N] + 1./(delta_t) - epsilon/(16.*delta_t) * (6.))  if j==i
	                   else (-0.25*beta[n][j%N]/delta_r - epsilon/(16.*delta_t) * (-4.))  if j==i-1
	                   else (0.25*beta[n][j%N]/delta_r - epsilon/(16.*delta_t) * (-4.))  if j==i+1
			   else (- epsilon/(16.*delta_t) * (1.)) if j==(i-2)
			   else (- epsilon/(16.*delta_t) * (1.)) if j==(i+2)
	                   else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==i-N-1
			   else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2.) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2.) + 1.0*alpha[n][j%N]/(psi[n][j%N]**2.*r_grid[j%N]) + 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3.) - 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3.))  if j==i-N
	                   else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2.))  if j==i-N+1
	                   else 0 for j in range(2*N)]

#	if(PSI_EVOL == True):
#		for i in range(N):
#			if(i == 0): #psi r=0 BC
#				psi_LHS[i, :] = [-3. if j==i
#						 else 4. if j==i+1
#						 else -1. if j==i+2
#						 else 0 for j in range(N)]
#			elif(i == N-1): #psi r=R BC
#				psi_LHS[i, :] = [3./(2.*delta_r) + 1./r_grid[j] if j==i
#						 else ]
#			else: #psi internal evolution equation
#				psi_LHS[i, :] = [1. - delta_t/2. * ( beta[n+1][j]/(3.*r_grid[j]) 
#                                                                    + (beta[n+1][j+1]-beta[n+1][j-1])/(6. * 2. * delta_r) ) if j==i
#                                                 else -delta_t/2. * beta[n+1][j]/(2.*delta_r) if j==i+1
#                                                 else delta_t/2. * beta[n+1][j]/(2.*delta_r) if j==i-1
#                                                 else 0 for j in range(N)]

	return 0

def update_u(timestep):
    u = np.zeros(2*N)
    for i in range(2*N):
        if(i < N):
            u[i] = xi[timestep, i]
        else:
            u[i] = Pi[timestep, i-N]
    return u

#update xi and Pi using the solved-for vector ans, which is u at timestep [timestep]
def update_r_s(ans, timestep):
    for i in range(2*N):
        if(i < N):
            xi[timestep, i] = ans[i]
        else:
            Pi[timestep, i-N] = ans[i]

    return 0

for n in range(1, timesteps):
    populate_matrices(n-1) #TODO: not sure this works

    u = update_u(n-1)

    bb = B.dot(u)

    ans = np.linalg.solve(A, bb)

    update_r_s(ans, n)

    if(GEOM_COUPLING == True):
	    #need to solve elliptics before populating CN matrices
	    #first set initial values of f_n
	    f_n = np.zeros(3*N)
	    f_n[0:N]     = psi[n, :]
	    f_n[N:2*N]   = beta[n, :]
	    f_n[2*N:3*N] = alpha[n, :]
	    f_n = solve_elliptics(f_n, xi[n, :], Pi[n, :], r_grid, correction_weight=correction_weight)
	    #now set psi, beta, alpha with solution to elliptics
	    psi[n, :]   = f_n[0:N]
	    beta[n, :]  = f_n[N:2*N]
	    alpha[n, :] = f_n[2*N:3*N]

    for i in range(N):
        #uses O(h^2) Crank-Nicolson time differencing
    	phi[n, i] = ( phi[n-1, i] + 0.5 * delta_t * ((alpha[n][i]/psi[n][i]**2.*Pi[n, i] 
                                                      + beta[n][i]*xi[n][i]) 
                                                    + (alpha[n-1][i]/psi[n-1][i]**2.*Pi[n-1, i] 
                                                       + beta[n-1][i]*xi[n-1, i])) )

print '-----computing mass aspect-----'
mass_aspect = np.zeros((timesteps, N)) 
#this computes the mass aspect function m(r,t)
for n in range(timesteps):
	for j in range(N):
		if(j == 0):
			mass_aspect[n, j] = ( r_grid[j] * psi[n,j]**6./(18.*alpha[n,j]**2.)
                                             *(r_grid[j]*(-3.*beta[n,j]+4.*beta[n,j+1]-beta[n,j+2])/(2.*delta_r)  
                                               - beta[n,j] )**2.
                                             - 2.*r_grid[j]**2.*(-3.*psi[n,j]+4.*psi[n,j+1]-psi[n,j+2])/(2.*delta_r)
                                               *(psi[n,j] + r_grid[j]*(-3.*psi[n,j]+4.*psi[n,j+1]-psi[n,j+2])/(2.*delta_r) ) )
		elif(j == N-1):
			mass_aspect[n, j] = ( r_grid[j] * psi[n,j]**6./(18.*alpha[n,j]**2.)
                                             *(r_grid[j]*(3.*beta[n,j]-4.*beta[n,j-1]+beta[n,j-2])/(2.*delta_r) 
                                               - beta[n,j] )**2. 
                                             - 2.*r_grid[j]**2.*(3.*psi[n,j]-4.*psi[n,j-1]+psi[n,j-2])/(2.*delta_r) 
                                               *(psi[n,j] + r_grid[j]*(3.*psi[n,j]-4.*psi[n,j-1]+psi[n,j-2])/(2.*delta_r) ) )
		else:
			mass_aspect[n, j] = ( r_grid[j] * psi[n,j]**6./(18.*alpha[n,j]**2.)
					     *(r_grid[j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r) 
                                               - beta[n,j] )**2.
					     - 2.*r_grid[j]**2.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
					       *(psi[n,j] + r_grid[j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r) ) )

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
phi_residual = np.zeros((timesteps, N))
xi_residual  = np.zeros((timesteps, N))
Pi_residual  = np.zeros((timesteps, N))

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


for n in range(timesteps-1):
	for j in range(N):
		if(j == 0):
			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t 
					       - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
					       - 0.5*beta[n,j]*xi[n,j] 
					       - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
                                               - 0.5*beta[n+1,j]*xi[n+1,j]) 
			xi_residual[n, j]  = 0.5*(xi[n, j] + xi[n+1,j])
			Pi_residual[n, j]  = ( -Pi[n+1,j+2] + 4.*Pi[n+1,j+1] - 3.*Pi[n+1,j] 
					       -Pi[n,j+2]   + 4.*Pi[n,j+1]   - 3.*Pi[n,j] )
			psi_ev_residual[n, j] = ( -psi[n+1,j+2] + 4.*psi[n+1,j+1] - 3.*psi[n+1,j]
                                               -psi[n,j+2]   + 4.*psi[n,j+1]   - 3.*psi[n,j] )
		elif(j == N-1):
			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t  
                                               - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
                                               - 0.5*beta[n,j]*xi[n,j]
                                               - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
                                               - 0.5*beta[n+1,j]*xi[n+1,j])
			xi_residual[n, j] = ( (xi[n+1,j]-xi[n,j])/delta_t 
                                           + 0.5 * ( (3.*xi[n+1,j] - 4.*xi[n+1,j-1] + xi[n+1, j-2])/(2.*delta_r) 
                                                    + xi[n+1,j]/r_grid[j]
						    +(3.*xi[n,j] - 4.*xi[n,j-1] + xi[n, j-2])/(2.*delta_r)  
						    + xi[n,j]/r_grid[j]) )
			Pi_residual[n, j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
                                           + 0.5 * ( (3.*Pi[n+1,j] - 4.*Pi[n+1,j-1] + Pi[n+1, j-2])/(2.*delta_r)
                                                    + Pi[n+1,j]/r_grid[j]
                                                    +(3.*Pi[n,j] - 4.*Pi[n,j-1] + Pi[n, j-2])/(2.*delta_r)
                                                    + Pi[n,j]/r_grid[j]) )
			psi_ev_residual[n,j] = ( (3.*psi[n,j]-4.*psi[n,j-1]+psi[n,j-2])/(2.*delta_r) + psi[n,j]/r_grid[j]
					     +(3.*psi[n+1,j]-4.*psi[n+1,j-1]+psi[n+1,j-2])/(2.*delta_r) + psi[n+1,j]/r_grid[j] ) 
		else:
			phi_residual[n, j] = ( (phi[n+1,j]-phi[n,j])/delta_t  
                                               - 0.5*alpha[n,j]/psi[n,j]**2.*Pi[n,j]
                                               - 0.5*beta[n,j]*xi[n,j]
                                               - 0.5*alpha[n+1,j]/psi[n+1,j]**2.*Pi[n+1,j]
                                               - 0.5*beta[n+1,j]*xi[n+1,j])
			xi_residual[n, j] = ( (xi[n+1,j]-xi[n,j])/delta_t
					    -0.5*(-2.*Pi[n,j]*alpha[n,j]/psi[n,j]**3. * (psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
						  + Pi[n,j]/psi[n,j]**2. * (alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
						  + alpha[n,j]/psi[n,j]**2. * (Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
						  + beta[n,j] * (xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
						  + xi[n,j] * (beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
						  -2.*Pi[n+1,j]*alpha[n+1,j]/psi[n+1,j]**3. * (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)    
                                                  + Pi[n+1,j]/psi[n+1,j]**2. * (alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)    
                                                  + alpha[n+1,j]/psi[n+1,j]**2. * (Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)    
                                                  + beta[n+1,j] * (xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)    
                                                  + xi[n+1,j] * (beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)    )
						)
			Pi_residual[n, j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
					     -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
					     -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
					     -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
					     -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
					     -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
					     -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
					     -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
					     -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
                                             -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
                                             -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
                                             -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                             -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
					    )
			psi_ev_residual[n,j] = ( (psi[n+1,j]-psi[n,j])/delta_t 
						-0.5*( beta[n,j]*(psi[n,j]/(3.*r_grid[j]) + (psi[n,j+1]-psi[n,j-1])/(2.*delta_r) )
						      +psi[n,j]/6.*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
						      +beta[n+1,j]*(psi[n+1,j]/(3.*r_grid[j]) + (psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r) )
                                                      +psi[n+1,j]/6.*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)   
							)
					    )

np.savetxt('phi_residual.txt', phi_residual)
np.savetxt('xi_residual.txt', xi_residual)
np.savetxt('Pi_residual.txt', Pi_residual) 
np.savetxt('psi_residual.txt', psi_residual)
np.savetxt('psi_ev_residual.txt', psi_ev_residual)
np.savetxt('beta_residual.txt', beta_residual)
np.savetxt('alpha_residual.txt', alpha_residual)

print '-----done-----'
