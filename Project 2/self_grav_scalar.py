import numpy as np
import pylab as pl
import subprocess
from scipy.integrate import cumtrapz, simps

#set parameters for simulation
N = 100
delta_r = 1./N
delta_t = 0.0005
courant = delta_t / delta_r
timesteps = 100

#define grid
M     = 1.
R     = 100. 
amp   = 1.
r_0   = 50.
delta = 10.

r_grid = np.linspace(2. * M, R, N)


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
		xi[0, i] = (phi[0, i+1] - phi[0, i-1])/(2. * delta_r)

#need to set up "approximately" ingoing initial data; do so using the same prescription
#as we used in Project 1, which is identical except except a->\psi^2 
#we are asked to define this condition only in terms of phi, xi, r; using approximation in example code
Pi[0, :] = xi[0, :]

A = np.zeros((2*N, 2*N))
B = np.zeros((2*N, 2*N))

#populate the matrix at timestep n
def populate_matrices(n):
	#define matrix A
	for i in range(N):
	    if(i == 0):
	        A[i, :] = [     (-0.5*-1/2*beta[n+1][j%N+2]/delta_r - 1.0*beta[n+1][j%N+1]/delta_r + 1.5*beta[n+1][j%N]/delta_r + 1/(2*delta_t)) if j==0 
			   else (-1.0*beta[n+1][j%N]/delta_r) if j==1 
			   else (-0.5*-1/2*beta[n+1][j%N]/delta_r) if j==2
			   else (-0.5*-1/2*alpha[n+1][j%N+2]/(delta_r*psi[n+1][j%N]**2) - 1.0*alpha[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**2) + 2.0*alpha[n+1][j%N]*psi[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**3) - 0.5*alpha[n+1][j%N]*psi[n+1][j%N+2]/(delta_r*psi[n+1][j%N]**3)) if j==N 
			   else (-1.0*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==N+1 
			   else (-0.5*-1/2*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==N+2
			   else 0 for j in range(2*N)]
	    elif(i == N-1):
	        A[i, :] = [     (-0.5*1/2*beta[n+1][j%N-2]/delta_r + 1.0*beta[n+1][j%N-1]/delta_r - 1.5*beta[n+1][j%N]/delta_r + 1/(2*delta_t)) if j==N-1 
			   else (1.0*beta[n+1][j%N]/delta_r) if j==N-2 
			   else (-0.5*1/2*beta[n+1][j%N]/delta_r) if j==N-3
			   else (-0.5*1/2*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==2*N-3
			   else (1.0*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==2*N-2 
			   else (-0.5*1/2*alpha[n+1][j%N-2]/(delta_r*psi[n+1][j%N]**2) + 1.0*alpha[n+1][j%N-1]/(delta_r*psi[n+1][j%N]**2) - 2.0*alpha[n+1][j%N]*psi[n+1][j%N-1]/(delta_r*psi[n+1][j%N]**3) + 0.5*alpha[n+1][j%N]*psi[n+1][j%N-2]/(delta_r*psi[n+1][j%N]**3)) if j==2*N-1 
			   else 0 for j in range(2*N)]
	    else:
	        A[i, :] = [     (-0.5*1/2*beta[n+1][j%N+1]/delta_r + 0.5*1/2*beta[n+1][j%N-1]/delta_r + 1/(2*delta_t)) if j==i 
			   else (0.5*1/2*beta[n+1][j%N]/delta_r) if j==(i-1) 
			   else (-0.5*1/2*beta[n+1][j%N]/delta_r) if j==(i+1) 
			   else (0.5*1/2*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==(N+i-1)
			   else (-0.5*1/2*alpha[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**2) + 0.5*1/2*alpha[n+1][j%N-1]/(delta_r*psi[n+1][j%N]**2) + 0.5*alpha[n+1][j%N]*psi[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**3) - 0.5*alpha[n+1][j%N]*psi[n+1][j%N-1]/(delta_r*psi[n+1][j%N]**3)) if j==(N+i) 
			   else (-0.5*1/2*alpha[n+1][j%N]/(delta_r*psi[n+1][j%N]**2)) if j==(N+i+1) 
			   else 0 for j in range(2*N)]
	
	for i in range(N, 2*N):
	    if(i == N):
	        A[i, :] = [     (-0.5*-0.166666666666667*beta[n+1][j%N+2]/delta_r - 0.5*-2.00000000000000*beta[n+1][j%N]/delta_r - 0.5*0.666666666666667*beta[n+1][j%N+1]/delta_r - 0.5*0.666666666666667*beta[n+1][j%N]/r_grid[j%N] + 1/(2*delta_t)) if j==N
	                   else (0.5*-2.00000000000000*beta[n+1][j%N]/delta_r) if j==N+1
			   else (-0.5*-1/2*beta[n+1][j%N]/delta_r) if j==N+2
	                   else (-0.5*-1.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N+2]/(delta_r*psi[n+1][j%N]**3) - 0.5*-1/2*alpha[n+1][j%N+2]*psi[n+1][j%N]**(-2.00000000000000)/delta_r + 0.5*-2.00000000000000*alpha[n+1][j%N+1]*psi[n+1][j%N]**(-2.00000000000000)/delta_r + 0.5*-2.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)/r_grid[j%N] - 0.5*-6.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)/delta_r - 0.5*4.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N+1]/(delta_r*psi[n+1][j%N]**3)) if j==0
	                   else (0.5*-2.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)/delta_r) if j==1
			   else (-0.5*-1/2*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)/delta_r) if j==2
	                   else 0 for j in range(2*N)]
	    elif(i == 2*N-1):
	        A[i, :] = [     (-0.5*0.166666666666667*beta[n+1][j%N-2]*delta_r**(-1.00000000000000) + 0.5*0.666666666666667*1.00000000000000*beta[n+1][j%N-1]*delta_r**(-1.00000000000000) - 0.5*0.666666666666667*beta[n+1][j%N]*r_grid[j%N]**(-1.00000000000000) - 0.5*2.00000000000000*beta[n+1][j%N]*delta_r**(-1.00000000000000) + 1/(2*delta_t))  if j==2*N-1
	                   else (0.5*2.00000000000000*beta[n+1][j%N]*delta_r**(-1.00000000000000))  if j==2*N-2
			   else (-0.5*1/2*beta[n+1][j%N]*delta_r**(-1.00000000000000))  if j==2*N-3
	                   else (-0.5*-4.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N-1]/psi[n+1][j%N]**3 - 0.5*1.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N-2]/psi[n+1][j%N]**3 - 0.5*1/2*alpha[n+1][j%N-2]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000) + 0.5*2.00000000000000*alpha[n+1][j%N-1]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000) - 0.5*2.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)*r_grid[j%N]**(-1.00000000000000) - 0.5*6.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==N-1
	                   else (0.5*2.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==N-2
			   else (-0.5*1/2*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==N-3
	                   else 0 for j in range(2*N)]
	    else:
	        A[i, :] = [     (0.5*-0.166666666666667*beta[n+1][j%N+1]*delta_r**(-1.00000000000000) - 0.5*-0.166666666666667*beta[n+1][j%N-1]*delta_r**(-1.00000000000000) - 0.5*0.666666666666667*beta[n+1][j%N]*r_grid[j%N]**(-1.00000000000000) + 1/(2*delta_t))  if j==i
	                   else (0.5*1.00000000000000*1/2*beta[n+1][j%N]*delta_r**(-1.00000000000000))  if j==i-1
	                   else (-0.5*1/2*beta[n+1][j%N]*delta_r**(-1.00000000000000))  if j==i+1
			   else (0.5*1.00000000000000*1/2*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==i-N-1
	                   else (0.5*1.00000000000000*1/2*alpha[n+1][j%N-1]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000) - 0.5*1.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N+1]/psi[n+1][j%N]**3 + 0.5*1.00000000000000*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N-1]/psi[n+1][j%N]**3 - 0.5*1/2*alpha[n+1][j%N+1]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000) - 0.5*2.00000000000000*alpha[n+1][j%N]*psi[n+1][j%N]**(-2.00000000000000)*r_grid[j%N]**(-1.00000000000000))  if j==i-N
	                   else (-0.5*1/2*alpha[n+1][j%N]*delta_r**(-1.00000000000000)*psi[n+1][j%N]**(-2.00000000000000))  if j==i-N+1
	                   else 0 for j in range(2*N)]
	
	#define matrix B, now fully to second order accuracy
	for i in range(N):
	    if(i == 0):
	        B[i, :] = [     (1.0*beta[n][j%N+1]/delta_r - 0.25*beta[n][j%N+2]/delta_r - 1.5*beta[n][j%N]/delta_r + 1/(2*delta_t))  if j==0 
			   else (1.0*beta[n][j%N]/delta_r)  if j==1 
			   else (-0.25*beta[n][j%N]/delta_r)  if j==2
			   else (1.0*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2) - 0.25*alpha[n][j%N+2]/(delta_r*psi[n][j%N]**2) - 2.0*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3) + 0.5*alpha[n][j%N]*psi[n][j%N+2]/(delta_r*psi[n][j%N]**3))  if j==N 
			   else (1.0*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==N+1 
			   else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==N+2
			   else 0 for j in range(2*N)]
	    elif(i == N-1):
	        B[i, :] = [     (-1.0*beta[n][j%N-1]/delta_r + 0.25*beta[n][j%N-2]/delta_r + 1.5*beta[n][j%N]/delta_r + 1/(2*delta_t))  if j==N-1 
			   else (-1.0*beta[n][j%N]/delta_r)  if j==N-2 
			   else (0.25*beta[n][j%N]/delta_r)  if j==N-3
			   else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==2*N-3
			   else (-1.0*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==2*N-2 
			   else (-1.0*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2) + 0.25*alpha[n][j%N-2]/(delta_r*psi[n][j%N]**2) + 2.0*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3) - 0.5*alpha[n][j%N]*psi[n][j%N-2]/(delta_r*psi[n][j%N]**3))  if j==2*N-1 
			   else 0 for j in range(2*N)]
	    else:
	        B[i, :] = [     (0.25*beta[n][j%N+1]/delta_r - 0.25*beta[n][j%N-1]/delta_r + 1/(2*delta_t))  if j==i 
			   else (-0.25*beta[n][j%N]/delta_r)  if j==(i-1) 
			   else (0.25*beta[n][j%N]/delta_r)  if j==(i+1) 
			   else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==(N+i-1) 
			   else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2) - 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3) + 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3))  if j==(N+i)
			   else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==(N+i+1) 
			   else 0 for j in range(2*N)]
	
	for i in range(N, 2*N):
	    if(i == N):
	        B[i, :] = [     (0.333333333333333*beta[n][j%N+1]/delta_r - 0.0833333333333333*beta[n][j%N+2]/delta_r + 0.333333333333333*beta[n][j%N]/r_grid[j%N] - 1.0*beta[n][j%N]/delta_r + 1/(2*delta_t))  if j==N
	                   else (1.0*beta[n][j%N]/delta_r)  if j==N+1
			   else (-0.25*beta[n][j%N]/delta_r)  if j==N+2
	                   else (1.0*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2) - 0.25*alpha[n][j%N+2]/(delta_r*psi[n][j%N]**2) + 1.0*alpha[n][j%N]/(psi[n][j%N]**2*r_grid[j%N]) + 2.0*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3) - 0.5*alpha[n][j%N]*psi[n][j%N+2]/(delta_r*psi[n][j%N]**3) - 3.0*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==0
	                   else (1.0*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==1
			   else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==2
	                   else 0 for j in range(2*N)]
	    elif(i == 2*N-1):
	        B[i, :] = [     (-0.333333333333333*beta[n][j%N-1]/delta_r + 0.0833333333333333*beta[n][j%N-2]/delta_r + 0.333333333333333*beta[n][j%N]/r_grid[j%N] + 1.0*beta[n][j%N]/delta_r + 1/(2*delta_t))  if j==2*N-1
	                   else (-1.0*beta[n][j%N]/delta_r)  if j==2*N-2
			   else (0.25*beta[n][j%N]/delta_r)  if j==2*N-3
	                   else (-1.0*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2) + 0.25*alpha[n][j%N-2]/(delta_r*psi[n][j%N]**2) + 1.0*alpha[n][j%N]/(psi[n][j%N]**2*r_grid[j%N]) - 2.0*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3) + 0.5*alpha[n][j%N]*psi[n][j%N-2]/(delta_r*psi[n][j%N]**3) + 3.0*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==N-1
	                   else (-1.0*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==N-2
			   else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==N-3
	                   else 0 for j in range(2*N)]
	    else:
	        B[i, :] = [     (0.0833333333333333*beta[n][j%N+1]/delta_r - 0.0833333333333333*beta[n][j%N-1]/delta_r + 0.333333333333333*beta[n][j%N]/r_grid[j%N] + 1/(2*delta_t))  if j==i
	                   else (-0.25*beta[n][j%N]/delta_r)  if j==i-1
	                   else (0.25*beta[n][j%N]/delta_r)  if j==i+1
	                   else (-0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==i-N-1
			   else (0.25*alpha[n][j%N+1]/(delta_r*psi[n][j%N]**2) - 0.25*alpha[n][j%N-1]/(delta_r*psi[n][j%N]**2) + 1.0*alpha[n][j%N]/(psi[n][j%N]**2*r_grid[j%N]) + 0.5*alpha[n][j%N]*psi[n][j%N+1]/(delta_r*psi[n][j%N]**3) - 0.5*alpha[n][j%N]*psi[n][j%N-1]/(delta_r*psi[n][j%N]**3))  if j==i-N
	                   else (0.25*alpha[n][j%N]/(delta_r*psi[n][j%N]**2))  if j==i-N+1
	                   else 0 for j in range(2*N)]
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

	#establish RADIATION ZONE to prevent reflection at outer boundary
#	Pi[timestep, N-1] = phi[timestep, N-1]/r_grid[N-1]*a(r_grid[N-1])/alpha(r_grid[N-1]) * (beta(r_grid[N-1]) - alpha(r_grid[N-1])/a(r_grid[N-1])) - Phi[timestep, N-1] #outgoing radiation condition
#	Pi[timestep, N-2] = phi[timestep, N-2]/r_grid[N-2]*a(r_grid[N-2])/alpha(r_grid[N-2]) * (beta(r_grid[N-2]) - alpha(r_grid[N-2])/a(r_grid[N-2])) - Phi[timestep, N-2] 
#	Pi[timestep, N-3] = phi[timestep, N-3]/r_grid[N-3]*a(r_grid[N-3])/alpha(r_grid[N-3]) * (beta(r_grid[N-3]) - alpha(r_grid[N-3])/a(r_grid[N-3])) - Phi[timestep, N-3]
    return 0

for n in range(1, timesteps):
    populate_matrices(n-1) #TODO: not sure this works

    u = update_u(n-1)

    bb = B.dot(u)

    ans = np.linalg.solve(A, bb)

    update_r_s(ans, n)


for n in range(timesteps):
	pl.plot(r_grid, xi[n][:])
	pl.savefig('%04d'%n + '.png')
	pl.close()
	pl.clf()

#    for i in range(N):
##	    phi[n, i] = phi[n-1, i] + delta_t * (alpha(r_grid[i])/a(r_grid[i])*Pi[n-1, i] + beta(r_grid[i])*Phi[n-1, i]) #uses O(h) forward time-differencing
#    	    phi[n, i] = phi[n-1, i] + 0.5 * delta_t * ((alpha(r_grid[i])/a(r_grid[i])*Pi[n, i] + beta(r_grid[i])*Phi[n, i]) + (alpha(r_grid[i])/a(r_grid[i])*Pi[n-1, i] + beta(r_grid[i])*Phi[n-1, i]))	#uses O(h^2) Crank-Nicolson time differencing
#
##this computes the mass aspect function dm/dr
#mass_aspect = 4. * np.pi * r_grid**2. * (alpha(r_grid)/(2. * a(r_grid)) * (Phi**2. + Pi**2.) + beta(r_grid)*Phi*Pi)
#
##this computes the total mass in the window \int_{2M}^{R} dm/dr dr
#m = np.zeros(timesteps)
#for i in range(timesteps):
#    m[i] = simps(mass_aspect[i, :], r_grid)
#
## Set plot parameters to make beautiful plots
#pl.rcParams['figure.figsize']  = 10, 10
#pl.rcParams['lines.linewidth'] = 1.5
#pl.rcParams['font.family']     = 'serif'
#pl.rcParams['font.weight']     = 'bold'
#pl.rcParams['font.size']       = 15
#pl.rcParams['font.sans-serif'] = 'serif'
#pl.rcParams['text.usetex']     = True
#pl.rcParams['axes.linewidth']  = 1.5
#pl.rcParams['axes.titlesize']  = 'large'
#pl.rcParams['axes.labelsize']  = 'large'
#
#pl.rcParams['xtick.major.size'] = 8
#pl.rcParams['xtick.minor.size'] = 4
#pl.rcParams['xtick.major.pad']  = 8
#pl.rcParams['xtick.minor.pad']  = 8
#pl.rcParams['xtick.color']      = 'k'
#pl.rcParams['xtick.labelsize']  = 'large'
#pl.rcParams['xtick.direction']  = 'in'
#
#pl.rcParams['ytick.major.size'] = 8
#pl.rcParams['ytick.minor.size'] = 4
#pl.rcParams['ytick.major.pad']  = 8
#pl.rcParams['ytick.minor.pad']  = 8
#pl.rcParams['ytick.color']      = 'k'
#pl.rcParams['ytick.labelsize']  = 'large'
#pl.rcParams['ytick.direction']  = 'in'
#
##this plots the total mass at each timestep
#pl.plot(range(timesteps), m)
#pl.title('Total Mass')
#pl.xlabel('Timestep')
#pl.ylabel('$$m$$')
#pl.savefig('total_mass.png')
#
##make temp folder to save frames which will be made into the movie
#command0 = subprocess.Popen('mkdir temp_folder/'.split(), stdout=subprocess.PIPE)
#command0.wait()
#
##save frames to make movie
#for i in range(timesteps):
#    if(i % 50 == 0):
#	print 'saving frame ' + str(i) + ' out of ' + str(timesteps)
#    figure, ax = pl.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
#    ax[0,0].plot(r_grid, phi[i, :])
#    ax[0,0].set_title('$$\\phi$$')
#    ax[1,0].plot(r_grid, Phi[i, :])
#    ax[1,0].set_title('$$\\Phi$$')
#    ax[1,1].plot(r_grid, Pi[i, :])
#    ax[1,1].set_title('$$\\Pi$$')
#    ax[0,1].plot(r_grid, mass_aspect[i, :])
#    ax[0,1].set_title('$$\\frac{dm}{dr}$$')
# 
#    #draw x label $r$
#    figure.add_subplot(111, frameon=False)
#    pl.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#    pl.xlabel('$r$', fontsize='large')
# 
#    #set y limits on plots
#    ax[0,0].set_ylim(-14., 10.)
#    ax[0,1].set_ylim(0., 1.5e7)
#    ax[1,0].set_ylim(-100., 60.)
#    ax[1,1].set_ylim(-100., 60.)
#
#    #save frames, close frames, clear memory
#    pl.savefig('temp_folder/%03d'%i + '.png')
#    pl.close()
#    pl.clf()
#
##make movie
#command1 = subprocess.Popen('ffmpeg -y -i temp_folder/%03d.png IEF_scalar_rad.m4v'.split(), stdout=subprocess.PIPE)
#command1.wait()
#command2 = subprocess.Popen('rm -r temp_folder/'.split(), stdout=subprocess.PIPE)
#command2.wait()
