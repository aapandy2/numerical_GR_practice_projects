import numpy as np
import pylab as pl
import subprocess
from scipy.integrate import cumtrapz, simps

#set parameters for simulation
N = 300
delta_x = 1./N
delta_t = 0.005
courant = delta_t / delta_x
timesteps = 800

#define grid
M = 1.
R = 100. 
r_grid = np.linspace(2. * M, R, N)

#initialize arrays
phi = np.zeros((timesteps, N))
Phi = np.zeros((timesteps, N)) 
Pi  = np.zeros((timesteps, N))

#define initial data
r_0   = 50.
delta = 5.
amp   = 1.

def alpha(r):
	ans = (r/(r + 2. * M))**(1./2.)
	return ans

def a(r):
	ans = 1./alpha(r)
	return ans

def beta(r):
	ans = (2. * M) / (r + 2. * M)
	return ans

phi[0, :] = amp * np.exp(-(r_grid-r_0)**2./delta**2.)

#Note: need to set Phi as the numerical derivative of phi;
#using an analytical derivative gives incorrect result
for i in range(N):
	if(i == 0):
#		Phi[0, i] = (phi[0, i+1] - phi[0, i])  /delta_x
		Phi[0, i] = (-phi[0, i+2] + 4.*phi[0, i+1] - 3.*phi[0, i])/(2.*delta_x)
	elif(i == N-1):
#		Phi[0, i] = (phi[0, i]   - phi[0, i-1])/delta_x
		Phi[0, i] = (phi[0, i-2] - 4.*phi[0, i-1] + 3.*phi[0, i])/(2.*delta_x)
	else:
		Phi[0, i] = (phi[0, i+1] - phi[0, i-1])/(2. * delta_x)

Pi[0, :]  = a(r_grid)/alpha(r_grid) * (1./r_grid * phi[0, :] + (1. - beta(r_grid)) * Phi[0, :])


A = np.zeros((2*N, 2*N))
B = np.zeros((2*N, 2*N))

#define matrix A
for i in range(N):
    if(i == 0):
        A[i, :] = [1. + 3.*courant/4.*beta(r_grid[j])                    if j==0 
		   else -4.*courant/4.*beta(r_grid[j])                   if j==1 
		   else courant/4.*beta(r_grid[j])                       if j==2
		   else 3.*courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])  if j==N 
		   else -4.*courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N]) if j==N+1 
		   else courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])     if j==N+2
		   else 0 for j in range(2*N)]
    elif(i == N-1):
        A[i, :] = [1 - 3.*courant/4.*beta(r_grid[j])                        if j==N-1 
		   else 4.*courant/4.*beta(r_grid[j])                       if j==N-2 
		   else -courant/4.*beta(r_grid[j])                         if j==N-3
		   else -courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])   if j==2*N-3
		   else 4.*courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==2*N-2 
		   else -3*courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==2*N-1 
		   else 0 for j in range(2*N)]

    else:
        A[i, :] = [1                                                  if j==i 
		   else courant/4.*beta(r_grid[j])                    if j==(i-1) 
		   else -courant/4.*beta(r_grid[j])                   if j==(i+1) 
		   else courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])  if j==(N+i-1) 
		   else -courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N]) if j==(N+i+1) 
		   else 0 for j in range(2*N)]

for i in range(N, 2*N):
    if(i == N):
        A[i, :] = [1. + 3.*courant/4.*beta(r_grid[j%N])                                              if j==N
                   else -4.*courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==N+1
		   else courant/(4.*r_grid[j%N-2]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])             if j==N+2
                   else 3.*courant/4.*alpha(r_grid[j])/a(r_grid[j])                                  if j==0
                   else -4.*courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==1
		   else courant/(4.*r_grid[j-2]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])     if j==2
                   else 0 for j in range(2*N)]
    elif(i == 2*N-1):
        A[i, :] = [1 - 3.*courant/4.*beta(r_grid[j%N])                                              if j==2*N-1
                   else 4.*courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==2*N-2
		   else -courant/(4.*r_grid[j%N+2]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])           if j==2*N-3
                   else -3.*courant/4.*alpha(r_grid[j])/a(r_grid[j])                                if j==N-1
                   else 4.*courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==N-2
		   else -courant/(4.*r_grid[j+2]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])   if j==N-3
                   else 0 for j in range(2*N)]
    else:
        A[i, :] = [1                                                                             if j==i
                   else courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==i-1
                   else -courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])        if j==i+1
                   else courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==i-N-1
                   else -courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])if j==i-N+1
                   else 0 for j in range(2*N)]

#define matrix B, now fully to second order accuracy
for i in range(N):
    if(i == 0):
        B[i, :] = [1. - 3.*courant/4.*beta(r_grid[j])                    if j==0 
		   else 4.*courant/4.*beta(r_grid[j])                    if j==1 
		   else -courant/4.*beta(r_grid[j])                      if j==2
		   else -3.*courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N]) if j==N 
		   else 4.*courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])  if j==N+1 
		   else -courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])    if j==N+2
		   else 0 for j in range(2*N)]
    elif(i == N-1):
        B[i, :] = [1 + 3.*courant/4.*beta(r_grid[j])                         if j==N-1 
		   else -4.*courant/4.*beta(r_grid[j])                       if j==N-2 
		   else courant/4.*beta(r_grid[j])                           if j==N-3
		   else courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])     if j==2*N-3
		   else -4.*courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==2*N-2 
		   else 3*courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])   if j==2*N-1 
		   else 0 for j in range(2*N)]
    else:
        B[i, :] = [1                                                  if j==i 
		   else -courant/4.*beta(r_grid[j])                   if j==(i-1) 
		   else courant/4.*beta(r_grid[j])                    if j==(i+1) 
		   else -courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N]) if j==(N+i-1) 
		   else courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])  if j==(N+i+1) 
		   else 0 for j in range(2*N)]

for i in range(N, 2*N):
    if(i == N):
        B[i, :] = [1. - 3.*courant/4.*beta(r_grid[j%N])                                             if j==N
                   else 4.*courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==N+1
		   else -courant/(4.*r_grid[j%N-2]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])           if j==N+2
                   else -3.*courant/4.*alpha(r_grid[j])/a(r_grid[j])                                if j==0
                   else 4.*courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==1
		   else -courant/(4.*r_grid[j-2]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])   if j==2
                   else 0 for j in range(2*N)]
    elif(i == 2*N-1):
        B[i, :] = [1 + 3.*courant/4.*beta(r_grid[j%N])                                               if j==2*N-1
                   else -4.*courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==2*N-2
		   else courant/(4.*r_grid[j%N+2]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])             if j==2*N-3
                   else 3.*courant/4.*alpha(r_grid[j])/a(r_grid[j])                                  if j==N-1
                   else -4.*courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==N-2
		   else courant/(4.*r_grid[j+2]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])     if j==N-3
                   else 0 for j in range(2*N)]
    else:
        B[i, :] = [1                                                                              if j==i
                   else -courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==i-1
                   else courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])          if j==i+1
                   else -courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==i-N-1
                   else courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])  if j==i-N+1
                   else 0 for j in range(2*N)]

def update_u(timestep):
    u = np.zeros(2*N)
    for i in range(2*N):
        if(i < N):
            u[i] = Phi[timestep, i]
        else:
            u[i] = Pi[timestep, i-N]
    return u

#update Phi and Pi using the solved-for vector ans, which is u at timestep [timestep]
def update_r_s(ans, timestep):
    for i in range(2*N):
        if(i < N):
            Phi[timestep, i] = ans[i]
        else:
            Pi[timestep, i-N] = ans[i]

	#establish RADIATION ZONE to prevent reflection at outer boundary
	Pi[timestep, N-1] = phi[timestep, N-1]/r_grid[N-1]*a(r_grid[N-1])/alpha(r_grid[N-1]) * (beta(r_grid[N-1]) - alpha(r_grid[N-1])/a(r_grid[N-1])) - Phi[timestep, N-1] #outgoing radiation condition
	Pi[timestep, N-2] = phi[timestep, N-2]/r_grid[N-2]*a(r_grid[N-2])/alpha(r_grid[N-2]) * (beta(r_grid[N-2]) - alpha(r_grid[N-2])/a(r_grid[N-2])) - Phi[timestep, N-2] 
	Pi[timestep, N-3] = phi[timestep, N-3]/r_grid[N-3]*a(r_grid[N-3])/alpha(r_grid[N-3]) * (beta(r_grid[N-3]) - alpha(r_grid[N-3])/a(r_grid[N-3])) - Phi[timestep, N-3]
    return 0

for n in range(1, timesteps):
    u = update_u(n-1)

    bb = B.dot(u)

    ans = np.linalg.solve(A, bb)

    update_r_s(ans, n)

    for i in range(N):
#	    phi[n, i] = phi[n-1, i] + delta_t * (alpha(r_grid[i])/a(r_grid[i])*Pi[n-1, i] + beta(r_grid[i])*Phi[n-1, i]) #uses O(h) forward time-differencing
    	    phi[n, i] = phi[n-1, i] + 0.5 * delta_t * ((alpha(r_grid[i])/a(r_grid[i])*Pi[n, i] + beta(r_grid[i])*Phi[n, i]) + (alpha(r_grid[i])/a(r_grid[i])*Pi[n-1, i] + beta(r_grid[i])*Phi[n-1, i]))	#uses O(h^2) Crank-Nicolson time differencing

#this computes the mass aspect function dm/dr
mass_aspect = 4. * np.pi * r_grid**2. * (alpha(r_grid)/(2. * a(r_grid)) * (Phi**2. + Pi**2.) + beta(r_grid)*Phi*Pi)

#this computes the total mass in the window \int_{2M}^{R} dm/dr dr
m = np.zeros(timesteps)
for i in range(timesteps):
    m[i] = simps(mass_aspect[i, :], r_grid)

#this plots the total mass at each timestep
pl.plot(range(timesteps), m)
pl.title('Total Mass')
pl.xlabel('Timestep')
pl.ylabel('m')
pl.show()

#make temp folder to save frames which will be made into the movie
command0 = subprocess.Popen('mkdir temp_folder/'.split(), stdout=subprocess.PIPE)
command0.wait()

#save frames to make movie
for i in range(timesteps):
    if(i % 50 == 0):
	print 'saving frame ' + str(i) + ' out of ' + str(timesteps)
    pl.plot(r_grid, phi[i, :])
#    pl.plot(r_grid, mass_aspect[i, :])
#    pl.xlim([r_grid[0]-0.1, r_grid[N-1]])
#    pl.ylim([-100., 60.])
#    pl.ylim([-0.05, 0.5])
    pl.ylim([-14., 14.])
    pl.savefig('temp_folder/%03d'%i + '.png')
    pl.clf()

#make movie
command1 = subprocess.Popen('ffmpeg -y -i temp_folder/%03d.png IEF_scalar_rad.m4v'.split(), stdout=subprocess.PIPE)
command1.wait()
command2 = subprocess.Popen('rm -r temp_folder/'.split(), stdout=subprocess.PIPE)
command2.wait()
