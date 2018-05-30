import numpy as np
import pylab as pl
import subprocess

#set parameters for simulation
N = 300
delta_x = 1./N
delta_t = 0.005
courant = delta_t / delta_x
timesteps = 250

#define grid
M = 1.
R = 10. #if M=1, this is 10M
r_grid = np.linspace(2. * M, R, N)

#initialize arrays
phi = np.zeros((timesteps, N))
Phi = np.zeros((timesteps, N)) #this is r
Pi  = np.zeros((timesteps, N)) #this is s

#define initial data
r_0   = 5.
delta = 0.05
amp   = 0.5

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
Phi[0, :] = 2. * amp * (r_grid - r_0)/delta**2. * np.exp(-(r_grid-r_0)**2./delta**2.)
Pi[0, :]  = a(r_grid)/alpha(r_grid) * (1./r_grid * phi[0, :] + (1. - beta(r_grid)) * Phi[0, :])

#pl.plot(r_grid, phi[0, :])
#pl.plot(r_grid, Phi[0, :])
#pl.plot(r_grid, Pi[0, :])
#pl.show()

A = np.zeros((2*N, 2*N))
B = np.zeros((2*N, 2*N))

#define matrix A
for i in range(N):
    if(i == 0):
        A[i, :] = [1. + courant/4.*beta(r_grid[j])                        if j==0 
		   else -courant/4.*beta(r_grid[j])                       if j==1 
		   else courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])  if j==N 
		   else -courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==N+1 
		   else 0 for j in range(2*N)]
    elif(i == N-1):
        A[i, :] = [1 - courant/4.*beta(r_grid[j])                         if j==N-1 
		   else courant/4.*beta(r_grid[j])                        if j==N-2 
		   else courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])  if j==2*N-2 
		   else -courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==2*N-1 
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
        A[i, :] = [1. + courant/4.*beta(r_grid[j%N])                                              if j==N
                   else -courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==N+1
                   else courant/4.*alpha(r_grid[j])/a(r_grid[j])                                  if j==0
                   else -courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==1
                   else 0 for j in range(2*N)]
    elif(i == 2*N-1):
        A[i, :] = [1 - courant/4.*beta(r_grid[j%N])                                             if j==2*N-1
                   else courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])        if j==2*N-2
                   else -courant/4.*alpha(r_grid[j])/a(r_grid[j])                               if j==N-1
                   else courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])if j==N-2
                   else 0 for j in range(2*N)]
    else:
        A[i, :] = [1                                                                             if j==i
                   else courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==i-1
                   else -courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])        if j==i+1
                   else courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==i-N-1
                   else -courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])if j==i-N+1
                   else 0 for j in range(2*N)]

for i in range(N):
    if(i == 0):
        B[i, :] = [1. - courant/4.*beta(r_grid[j])                        if j==0
                   else courant/4.*beta(r_grid[j])                       if j==1
                   else -courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])  if j==N
                   else courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==N+1
                   else 0 for j in range(2*N)]
    elif(i == N-1):
        B[i, :] = [1 + courant/4.*beta(r_grid[j])                         if j==N-1
                   else -courant/4.*beta(r_grid[j])                        if j==N-2
                   else -courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N])  if j==2*N-2
                   else courant/4.*alpha(r_grid[j % N])/a(r_grid[j % N]) if j==2*N-1
                   else 0 for j in range(2*N)]
    else:
        B[i, :] = [1                                                  if j==i
                   else -courant/4.*beta(r_grid[j])                    if j==(i-1)
                   else courant/4.*beta(r_grid[j])                   if j==(i+1)
                   else -courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N])  if j==(N+i-1)
                   else courant/4.*alpha(r_grid[j%N])/a(r_grid[j%N]) if j==(N+i+1)
                   else 0 for j in range(2*N)]

for i in range(N, 2*N):
    if(i == N):
        B[i, :] = [1. - courant/4.*beta(r_grid[j%N])                                              if j==N
                   else courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==N+1
                   else -courant/4.*alpha(r_grid[j])/a(r_grid[j])                                  if j==0
                   else courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==1
                   else 0 for j in range(2*N)]
    elif(i == 2*N-1):
        B[i, :] = [1 + courant/4.*beta(r_grid[j%N])                                             if j==2*N-1
                   else -courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])        if j==2*N-2
                   else courant/4.*alpha(r_grid[j])/a(r_grid[j])                               if j==N-1
                   else -courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])if j==N-2
                   else 0 for j in range(2*N)]
    else:
        B[i, :] = [1                                                                             if j==i
                   else -courant/(4.*r_grid[j%N+1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])         if j==i-1
                   else courant/(4.*r_grid[j%N-1]**2.)*r_grid[j%N]**2.*beta(r_grid[j%N])        if j==i+1
                   else -courant/(4.*r_grid[j+1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j]) if j==i-N-1
                   else courant/(4.*r_grid[j-1]**2.)*r_grid[j]**2.*alpha(r_grid[j])/a(r_grid[j])if j==i-N+1
                   else 0 for j in range(2*N)]

#print np.matrix(A)
#print '----------------------------------------------------------'
#print np.matrix(B)

def update_u(timestep):
    u = np.zeros(2*N)
    for i in range(2*N):
        if(i < N):
            u[i] = Phi[timestep, i]
        else:
            u[i] = Pi[timestep, i-N]
    return u

#update r and s using the solved-for vector ans, which is u at timestep [timestep]
def update_r_s(ans, timestep):
    for i in range(2*N):
        if(i < N):
            Phi[timestep, i] = ans[i]
        else:
            Pi[timestep, i-N] = ans[i]            
#            Pi[timestep, 0]   = Phi[timestep, 0]      #Sommerfeld conditions; note: works better to set s using r
#            Pi[timestep, N-1] = -Phi[timestep, N-1] #Sommerfeld conditions
    return 0

for n in range(1, timesteps):
    u = update_u(n-1)

    bb = B.dot(u)

    ans = np.linalg.solve(A, bb)

    update_r_s(ans, n)

#command0 = subprocess.Popen('mkdir temp_folder/'.split(), stdout=subprocess.PIPE)
#command0.wait()

for i in range(timesteps):
    if(i % 50 == 0):
	print 'saving frame ' + str(i) + ' out of ' + str(timesteps)
    pl.plot(r_grid, Pi[i, :])
    pl.ylim([-10., 10.])
    pl.savefig('temp_folder/%03d'%i + '.png')
    pl.clf()

#command1 = subprocess.Popen('ffmpeg -y -i temp_folder/%03d.png wave_equation.m4v'.split(), stdout=subprocess.PIPE)
#command1.wait()
#command2 = subprocess.Popen('rm -r temp_folder/'.split(), stdout=subprocess.PIPE)
#command2.wait()
