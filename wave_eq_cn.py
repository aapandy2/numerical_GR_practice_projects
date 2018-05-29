import numpy as np
import pylab as pl
import subprocess

#set parameters for simulation
N = 800
delta_x = 1./N
delta_t = 0.005
courant = delta_t / delta_x
timesteps = 250

#define grid
x_grid = np.linspace(0., 1., N)

#initialize arrays
phi = np.zeros((timesteps, N))
r = np.zeros((timesteps, N))
s = np.zeros((timesteps, N))

#define initial data
x_c = 0.5
width = 0.05
amp = 0.5
idsignum = 1. # -1=ingoing; 0=t-symmetric; +1=outgoing

r[0, :] = amp * np.exp(-(x_grid-x_c)**2./width**2.)
s[0, :] = idsignum * amp * np.exp(-(x_grid-x_c)**2./width**2.)

A = np.zeros((2*N, 2*N))
B = np.zeros((2*N, 2*N))

#define matrix A
for i in range(N):
    if(i == 0):
        A[i, :] = [1 if j==0 else (courant/4.) if j==N else (-courant/4.) if j==N+1 else 0 for j in range(2*N)]
    elif(i == N-1):
        A[i, :] = [1 if j==N-1 else (courant/4.) if j==2*N-2 else (-courant/4.) if j==2*N-1 else 0 for j in range(2*N)]
    else:
        A[i, :] = [1 if j==i else (courant/4.) if j==(N+i-1) else (-courant/4.) if j==(N+i+1) else 0 for j in range(2*N)]
    
for i in range(N, 2*N):
    if(i == N):
        A[i, :] = [1 if j==N else (courant/4.) if j==0 else (-courant/4.) if j==1 else 0 for j in range(2*N)]
    elif(i == 2*N-1):
        A[i, :] = [1 if j==2*N-1 else (courant/4.) if j==N-2 else (-courant/4.) if j==N-1 else 0 for j in range(2*N)]
    else:
        A[i, :] = [1 if j==i else (courant/4.) if j==(i-1-N) else (-courant/4.) if j==(i+1-N) else 0 for j in range(2*N)]


#define matrix B
for i in range(N):
    if(i == 0):
        B[i, :] = [1 if j==0 else (-courant/4.) if j==N else (courant/4.) if j==N+1 else 0 for j in range(2*N)]
    elif(i == N-1):
        B[i, :] = [1 if j==N-1 else (-courant/4.) if j==2*N-2 else (courant/4.) if j==2*N-1 else 0 for j in range(2*N)]
    else:
        B[i, :] = [1 if j==i else (-courant/4.) if j==(N+i-1) else (courant/4.) if j==(N+i+1) else 0 for j in range(2*N)]
    
for i in range(N, 2*N):
    if(i == N):
        B[i, :] = [1 if j==N else (-courant/4.) if j==0 else (courant/4.) if j==1 else 0 for j in range(2*N)]
    elif(i == 2*N-1):
        B[i, :] = [1 if j==2*N-1 else (-courant/4.) if j==N-2 else (courant/4.) if j==N-1 else 0 for j in range(2*N)]
    else:
        B[i, :] = [1 if j==i else (-courant/4.) if j==(i-1-N) else (courant/4.) if j==(i+1-N) else 0 for j in range(2*N)]


def update_u(timestep):
    u = np.zeros(2*N)
    for i in range(2*N):
        if(i < N):
            u[i] = r[timestep, i]
        else:
            u[i] = s[timestep, i-N]
    return u

#update r and s using the solved-for vector ans, which is u at timestep [timestep]
def update_r_s(ans, timestep):
    for i in range(2*N):
        if(i < N):
            r[timestep, i] = ans[i]
        else:
            s[timestep, i-N] = ans[i]            
            s[timestep, 0] = r[timestep, 0]      #Sommerfeld conditions; note: works better to set s using r
            s[timestep, N-1] = -r[timestep, N-1] #Sommerfeld conditions
    return 0

for n in range(1, timesteps):
    u = update_u(n-1)

    bb = B.dot(u)

    ans = np.linalg.solve(A, bb)

    update_r_s(ans, n)

command0 = subprocess.Popen('mkdir temp_folder/'.split(), stdout=subprocess.PIPE)
command0.wait()

for i in range(timesteps):
    if(i % 50 == 0):
	print 'saving frame ' + str(i) + ' out of ' + str(timesteps)
    pl.plot(x_grid, r[i, :])
    pl.ylim([-0.5, 0.5])
    pl.savefig('temp_folder/%03d'%i + '.png')
    pl.clf()

command1 = subprocess.Popen('ffmpeg -y -i temp_folder/%03d.png wave_equation.m4v'.split(), stdout=subprocess.PIPE)
command1.wait()
command2 = subprocess.Popen('rm -r temp_folder/'.split(), stdout=subprocess.PIPE)
command2.wait()
