import numpy as np
import pylab as pl
import sys

#simulation parameters
N         = 256
courant   = 1.0
R         = 50.
dr        = R/(N-1.)
dt        = courant * dr
timesteps = 250
eps       = 0.3

correction_weight = 1.

GEOM_COUPLING = True
EVOL_PSI      = True

#define grid
r = np.linspace(0., R, N)

#initialize arrays
phi   = np.zeros((timesteps, N))
xi    = np.zeros((timesteps, N))
Pi    = np.zeros((timesteps, N))
psi_h = np.zeros((timesteps, N))

psi   = np.ones((timesteps, N))
beta  = np.zeros((timesteps, N))
alpha = np.ones((timesteps, N))

theta = np.zeros((timesteps, N))
ricci = np.zeros((timesteps, N))

#define matter at initial timestep
A     = 0.06
Delta = 5.
r0    = 20.

phi[0, :] = A * np.exp(-(r-r0)**2./Delta**2.)
xi[0, :]  = -2. * A * (r-r0)/Delta**2. * np.exp(-(r-r0)**2./Delta**2.)
Pi[0, :]  = xi[0, :]

#returns time average of array arr at n, j
def tAVG(arr, n, j):
    ans = (arr[n+1,j]+arr[n,j])/2.
    return ans

#populate CN advanced timestep matrix ADV
ADV = np.zeros((2*N, 2*N))
KWN = np.zeros((2*N, 2*N))

def populate_matrices(n):
    for i in range(N):
        if(i == 0):
            ADV[i, :] = [1./2. if j==i else 0 for j in range(2*N)]
        elif(i == N-1):
            ADV[i, :] = [1./dt + 3./(2.*2.*dr) + 1./(2.*r[i]) if j==i
                         else -4./(2.*2.*dr)                  if j==i-1
                         else 1./(2.*2.*dr)                   if j==i-2
                         else 0 for j in range(2*N)]
        else:
            ADV[i, :] = [( -(tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr) * 1./2.
                           + 1./dt ) if j==i
                         else -tAVG(beta, n, i) * 1./(2.*2.*dr) if j==i+1
                         else tAVG(beta, n, i) * 1./(2.*2.*dr) if j==i-1
                         else ( -(tAVG(alpha,n,i+1)-tAVG(alpha,n,i-1))/(2.*dr*tAVG(psi, n, i)**2.) * 1./2.
                              + 2.*tAVG(alpha,n,i)*(tAVG(psi,n,i+1)-tAVG(psi,n,i-1))/(2.*dr*tAVG(psi,n,i)**3.) * 1./2.) if j==i+N
                         else -tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr) if j==i+N+1
                         else  tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr) if j==i+N-1
                         else 0 for j in range(2*N)]
    for i in range(N, 2*N):
        if(i == N):
            ADV[i, :] = [     -3./(2.*2.*dr) if j==i
                         else  4./(2.*2.*dr) if j==i+1
                         else -1./(2.*2.*dr) if j==i+2
                         else 0 for j in range(2*N)]
        elif(i == 2*N-1):
            ADV[i, :] = [1./dt + 3./(2.*2.*dr) + 1./(2.*r[i%N]) if j==i
                         else -4./(2.*2.*dr)                  if j==i-1
                         else 1./(2.*2.*dr)                   if j==i-2
                         else 0 for j in range(2*N)]
        else:
            ADV[i, :] = [( -2.*tAVG(beta, n, i%N)/(3.*r[i%N])*(1./2.) 
                           - (tAVG(beta, n, i%N+1)-tAVG(beta, n, i%N-1))/(2.*dr * 3. * 2.) 
                           + 1./dt )  if j==i
                         else -tAVG(beta, n, i%N) * 1./(2.*2.*dr) if j==i+1 
                         else -tAVG(beta, n, i%N) *-1./(2.*2.*dr) if j==i-1
                         else ( -2.*tAVG(alpha, n, i%N)/(r[i%N]*tAVG(psi, n, i%N)**2.) * (1./2.) 
                                -(tAVG(alpha, n, i%N+1)-tAVG(alpha, n, i%N-1))/(2.*dr*tAVG(psi, n, i%N)**2.) * (1./2.)
                                -2.*tAVG(alpha, n, i%N)*(tAVG(psi,n,i%N+1)-tAVG(psi,n,i%N-1))/(2.*dr*tAVG(psi,n,i%N)**3.) * (1./2.)
                              ) if j==i-N
                         else -tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. * 1./(2.*2.*dr) if j==i-N+1
                         else -tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. *-1./(2.*2.*dr) if j==i-N-1
                         else 0 for j in range(2*N)]
    
    #populate CN known timestep matrix KWN
    for i in range(N):
        if(i == 0):
            KWN[i, :] = [-1./2. if j==i else 0 for j in range(2*N)]
        elif(i == 1):
            KWN[i, :] = [( (tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr) * 1./2.
                           + 1./dt -eps/(16.*dt)*(6.-1.)) if j==i
                         else tAVG(beta, n, i) * 1./(2.*2.*dr) -eps/(16.*dt)*(-4.) if j==i+1
                         else -tAVG(beta, n, i) * 1./(2.*2.*dr)-eps/(16.*dt)*(-4.) if j==i-1
                         else -eps/(16.*dt)*1. if j==i+2
                         else -( -(tAVG(alpha,n,i+1)-tAVG(alpha,n,i-1))/(2.*dr*tAVG(psi, n, i)**2.) * 1./2.
                              + 2.*tAVG(alpha,n,i)*(tAVG(psi,n,i+1)-tAVG(psi,n,i-1))/(2.*dr*tAVG(psi,n,i)**3.) * 1./2.) if j==i+N
                         else tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr)  if j==i+N+1
                         else -tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr) if j==i+N-1
                         else 0 for j in range(2*N)]
        elif(i == N-1):
            KWN[i, :] = [1./dt - 3./(2.*2.*dr) - 1./(2.*r[i]) if j==i
                         else 4./(2.*2.*dr)                   if j==i-1
                         else -1./(2.*2.*dr)                  if j==i-2
                         else 0 for j in range(2*N)]
        elif(i == N-2):
            KWN[i, :] = [( (tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr) * 1./2.
                           + 1./dt) if j==i
                         else tAVG(beta, n, i) * 1./(2.*2.*dr)  if j==i+1
                         else -tAVG(beta, n, i) * 1./(2.*2.*dr) if j==i-1
                         else -( -(tAVG(alpha,n,i+1)-tAVG(alpha,n,i-1))/(2.*dr*tAVG(psi, n, i)**2.) * 1./2.
                              + 2.*tAVG(alpha,n,i)*(tAVG(psi,n,i+1)-tAVG(psi,n,i-1))/(2.*dr*tAVG(psi,n,i)**3.) * 1./2.) if j==i+N
                         else tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr)  if j==i+N+1
                         else -tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr) if j==i+N-1
                         else 0 for j in range(2*N)]
        else:
            KWN[i, :] = [( (tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr) * 1./2.
                           + 1./dt -eps/(16.*dt)*6.) if j==i
                         else tAVG(beta, n, i) * 1./(2.*2.*dr) -eps/(16.*dt)*(-4.) if j==i+1
                         else -tAVG(beta, n, i) * 1./(2.*2.*dr)-eps/(16.*dt)*(-4.) if j==i-1
                         else -eps/(16.*dt)*1. if j==i+2
                         else -eps/(16.*dt)*1. if j==i-2
                         else -( -(tAVG(alpha,n,i+1)-tAVG(alpha,n,i-1))/(2.*dr*tAVG(psi, n, i)**2.) * 1./2.
                              + 2.*tAVG(alpha,n,i)*(tAVG(psi,n,i+1)-tAVG(psi,n,i-1))/(2.*dr*tAVG(psi,n,i)**3.) * 1./2.) if j==i+N 
                         else tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr)  if j==i+N+1
                         else -tAVG(alpha, n, i)/tAVG(psi, n, i)**2. * 1./(2.*2.*dr) if j==i+N-1
                         else 0 for j in range(2*N)]
    for i in range(N, 2*N):
        if(i == N):
            KWN[i, :] = [      3./(2.*2.*dr) if j==i
                         else -4./(2.*2.*dr) if j==i+1
                         else  1./(2.*2.*dr) if j==i+2
                         else 0 for j in range(2*N)]
        elif(i == N+1):
            KWN[i, :] = [( 2.*tAVG(beta, n, i%N)/(3.*r[i%N])*(1./2.)
                           + (tAVG(beta, n, i%N+1)-tAVG(beta, n, i%N-1))/(2.*dr * 3. * 2.)
                           + 1./dt -eps/(16.*dt)*(6.+1.))  if j==i
                         else tAVG(beta, n, i%N) * 1./(2.*2.*dr) -eps/(16.*dt)*(-4.) if j==i+1
                         else tAVG(beta, n, i%N) *-1./(2.*2.*dr) -eps/(16.*dt)*(-4.) if j==i-1
                         else -eps/(16.*dt)*1. if j==i+2
                         else ( 2.*tAVG(alpha, n, i%N)/(r[i%N]*tAVG(psi, n, i%N)**2.) * (1./2.)
                                +(tAVG(alpha, n, i%N+1)-tAVG(alpha, n, i%N-1))/(2.*dr*tAVG(psi, n, i%N)**2.) * (1./2.)
                                +2.*tAVG(alpha, n, i%N)*(tAVG(psi,n,i%N+1)-tAVG(psi,n,i%N-1))/(2.*dr*tAVG(psi,n,i%N)**3.) * (1./2.)
                              ) if j==i-N
                         else tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. * 1./(2.*2.*dr) if j==i-N+1
                         else tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. *-1./(2.*2.*dr) if j==i-N-1
                         else 0 for j in range(2*N)]
        elif(i == 2*N-1):
            KWN[i, :] = [1./dt - 3./(2.*2.*dr) - 1./(2.*r[i%N]) if j==i
                         else 4./(2.*2.*dr)                   if j==i-1
                         else -1./(2.*2.*dr)                  if j==i-2
                         else 0 for j in range(2*N)]
        elif(i == 2*N-2):
            KWN[i, :] = [( 2.*tAVG(beta, n, i%N)/(3.*r[i%N])*(1./2.)
                           + (tAVG(beta, n, i%N+1)-tAVG(beta, n, i%N-1))/(2.*dr * 3. * 2.)
                           + 1./dt )  if j==i
                         else tAVG(beta, n, i%N) * 1./(2.*2.*dr) if j==i+1
                         else tAVG(beta, n, i%N) *-1./(2.*2.*dr) if j==i-1
                         else ( 2.*tAVG(alpha, n, i%N)/(r[i%N]*tAVG(psi, n, i%N)**2.) * (1./2.)
                                +(tAVG(alpha, n, i%N+1)-tAVG(alpha, n, i%N-1))/(2.*dr*tAVG(psi, n, i%N)**2.) * (1./2.)
                                +2.*tAVG(alpha, n, i%N)*(tAVG(psi,n,i%N+1)-tAVG(psi,n,i%N-1))/(2.*dr*tAVG(psi,n,i%N)**3.) * (1./2.)
                              ) if j==i-N
                         else tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. * 1./(2.*2.*dr) if j==i-N+1
                         else tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. *-1./(2.*2.*dr) if j==i-N-1
                         else 0 for j in range(2*N)]
        else:
            KWN[i, :] = [( 2.*tAVG(beta, n, i%N)/(3.*r[i%N])*(1./2.)
                           + (tAVG(beta, n, i%N+1)-tAVG(beta, n, i%N-1))/(2.*dr * 3. * 2.)
                           + 1./dt -eps/(16.*dt)*6.)  if j==i
                         else tAVG(beta, n, i%N) * 1./(2.*2.*dr) -eps/(16.*dt)*(-4.) if j==i+1 
                         else tAVG(beta, n, i%N) *-1./(2.*2.*dr) -eps/(16.*dt)*(-4.) if j==i-1
                         else -eps/(16.*dt)*1. if j==i+2
                         else -eps/(16.*dt)*1. if j==i-2
                         else ( 2.*tAVG(alpha, n, i%N)/(r[i%N]*tAVG(psi, n, i%N)**2.) * (1./2.)
                                +(tAVG(alpha, n, i%N+1)-tAVG(alpha, n, i%N-1))/(2.*dr*tAVG(psi, n, i%N)**2.) * (1./2.)
                                +2.*tAVG(alpha, n, i%N)*(tAVG(psi,n,i%N+1)-tAVG(psi,n,i%N-1))/(2.*dr*tAVG(psi,n,i%N)**3.) * (1./2.)
                              ) if j==i-N
                         else tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. * 1./(2.*2.*dr) if j==i-N+1
                         else tAVG(alpha, n, i%N)/tAVG(psi, n, i%N)**2. *-1./(2.*2.*dr) if j==i-N-1
                         else 0 for j in range(2*N)]

    return 0

PADV = np.zeros((N, N))
PKWN = np.zeros((N, N))
psi_constant_vector = np.zeros(N)
for i in range(N):
    if(i == N-1):
        psi_constant_vector[i] = 1./r[i]
def populate_psi_matrix(n):
    for i in range(N):
        if(i == 0):
            PADV[i, :] = [    -3./(2.*2.*dr) if j==i
                         else  4./(2.*2.*dr) if j==i+1
                         else -1./(2.*2.*dr) if j==i+2
                         else 0 for j in range(N)]
        elif(i == N-1):
            PADV[i, :] = [3./(2.*2.*dr) + 1./(2.*r[i]) if j==i
                          else -4./(2.*2.*dr)     if j==i-1
                          else 1./(2.*2.*dr)      if j==i-2
                          else 0 for j in range(N)]
        else:
            PADV[i, :] = [( 1./dt 
                            - tAVG(beta, n, i)/(3.*r[i]) * 1./2.
                            -(tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr*6.) * 1./2.
                          ) if j==i
                          else -tAVG(beta, n, i)/(2. * 2. * dr) if j==i+1
                          else  tAVG(beta, n, i)/(2. * 2. * dr) if j==i-1
                          else 0 for j in range(N)]

    for i in range(N):
        if(i == 0):
            PKWN[i, :] = [    3./(2.*2.*dr)  if j==i
                         else -4./(2.*2.*dr) if j==i+1
                         else 1./(2.*2.*dr)  if j==i+2
                         else 0 for j in range(N)]
        elif(i == 1):
            PKWN[i, :] = [( 1./dt
                            + tAVG(beta, n, i)/(3.*r[i]) * 1./2.
                            +(tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr*6.) * 1./2.
                            -eps/(16.*dt)*(6.+1.)) if j==i
                          else tAVG(beta, n, i)/(2. * 2. * dr)-eps/(16.*dt)*(-4.)  if j==i+1
                          else -tAVG(beta, n, i)/(2. * 2. * dr)-eps/(16.*dt)*(-4.) if j==i-1
                          else -eps/(16.*dt)*(1.) if j==i-2
                          else -eps/(16.*dt)*(1.) if j==i+2
                          else 0 for j in range(N)]
        elif(i == N-2):
            PKWN[i, :] = [( 1./dt
                            + tAVG(beta, n, i)/(3.*r[i]) * 1./2.
                            +(tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr*6.) * 1./2.
                          ) if j==i
                          else tAVG(beta, n, i)/(2. * 2. * dr)  if j==i+1
                          else -tAVG(beta, n, i)/(2. * 2. * dr) if j==i-1
                          else 0 for j in range(N)]
        elif(i == N-1):
            PKWN[i, :] = [-3./(2.*2.*dr) - 1./(2.*r[i]) if j==i
                          else 4./(2.*2.*dr)       if j==i-1
                          else -1./(2.*2.*dr)      if j==i-2
                          else 0 for j in range(N)]
        else:
            PKWN[i, :] = [( 1./dt
                            + tAVG(beta, n, i)/(3.*r[i]) * 1./2.
                            +(tAVG(beta, n, i+1)-tAVG(beta, n, i-1))/(2.*dr*6.) * 1./2.
                            -eps/(16.*dt)*(6.)) if j==i
                          else tAVG(beta, n, i)/(2. * 2. * dr)-eps/(16.*dt)*(-4.)  if j==i+1
                          else -tAVG(beta, n, i)/(2. * 2. * dr)-eps/(16.*dt)*(-4.) if j==i-1
                          else -eps/(16.*dt)*(1.) if j==i-2
                          else -eps/(16.*dt)*(1.) if j==i+2
                          else 0 for j in range(N)]
    return 0

jacobian = np.zeros((3*N, 3*N))
#note: arrays passed in are 1D
def populate_jacobian(xi, Pi, psi, beta, alpha):
    for i in range(N):
        if(i == 0):
            jacobian[i, :] = [-3./(2.*dr) if j==i
                              else 4./(2.*dr) if j==i+1
                              else -1./(2.*dr) if j==i+2
                              else 0 for j in range(3*N)]
        elif(i == N-1):
            jacobian[i, :] = [3./(2.*dr) + 1./r[i] if j==i
                              else -4./(2.*dr) if j==i-1
                              else 1./(2.*dr) if j==i-2
                              else 0 for j in range(3*N)]
        else:
            jacobian[i, :] = [( -2./dr**2. 
                                +5.*psi[i]**4./12. * (1./alpha[i] * ( (beta[i+1]-beta[i-1])/(2.*dr) - beta[i]/r[i]  ) )**2.
                                +np.pi*(xi[i]**2.+Pi[i]**2.) ) if j==i
                              else 1./dr**2. + 1./(2.*dr)*(2./r[i]) if j==i+1
                              else 1./dr**2. - 1./(2.*dr)*(2./r[i]) if j==i-1
                              else 0 for j in range(3*N)]
    for i in range(N, 2*N):
        if(i == N):
            jacobian[i, :] = [1. if j==i else 0 for j in range(3*N)]
        elif(i == 2*N-1):
            jacobian[i, :] = [3./(2.*dr) + 1./r[i%N] if j==i
                              else -4./(2.*dr) if j==i-1
                              else 1./(2.*dr) if j==i-2
                              else 0 for j in range(3*N)]
        else:
            jacobian[i, :] = [( -2./dr**2.
                                -1./r[i%N] * ( 2./r[i%N] + 6./psi[i%N]*(psi[i%N+1]-psi[i%N-1])/(2.*dr) - (alpha[i%N+1]-alpha[i%N-1])/(2.*dr)/alpha[i%N] )
                              ) if j==i
                              else ( 1./dr**2.
                                    +1./(2.*dr)*( 2./r[i%N] + 6./psi[i%N]*(psi[i%N+1]-psi[i%N-1])/(2.*dr) - (alpha[i%N+1]-alpha[i%N-1])/(2.*dr)/alpha[i%N] )
                                   ) if j==i+1
                              else ( 1./dr**2.
                                    -1./(2.*dr)*( 2./r[i%N] + 6./psi[i%N]*(psi[i%N+1]-psi[i%N-1])/(2.*dr) - (alpha[i%N+1]-alpha[i%N-1])/(2.*dr)/alpha[i%N] )
                                   ) if j==i-1
                              else 0 for j in range(3*N)]
    for i in range(2*N, 3*N):
        if(i == 2*N):
            jacobian[i, :] = [-3./(2.*dr) if j==i
                              else 4./(2.*dr) if j==i+1
                              else -1./(2.*dr) if j==i+2
                              else 0 for j in range(3*N)]
        elif(i == 3*N-1):
            jacobian[i, :] = [3./(2.*dr) + 1./r[i%N] if j==i
                              else -4./(2.*dr) if j==i-1
                              else 1./(2.*dr) if j==i-2
                              else 0 for j in range(3*N)]
        else:
            jacobian[i, :] = [( -2./dr**2. 
                                +alpha[i%N]**(-2.)*( 2.*psi[i%N]**4./3. * ((beta[i%N+1]-beta[i%N-1])/(2.*dr) - beta[i%N]/r[i%N])**2. )
                                -8.*np.pi*Pi[i%N]**2. ) if j==i
                              else (1./dr**2.
                                    + 1./(2.*dr) * ( 2./r[i%N] + 2./psi[i%N] * (psi[i%N+1]-psi[i%N-1])/(2.*dr) )
                                   ) if j==i+1
                              else (1./dr**2.
                                    - 1./(2.*dr) * ( 2./r[i%N] + 2./psi[i%N] * (psi[i%N+1]-psi[i%N-1])/(2.*dr) )
                                   ) if j==i-1
                              else 0 for j in range(3*N)]

    return 0

#solve matrix equation ADV * advanced timestep soln vector = KWN dot prev timestep soln vector
def solve_system(n):
    prev_ts = np.append(xi[n, :], Pi[n, :])
    next_ts = np.linalg.solve(ADV, KWN.dot(prev_ts))
    return next_ts

#update matter variables for timestep n+1 using solution from solve_system(n)
def update_matter(next_ts, n):
    xi[n+1, :] = next_ts[0:N]
    Pi[n+1, :] = next_ts[N:2*N]
    return 0

def xi_residual(n):
    xi_res = np.ones(N)
    for j in range(N):
        if(j == 0):
            xi_res[j] = tAVG(xi, n, j)
        elif(j == 1):
            xi_res[j] = ( -tAVG(Pi, n, j) * (tAVG(alpha, n, j+1)-tAVG(alpha, n, j-1))/(2.*dr) / tAVG(psi, n, j)**2.
                          -tAVG(xi, n, j) * (tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr)
                          -tAVG(beta, n, j) * (tAVG(xi, n, j+1)-tAVG(xi, n, j-1))/(2.*dr)
                          +(xi[n+1, j] - xi[n, j])/dt
                          -tAVG(alpha, n, j)/tAVG(psi, n, j)**2. * (tAVG(Pi, n, j+1)-tAVG(Pi, n, j-1))/(2.*dr)
                          +2.*tAVG(alpha, n, j)*tAVG(Pi, n, j)/tAVG(psi, n, j)**3. * (tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                          +eps/(16.*dt)*(-xi[n,j]-4.*xi[n,j-1]+6.*xi[n,j]-4.*xi[n,j+1]+xi[n,j+2]) )

        elif(j == N-2):
            xi_res[j] = ( -tAVG(Pi, n, j) * (tAVG(alpha, n, j+1)-tAVG(alpha, n, j-1))/(2.*dr) / tAVG(psi, n, j)**2.
                          -tAVG(xi, n, j) * (tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr)
                          -tAVG(beta, n, j) * (tAVG(xi, n, j+1)-tAVG(xi, n, j-1))/(2.*dr)
                          +(xi[n+1, j] - xi[n, j])/dt
                          -tAVG(alpha, n, j)/tAVG(psi, n, j)**2. * (tAVG(Pi, n, j+1)-tAVG(Pi, n, j-1))/(2.*dr)
                          +2.*tAVG(alpha, n, j)*tAVG(Pi, n, j)/tAVG(psi, n, j)**3. * (tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                        )
        elif(j == N-1):
            xi_res[j] = (xi[n+1, j] - xi[n, j])/dt + (3.*tAVG(xi, n, j)-4.*tAVG(xi, n, j-1)+tAVG(xi, n, j-2))/(2.*dr) + tAVG(xi, n, j)/r[j]
        else:
            xi_res[j] = ( -tAVG(Pi, n, j) * (tAVG(alpha, n, j+1)-tAVG(alpha, n, j-1))/(2.*dr) / tAVG(psi, n, j)**2.
                          -tAVG(xi, n, j) * (tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr)
                          -tAVG(beta, n, j) * (tAVG(xi, n, j+1)-tAVG(xi, n, j-1))/(2.*dr)
                          +(xi[n+1, j] - xi[n, j])/dt
                          -tAVG(alpha, n, j)/tAVG(psi, n, j)**2. * (tAVG(Pi, n, j+1)-tAVG(Pi, n, j-1))/(2.*dr)
                          +2.*tAVG(alpha, n, j)*tAVG(Pi, n, j)/tAVG(psi, n, j)**3. * (tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                          +eps/(16.*dt)*(xi[n,j-2]-4.*xi[n,j-1]+6.*xi[n,j]-4.*xi[n,j+1]+xi[n,j+2]) )


    return xi_res

def Pi_residual(n):
    Pi_res = np.ones(N)
    for j in range(N):
        if(j == 0):
            Pi_res[j] = (-3.*tAVG(Pi, n, j)+4.*tAVG(Pi, n, j+1)-tAVG(Pi, n, j+2))/(2.*dr)
        elif(j == 1):
            Pi_res[j] = ( -2.*tAVG(beta, n, j)*tAVG(Pi, n, j)/(3.*r[j])
                          -2.*tAVG(alpha, n, j)*tAVG(xi, n, j)/(tAVG(psi, n, j)**2.*r[j])
                          -tAVG(xi, n, j)/tAVG(psi, n, j)**2. * (tAVG(alpha, n, j+1)-tAVG(alpha, n, j-1))/(2.*dr)
                          -tAVG(Pi, n, j)*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr*3.)
                          -tAVG(alpha, n, j)*(tAVG(xi, n, j+1)-tAVG(xi, n, j-1))/(2.*dr*tAVG(psi, n, j)**2.)
                          -tAVG(beta, n, j)*(tAVG(Pi, n, j+1)-tAVG(Pi, n, j-1))/(2.*dr)
                          +(Pi[n+1, j]-Pi[n, j])/dt
                          -2.*tAVG(alpha, n, j)*tAVG(xi, n, j)/tAVG(psi, n, j)**3.*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                          +eps/(16.*dt)*(Pi[n,j]-4.*Pi[n,j-1]+6.*Pi[n,j]-4.*Pi[n,j+1]+Pi[n,j+2]) )
        elif(j == N-2):
            Pi_res[j] = ( -2.*tAVG(beta, n, j)*tAVG(Pi, n, j)/(3.*r[j])
                          -2.*tAVG(alpha, n, j)*tAVG(xi, n, j)/(tAVG(psi, n, j)**2.*r[j])
                          -tAVG(xi, n, j)/tAVG(psi, n, j)**2. * (tAVG(alpha, n, j+1)-tAVG(alpha, n, j-1))/(2.*dr)
                          -tAVG(Pi, n, j)*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr*3.)
                          -tAVG(alpha, n, j)*(tAVG(xi, n, j+1)-tAVG(xi, n, j-1))/(2.*dr*tAVG(psi, n, j)**2.)
                          -tAVG(beta, n, j)*(tAVG(Pi, n, j+1)-tAVG(Pi, n, j-1))/(2.*dr)
                          +(Pi[n+1, j]-Pi[n, j])/dt
                          -2.*tAVG(alpha, n, j)*tAVG(xi, n, j)/tAVG(psi, n, j)**3.*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                        )
        elif(j == N-1):
            Pi_res[j] = (Pi[n+1, j] - Pi[n, j])/dt + (3.*tAVG(Pi, n, j)-4.*tAVG(Pi, n, j-1)+tAVG(Pi, n, j-2))/(2.*dr) + tAVG(Pi, n, j)/r[j]
        else:
            Pi_res[j] = ( -2.*tAVG(beta, n, j)*tAVG(Pi, n, j)/(3.*r[j])
                          -2.*tAVG(alpha, n, j)*tAVG(xi, n, j)/(tAVG(psi, n, j)**2.*r[j])
                          -tAVG(xi, n, j)/tAVG(psi, n, j)**2. * (tAVG(alpha, n, j+1)-tAVG(alpha, n, j-1))/(2.*dr)
                          -tAVG(Pi, n, j)*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr*3.)
                          -tAVG(alpha, n, j)*(tAVG(xi, n, j+1)-tAVG(xi, n, j-1))/(2.*dr*tAVG(psi, n, j)**2.)
                          -tAVG(beta, n, j)*(tAVG(Pi, n, j+1)-tAVG(Pi, n, j-1))/(2.*dr)
                          +(Pi[n+1, j]-Pi[n, j])/dt
                          -2.*tAVG(alpha, n, j)*tAVG(xi, n, j)/tAVG(psi, n, j)**3.*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                          +eps/(16.*dt)*(Pi[n,j-2]-4.*Pi[n,j-1]+6.*Pi[n,j]-4.*Pi[n,j+1]+Pi[n,j+2]) )
    return Pi_res

def psi_h_residual(n):
    psi_h_res = np.zeros(N)
    for j in range(N):
        if(j == 0):
            psi_h_res[j] = (-tAVG(psi, n, j+2)+4.*tAVG(psi, n, j+1)-3.*tAVG(psi, n, j))/(2.*dr) 
	elif(j == 1):
	    psi_h_res[j] = ( (psi[n+1, j] - psi[n, j])/dt - tAVG(beta, n, j)*tAVG(psi, n, j)/(3.*r[j])
                            -tAVG(beta, n, j)*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                            -tAVG(psi, n, j)*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr) * 1./6.
                            +eps/(16.*dt)*(psi[n,j]-4.*psi[n,j-1]+6.*psi[n,j]-4.*psi[n,j+1]+psi[n,j+2]) )
        elif(j == N-1):
            psi_h_res[j] = (3.*tAVG(psi, n, j)-4.*tAVG(psi, n, j-1)+tAVG(psi, n, j-2))/(2.*dr) + (tAVG(psi, n, j)-1.)/r[j]
	elif(j == N-2):
            psi_h_res[j] = ( (psi[n+1, j] - psi[n, j])/dt - tAVG(beta, n, j)*tAVG(psi, n, j)/(3.*r[j])
                            -tAVG(beta, n, j)*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                            -tAVG(psi, n, j)*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr) * 1./6.
                            )
        else:
            psi_h_res[j] = ( (psi[n+1, j] - psi[n, j])/dt - tAVG(beta, n, j)*tAVG(psi, n, j)/(3.*r[j])
                            -tAVG(beta, n, j)*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)
                            -tAVG(psi, n, j)*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr) * 1./6.
                            +eps/(16.*dt)*(psi[n,j-2]-4.*psi[n,j-1]+6.*psi[n,j]-4.*psi[n,j+1]+psi[n,j+2]) )

    return psi_h_res

#note: takes 1D arrays
def psi_residual(xi, Pi, psi, beta, alpha):
    psi_res = np.ones(N)
    for i in range(N):
        if(i == 0):
            psi_res[i] = (-psi[i+2]+4.*psi[i+1]-3.*psi[i])/(2.*dr)
        elif(i == N-1):
            psi_res[i] = (3.*psi[i]-4.*psi[i-1]+psi[i-2])/(2.*dr) + (psi[i]-1.)/r[i]
        else:
            psi_res[i] = ( (psi[i+1]-2.*psi[i]+psi[i-1])/dr**2. + (psi[i+1]-psi[i-1])/(2.*dr) * (2./r[i])
                          +psi[i]**5./12. * (1./alpha[i]*( (beta[i+1]-beta[i-1])/(2.*dr) - beta[i]/r[i] ))**2.
                          +np.pi*psi[i]*(xi[i]**2. + Pi[i]**2.)
                         )
    return psi_res

#note: takes 1D arrays
def beta_residual(xi, Pi, psi, beta, alpha):
    beta_res = np.ones(N)
    for i in range(N):
        if(i == 0):
            beta_res[i] = beta[i]
        elif(i == N-1):
            beta_res[i] = (3.*beta[i]-4.*beta[i-1]+beta[i-2])/(2.*dr) + beta[i]/r[i]
        else:
            beta_res[i] = ( (beta[i+1]-2.*beta[i]+beta[i-1])/dr**2.
                           +( (beta[i+1]-beta[i-1])/(2.*dr) - beta[i]/r[i] )
                             *(2./r[i] + 6./psi[i]*(psi[i+1]-psi[i-1])/(2.*dr) - (alpha[i+1]-alpha[i-1])/(2.*dr)/alpha[i] )
                           +12.*np.pi*alpha[i]*xi[i]*Pi[i]/psi[i]**2.
                          )
    return beta_res

#note: takes 1D arrays
def alpha_residual(xi, Pi, psi, beta, alpha):
    alpha_res = np.ones(N)
    for i in range(N):
        if(i == 0):
            alpha_res[i] = (-alpha[i+2]+4.*alpha[i+1]-3.*alpha[i])/(2.*dr)
        elif(i == N-1):
            alpha_res[i] = (3.*alpha[i]-4.*alpha[i-1]+alpha[i-2])/(2.*dr) + (alpha[i]-1.)/r[i]
        else:
            alpha_res[i] = ( (alpha[i+1]-2.*alpha[i]+alpha[i-1])/dr**2. 
                         +(alpha[i+1]-alpha[i-1])/(2.*dr) * ( 2./r[i] + 2./psi[i]*(psi[i+1]-psi[i-1])/(2.*dr) )
                         -alpha[i]**(-1.)*(2.*psi[i]**4./3. * ( (beta[i+1]-beta[i-1])/(2.*dr) - beta[i]/r[i] )**2. )
                         -8.*np.pi*alpha[i]*Pi[i]**2.
                        )

    return alpha_res

def solve_elliptics(n):
    #solve elliptics at nth timestep
    etol     = 1e-8
    res     = 10. #setting res to arbitrary value before while loop so it initiates
    counter = 0
    max_counter = 20
    
    while(res > etol and counter <= max_counter):
        counter += 1
        
        psi_res   =   psi_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        beta_res  =  beta_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        alpha_res = alpha_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
    
        resvector = np.append(psi_res, [beta_res, alpha_res])
    
        res = np.amax(np.abs(resvector))
    
        populate_jacobian(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
    
        diffvector = np.linalg.solve(jacobian, -resvector)
    
        psi[n, :]   += diffvector[0:N]     * correction_weight
        beta[n, :]  += diffvector[N:2*N]   * correction_weight
        alpha[n, :] += diffvector[2*N:3*N] * correction_weight
    
        psi_res   =   psi_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        beta_res  =  beta_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        alpha_res = alpha_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
    
        resvector = np.append(psi_res, [beta_res, alpha_res])
    
        res = np.amax(np.abs(resvector))
    
        print 'res after', counter, 'Newton iterations is:', res

    return counter

def solve_elliptics_nopsi(n):
    #solve elliptics at nth timestep
    etol     = 1e-8
    res     = 10. #setting res to arbitrary value before while loop so it initiates
    counter = 0
    max_counter = 20

    while(res > etol and counter <= max_counter):
        counter += 1

        psi_res   =   psi_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        beta_res  =  beta_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        alpha_res = alpha_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])

        resvector = np.append(psi_res, [beta_res, alpha_res])

        res = np.amax(np.abs(resvector))

        populate_jacobian(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])

        diffvector = np.linalg.solve(jacobian, -resvector)

	#don't update psi
        beta[n, :]  += diffvector[N:2*N]   * correction_weight
        alpha[n, :] += diffvector[2*N:3*N] * correction_weight

	#no need to update psi res using elliptic res either
        beta_res  =  beta_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
        alpha_res = alpha_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])

        resvector = np.append(psi_res, [beta_res, alpha_res])
	
	#only include beta, alpha elliptic eqn residuals
        res = np.amax(np.abs(resvector[N:3*N]))
        print 'res after', counter, 'Newton iterations is:', res

    return counter

if(GEOM_COUPLING == False):
    #solve system iteratively for (timesteps) iterations
    sys.stdout.flush() #testing if this flushes output to SLURM .out file

    for n in range(timesteps-1):
        print 'solved n =', n
        sys.stdout.flush() #testing if this flushes output to SLURM .out file

        populate_matrices(n)
        next_ts = solve_system(n)
        update_matter(next_ts, n)

else:
    if(EVOL_PSI == False):
        #solve elliptics at first timestep n = 0
        solve_elliptics(0)
    
        #solve system iteratively for (timesteps) iterations
        for n in range(timesteps-1):
            print '\n-----', n, '-----'
	    sys.stdout.flush() #testing if this flushes output to SLURM .out file   
 
            #solve system for n+1 starting with guess for n+1 elliptics
            psi[n+1, :]   = psi[n, :]
            beta[n+1, :]  = beta[n, :]
            alpha[n+1, :] = alpha[n, :]
    
            #refine n+1 elliptics iteratively until matter_res < mtol
            mres             = 10.  #setting to arbitrary value so loop initiates
            elliptic_counter = 10 #setting to arbitrary value so loop initiates
            mtol             = 1e-8 #depends strongly on resolution
            mcounter         = 0
            max_mcounter     = 20
    
            while(mres > mtol and mcounter <= max_mcounter):
#            while(elliptic_counter > 1 and mcounter <= max_mcounter):
                mcounter += 1
    
                #solve system
                populate_matrices(n)
                next_ts = solve_system(n)
                update_matter(next_ts, n)
    
                elliptic_counter = solve_elliptics(n+1)
    
                #compute matter residuals
                xi_res = xi_residual(n)
                Pi_res = Pi_residual(n)
                mres_vector = np.append(xi_res, Pi_res)
                mres = np.amax(np.abs(mres_vector))
    
                print 'matter iteration:', mcounter, 'matter residual:', mres, elliptic_counter 

            #compute outgoing null expansion theta to look for apparent horizons
            #1.) assuming no apparent horizon at r=0 (would contain nothing)
            #2.) assuming no apparent horizon at r=R (would contain entire grid)
            for j in range(N):
                ricci[n, j] = 8.*np.pi*(tAVG(xi, n, j)**2.-tAVG(Pi, n, j)**2.)/tAVG(psi, n, j)**4.

                if(j != 0 and j != N-1):
                    theta[n, j] = ( 2.*r[j]/(3.*tAVG(alpha, n, j)) * ( (tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr*r[j])
                                                                      - tAVG(beta, n, j)/r[j]**2. )
                                   +2./(r[j]*tAVG(psi, n, j)**4.) * ( tAVG(psi, n, j)**2.
                                                                     +2.*r[j]*tAVG(psi, n, j)
                                                                      *(tAVG(psi,n,j+1)-tAVG(psi,n,j-1))/(2.*dr) )
                                  )
                    if(j != 1 and theta[n, j-1] <= 0. and theta[n, j] > 0.):
                        print '\nAPPARENT HORIZON DETECTED at r = ', r[j-1]


    else:
        #solve elliptics at first timestep n = 0
        solve_elliptics(0)

        #copy over psi_h from psi
        psi_h[0, :] = psi[0, :]

        #solve system iteratively for timesteps n > 0
        for n in range(timesteps-1):
            print '\n-----', n, '-----'
	    sys.stdout.flush() #testing if this flushes output to SLURM .out file
            
            #solve system for n+1 starting with guess for n+1 elliptics
            psi[n+1, :]   = psi[n, :]
            beta[n+1, :]  = beta[n, :]
            alpha[n+1, :] = alpha[n, :]
            psi_h[n+1, :] = psi[n+1, :]

            #refine n+1 elliptics iteratively until elliptics are solved within one step
            elliptic_counter = 10 #abitrary value so loop starts
            mres             = 10.
            mcounter         = 0
            max_mcounter     = 20
            mtol             = 1e-8

            while(mres > mtol and mcounter <= max_mcounter):
#            while(elliptic_counter > 1 and mcounter <= max_mcounter):
                mcounter += 1

                #evolve psi_h and copy over into psi
                populate_psi_matrix(n)
                psi_h[n+1, :] = np.linalg.solve(PADV, PKWN.dot(psi_h[n, :]) + psi_constant_vector)
                psi[n+1, :]   = psi_h[n+1, :]

                #solve matter equations
                populate_matrices(n)
                next_ts = solve_system(n)
                update_matter(next_ts, n)

                elliptic_counter = solve_elliptics_nopsi(n+1)

                #compute hyperbolic residuals
                xi_res    = xi_residual(n)
                Pi_res    = Pi_residual(n)
                psi_h_res = psi_h_residual(n)
                mres_vector = np.append(xi_res, [Pi_res, psi_h_res])
                mres = np.amax(np.abs(mres_vector))

                print 'matter iteration:', mcounter, 'matter residual:', mres, elliptic_counter

            #compute outgoing null expansion theta to look for apparent horizons
            #1.) assuming no apparent horizon at r=0 (would contain nothing)
            #2.) assuming no apparent horizon at r=R (would contain entire grid)
            for j in range(N):
                ricci[n, j] = 8.*np.pi*(tAVG(xi, n, j)**2.-tAVG(Pi, n, j)**2.)/tAVG(psi, n, j)**4.

                if(j != 0 and j != N-1):
                    theta[n, j] = ( 2.*r[j]/(3.*tAVG(alpha, n, j)) * ( (tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr*r[j]) 
                                                                      - tAVG(beta, n, j)/r[j]**2. )
                                   +2./(r[j]*tAVG(psi, n, j)**4.) * ( tAVG(psi, n, j)**2. 
                                                                     +2.*r[j]*tAVG(psi, n, j)
                                                                      *(tAVG(psi,n,j+1)-tAVG(psi,n,j-1))/(2.*dr) )
                                  )
                    if(j != 1 and theta[n, j-1] <= 0. and theta[n, j] > 0.):
                        print '\nAPPARENT HORIZON DETECTED at r = ', r[j-1]

print '----- computing phi -----'
#compute scalar field phi
for n in range(timesteps-1):
    for j in range(N):
        phi[n+1, j] = phi[n, j] + dt*(tAVG(alpha, n, j)*tAVG(Pi, n, j)/tAVG(psi, n, j)**2. + tAVG(beta, n, j)*tAVG(xi, n, j))

print '----- computing mass aspect -----'
#compute mass aspect function
m = np.zeros((timesteps, N))
for n in range(timesteps-1):
    for j in range(N):
        if(j != 0 and j != N-1):
            m[n, j] = ( r[j]*tAVG(psi, n, j)**6./(18.*tAVG(alpha, n, j)**2.) * ( r[j]*(tAVG(beta, n, j+1)-tAVG(beta, n, j-1))/(2.*dr)
                                                                                 - tAVG(beta, n, j))**2.
                        -2.*r[j]**2. * (tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr)*(tAVG(psi, n, j)
                                                                                 +r[j]*(tAVG(psi, n, j+1)-tAVG(psi, n, j-1))/(2.*dr) )
                        )
        elif(j == 0):
            m[n, j] = m[n, j+1]
        else:
            m[n, j] = m[n, j-1]

print '----- computing final residuals -----'
xi_res    = np.zeros((timesteps, N))
Pi_res    = np.zeros((timesteps, N))
psi_h_res = np.zeros((timesteps, N))
psi_res   = np.zeros((timesteps, N))
beta_res  = np.zeros((timesteps, N))
alpha_res = np.zeros((timesteps, N))
#compute residuals
for n in range(timesteps-1):
    xi_res[n, :]    = xi_residual(n)
    Pi_res[n, :]    = Pi_residual(n)
    psi_h_res[n, :] = psi_h_residual(n) 
    psi_res[n, :]   =   psi_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
    beta_res[n, :]  =  beta_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])
    alpha_res[n, :] = alpha_residual(xi[n, :], Pi[n, :], psi[n, :], beta[n, :], alpha[n, :])


print '----- saving datafiles -----'
#save datafiles
np.savetxt('r.txt', r)
np.savetxt('xi.txt', xi)
np.savetxt('Pi.txt', Pi)
np.savetxt('psi.txt', psi)
np.savetxt('psi_h.txt', psi_h)
np.savetxt('beta.txt', beta)
np.savetxt('alpha.txt', alpha)

np.savetxt('phi.txt', phi)
np.savetxt('m.txt', m)
np.savetxt('theta.txt', theta)
np.savetxt('ricci.txt', ricci)

np.savetxt('xi_res.txt', xi_res)
np.savetxt('Pi_res.txt', Pi_res)
np.savetxt('psi_h_res.txt', psi_h_res)
np.savetxt('psi_res.txt', psi_res)
np.savetxt('beta_res.txt', beta_res)
np.savetxt('alpha_res.txt', alpha_res)

print '----- done -----'
