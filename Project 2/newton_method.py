from sympy import *
import numpy as np

N = 10
delta_r =1./(N)
R = 100.
r_grid = np.linspace(delta_r, R, N)
m = 5.

A = np.zeros((N, N))
f_n = np.ones(N)

for i in range(N):
	if(i == 0): #r=0 BC: f'=0
		A[i, :] = [-1.      if j==i+2
			   else 4.  if j==i+1
			   else -3. if j==i
			   else 0  for j in range(N)]
	elif(i == N-1): #r=R BC: f=1
		A[i, :] = [1. if j==i
			   else 0 for j in range(N)]
	else:
		A[i, :] = [-2./delta_r**2. - m*f_n[j]**(m-1.) if j==i
			   else 1./delta_r**2. + 1./(delta_r * r_grid[j%N]) if j==i+1
			   else 1./delta_r**2. - 1./(delta_r * r_grid[j%N]) if j==i-1
			   else 0 for j in range(N)]

Ainv = np.linalg.inv(A)
eqszero = np.dot(Ainv, A)
eqszero[np.abs(eqszero) < 1e-13] = 0
print 'First verify that A^-1 A = 1'
print 'A^-1 A =\n', eqszero
print '-------------------------'

def residual(f_n):
	ans = np.zeros(N)
	for j in range(N):
		if(j == 0):
			ans[j] = (-f_n[j+2] + 4.*f_n[j+1] - 3.*f_n[j])
		elif(j == N-1):
			ans[j] = (f_n[j]-1.)
		else:
			ans[j] = ( (f_n[j+1] - 2.*f_n[j] + f_n[j-1])/delta_r**2. 
                              + (f_n[j+1]-f_n[j-1])/(delta_r*r_grid[j%N])
			      - (f_n[j])**m )

	return ans

print 'solve nonlinear system'
print 'n = 0', 'max(residual) =', np.amax(residual(f_n))
#num_iterations = 125
tolerance = 1e-5
res  = 1.
n    = 0
#for n in range(num_iterations):
while(np.abs(res) > tolerance):
	n   = n + 1
	f_n = f_n - np.dot(Ainv, residual(f_n))
	res = np.amax(np.abs(residual(f_n)))
	print 'n =', n, 'max(residual) =', res
