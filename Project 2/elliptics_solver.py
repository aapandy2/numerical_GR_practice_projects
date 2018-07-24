import numpy as np

def jacobian(f_n, xi, Pi, r_grid):
	
	N = np.shape(r_grid)[0]
	delta_r = 1./N

	psi   = f_n[0:N]
        beta  = f_n[N:2*N]
        alpha = f_n[2*N:3*N]

	A   = np.zeros((3*N, 3*N))

	#going to write equations with soln vector (psi_0, ..., beta_0, ..., alpha_0, ...)
	for i in range(N):
		if(i == 0): #r=0 BC: psi'=0
			A[i, :] = [-1.      if j==i+2
				   else 4.  if j==i+1
				   else -3. if j==i
				   else 0  for j in range(3*N)]
		elif(i == N-1): #r=R BC: psi'+(psi-1)/r=0
			A[i, :] = [3./(2.*delta_r) + 1./r_grid[i%N] if j==i
				   else -4./(2.*delta_r)            if j==i-1
				   else 1./(2.*delta_r)             if j==i-2
				   else 0 for j in range(3*N)]
		else: #psi interal eqn Jacobian
			A[i, :] = [( -2./delta_r**2. + 5.*psi[i%N]**4./12. 
	                            * (1./alpha[i%N] * ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) 
	                               - beta[i%N]/r_grid[i%N]))**2. 
	                           + np.pi*(xi[i%N]**2. + Pi[i%N]**2.) ) if j==i
				   else 1./delta_r**2. + 1./(delta_r * r_grid[i%N]) if j==i+1
				   else 1./delta_r**2. - 1./(delta_r * r_grid[i%N]) if j==i-1
				   else psi[i%N]**5./(12.*alpha[i%N]**2.) * ( (beta[i%N-1]-beta[i%N+1])/(delta_r*r_grid[i%N])  
									     + 2.*beta[i%N]/(r_grid[i%N]**2.)) if j==i+N
				   else psi[i%N]**5./(12.*alpha[i%N]**2.) * ( (beta[i%N-1]-beta[i%N+1])/(2.*delta_r**2.)  
									     + 1.*beta[i%N]/(delta_r*r_grid[i%N])) if j==i+N-1
				   else psi[i%N]**5./(12.*alpha[i%N]**2.) * ( (beta[i%N+1]-beta[i%N-1])/(2.*delta_r**2.) 
									     - 1.*beta[i%N]/(delta_r*r_grid[i%N])) if j==i+N+1
				   else psi[i%N]**5./(12.) * ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r)
                                       - beta[i%N]/r_grid[i%N])**2. * (-2.*alpha[i%N]**(-3.)) if j==i+2*N
				   else 0 for j in range(3*N)]
	for i in range(N, 2*N):
		if(i == N): #r=0 BC: beta = 0
			A[i, :] = [1. if j==i else 0 for j in range(3*N)]
		elif(i == 2*N-1): #r=R BC: beta' + beta/r = 0
			A[i, :] = [3./(2.*delta_r) + 1./r_grid[i%N] if j==i
	                           else -4./(2.*delta_r)            if j==i-1
	                           else 1./(2.*delta_r)             if j==i-2
	                           else 0 for j in range(3*N)]
		else: #beta internal eqn Jacobian
			A[i, :] = [-2./delta_r**2. - 1./r_grid[i%N]*(2./r_grid[i%N] + 6.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r*psi[i%N]) - (alpha[i%N+1] - alpha[i%N-1])/(2.*delta_r*alpha[i%N]) ) if j==i
				   else 1./delta_r**2. - 1./(2.*delta_r)*(2./r_grid[i%N] + 6.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r*psi[i%N]) - (alpha[i%N+1] - alpha[i%N-1])/(2.*delta_r*alpha[i%N]) ) if j==i-1
				   else 1./delta_r**2. + 1./(2.*delta_r)*(2./r_grid[i%N] + 6.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r*psi[i%N]) - (alpha[i%N+1] - alpha[i%N-1])/(2.*delta_r*alpha[i%N]) ) if j==i+1
				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
					  *(-psi[i%N]**(-2.))*6.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r)
					 +(-2.*psi[i%N]**(-3.))*12.*np.pi*alpha[i%N]*xi[i%N]*Pi[i%N]
				        ) if j==i-N
				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
                                          *(-6.)/(2.*delta_r*psi[i%N]) 
					) if j==i-N-1
				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
                                          *(6.)/(2.*delta_r*psi[i%N])
					) if j==i-N+1
				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
                                          *-1.*(alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)*(-alpha[i%N]**(-2.))
					 + 12.*np.pi*xi[i%N]*Pi[i%N]/psi[i%N]**2.
					) if j==i+N
				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
                                          *-1.*(-1.)/(2.*delta_r*alpha[i%N])
					) if j==i+N-1
				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
                                          *-1.*(1.)/(2.*delta_r*alpha[i%N])
                                        ) if j==i+N+1
				   else 0 for j in range(3*N)]
	
	for i in range(2*N, 3*N):
		if(i == 2*N): #r=0 BC: alpha' = 0
			A[i, :] = [-1.      if j==i+2
	                           else 4.  if j==i+1
	                           else -3. if j==i 
	                           else 0  for j in range(3*N)]
		elif(i == 3*N-1): #r=R BC: alpha' + (alpha - 1)/r = 0
			A[i, :] = [3./(2.*delta_r) + 1./r_grid[i%N] if j==i
	                           else -4./(2.*delta_r)            if j==i-1
	                           else 1./(2.*delta_r)             if j==i-2
	                           else 0 for j in range(3*N)]
		else: #alpha internal eqn Jacobian
			A[i, :] = [-2./delta_r**2. - (-1.)*alpha[i%N]**(-2.)*(2.*psi[i%N]**4./3. * ( (beta[i%N+1] - beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N]  )**2.) - 8. * np.pi * Pi[i%N]**2. if j==i
				   else 1./delta_r**2. - 1./(2.*delta_r) * (2./r_grid[i%N] + 2.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r * psi[i%N]) ) if j==i-1
				   else 1./delta_r**2. + 1./(2.*delta_r) * (2./r_grid[i%N] + 2.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r * psi[i%N]) ) if j==i+1
				   else ( (alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)
					   *2.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r)*(-psi[i%N]**(-2.))
					 - 1./alpha[i%N]*(2.*(4.*psi[i%N]**3.)/3. 
					   * ( (beta[i%N+1] - beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N] )**2. )
					) if j==i-2*N
				   else ( (alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)*(2.*-1.)/(2.*delta_r*psi[i%N])
					) if j==i-2*N-1
				   else ( (alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)*(2.* 1.)/(2.*delta_r*psi[i%N])
					) if j==i-2*N+1
				   else ( -1./alpha[i%N]* (2.*psi[i%N]**4./3.)
					   *( (beta[i%N-1]-beta[i%N+1])/(delta_r*r_grid[i%N])
					     +(2.*beta[i%N])/r_grid[i%N]**2.  )
					) if j==i-N
				   else ( -1./alpha[i%N]* (2.*psi[i%N]**4./3.)
                                           *( (beta[i%N-1]-beta[i%N+1])/(2.*delta_r**2.)   
                                             +(beta[i%N])/(delta_r*r_grid[i%N])  )
					) if j==i-N-1
				   else ( -1./alpha[i%N]* (2.*psi[i%N]**4./3.)
                                           *( (beta[i%N+1]-beta[i%N-1])/(2.*delta_r**2.)        
                                             -(beta[i%N])/(delta_r*r_grid[i%N])  )
					) if j==i-N+1
				   else 0 for j in range(3*N)]

	return A


def inv_matrix(jacobian):
	ans = np.linalg.inv(jacobian)
	return ans

def residual(f_n, xi, Pi, r_grid):
	N = np.shape(r_grid)[0]

        delta_r = 1./N

	psi   = f_n[0:N]
        beta  = f_n[N:2*N]
        alpha = f_n[2*N:3*N]

	ans = np.zeros(3*N)
	for j in range(N):
		if(j == 0): #r=0 BC: psi'=0
			ans[j] = (-psi[j%N+2] + 4.*psi[j%N+1] - 3.*psi[j%N])
		elif(j == N-1): #r=R BC: psi'+(psi-1)/r=0
			ans[j] = (3.*psi[j%N] - 4.*psi[j%N-1] + psi[j%N-2])/(2.*delta_r) + (psi[j%N]-1)/r_grid[j%N]
		else: #internal psi equation
			ans[j] = ( (psi[j%N+1] - 2.*psi[j%N] + psi[j%N-1])/delta_r**2. + (psi[j%N+1]-psi[j%N-1])/(2.*delta_r)*(2./r_grid[j%N]) + psi[j%N]**5./12. * (1./alpha[j%N] * ( (beta[j%N+1]-beta[j%N-1])/(2.*delta_r) - beta[j%N]/r_grid[j%N]  )  )**2. + np.pi * psi[j%N] * (xi[j%N]**2. + Pi[j%N]**2.) )

	for j in range(N, 2*N):
		if(j == N): #r=0 BC: beta = 0
			ans[j] = beta[j%N]
		elif(j == 2*N-1): #r=R BC: beta' + beta/r = 0 
			ans[j] = (3.*beta[j%N] - 4.*beta[j%N-1] + beta[j%N-2])/(2.*delta_r) + beta[j%N]/r_grid[j%N]
		else: #internal beta equation
			ans[j] = ( (beta[j%N+1] - 2.*beta[j%N] + beta[j%N-1])/delta_r**2. + ( (beta[j%N+1]-beta[j%N-1])/(2.*delta_r) - beta[j%N]/r_grid[j%N]  ) * (2./r_grid[j%N] + 6.*(psi[j%N+1]-psi[j%N-1])/(2.*delta_r*psi[j%N]) - (alpha[j%N+1]-alpha[j%N-1])/(2.*delta_r*alpha[j%N])) + 12.*np.pi*alpha[j%N]*xi[j%N]*Pi[j%N]/psi[j%N]**2. )

	for j in range(2*N, 3*N):
		if(j == 2*N): #r=0 BC: alpha' = 0
			ans[j] = (-alpha[j%N+2] + 4.*alpha[j%N+1] - 3.*alpha[j%N]) 
		elif(j == 3*N-1): #r=R BC: alpha' + (alpha - 1)/r = 0
			ans[j] = (3.*alpha[j%N] - 4.*alpha[j%N-1] + alpha[j%N-2])/(2.*delta_r) + (alpha[j%N]-1)/r_grid[j%N]
		else: #internal alpha equation
			ans[j] = ( (alpha[j%N+1] - 2.*alpha[j%N] + alpha[j%N-1])/delta_r**2. + (alpha[j%N+1]-alpha[j%N-1])/(2.*delta_r) * (2./r_grid[j%N] + 2.*(psi[j%N+1] - psi[j%N-1])/(2.*delta_r*psi[j%N])) - alpha[j%N]**(-1.)*(2*psi[j%N]**4./3. * ( (beta[j%N+1]-beta[j%N-1])/(2.*delta_r) - beta[j%N]/r_grid[j%N] )**2. ) - 8.*np.pi*alpha[j%N]*Pi[j%N]**2. )

	return ans


def solve_elliptics(f_n, xi, Pi, r_grid, correction_weight=1.):

	A    = jacobian(f_n, xi, Pi, r_grid)
	Ainv = np.linalg.inv(A)
	eqszero = np.dot(Ainv, A)
#	eqszero[np.abs(eqszero) < 1e-13] = 0
#	print 'First verify that A^-1 A = 1'
#	print 'max(|A^-1 A - Identity|) =', np.amax(np.abs(eqszero - np.identity(np.shape(eqszero)[0])))
	print '-------------------------'

	N = np.shape(xi)[0]
	print 'initial:', np.mean(f_n[0:N]), np.mean(f_n[N:2*N]), np.mean(f_n[2*N:3*N])
	
	print 'solve nonlinear system w. correction_weight =', correction_weight
	res = residual(f_n, xi, Pi, r_grid) * correction_weight
	print 'iteration = 0', 'max(residual) =', np.amax(res/correction_weight)
	tolerance    = 1e-8
	iteration    = 0
	max_iter     = 1000
	
	while(np.abs(np.amax(res)) > tolerance and iteration < max_iter):
		iteration   = iteration + 1
		res = residual(f_n, xi, Pi, r_grid) * correction_weight
		jacobi = jacobian(f_n, xi, Pi, r_grid)
		inv_jacobian = inv_matrix(jacobi)
		f_n = f_n - np.dot(inv_jacobian, res)
		print 'iteration =', iteration, 'max(residual) =', np.amax(np.abs(res)/correction_weight)

	return f_n

#print solve_elliptics(f_n, xi, Pi, r_grid)
