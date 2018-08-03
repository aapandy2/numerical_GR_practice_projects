import numpy as np

def matter_residuals(xi, Pi, psi, beta, alpha, r_grid, n, delta_t, epsilon):
	N = np.shape(r_grid)[0]
	delta_r = 1./N

	xi_residual  = np.zeros(N)
	Pi_residual  = np.zeros(N)

	for j in range(N):
		if(j == 0):
			xi_residual[j]  = 0.5*(xi[n, j] + xi[n+1,j])
			Pi_residual[j]  = ( -Pi[n+1,j+2] + 4.*Pi[n+1,j+1] - 3.*Pi[n+1,j] 
				            -Pi[n,j+2]   + 4.*Pi[n,j+1]   - 3.*Pi[n,j] )
		elif(j == 1):
                        xi_residual[j] = ( (xi[n+1,j]-xi[n,j])/delta_t
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
                                               +(epsilon/(16.*delta_t))*(-xi[n,j]-4*xi[n,j-1]+6*xi[n,j]-4*xi[n,j+1]+xi[n,j+2])
                                             )
                        Pi_residual[j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
                                          -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
				          -0.5*(4./2.)*beta[n,j]*Pi[n,j]/psi[n,j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
                                          -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
                                          -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
                                          -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
                                          -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
                                          -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
                                          -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
                                          -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
				          -0.5*(4./2.)*beta[n+1,j]*Pi[n+1,j]/psi[n+1,j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                          -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
                                          -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
                                          -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
                                          -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
                                          -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                          -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
                                          +(epsilon/(16.*delta_t))*(Pi[n,j]-4*Pi[n,j-1]+6*Pi[n,j]-4*Pi[n,j+1]+Pi[n,j+2])
                                         )
		elif(j == N-1):
			xi_residual[j] = ( (xi[n+1,j]-xi[n,j])/delta_t 
                                        + 0.5 * ( (3.*xi[n+1,j] - 4.*xi[n+1,j-1] + xi[n+1, j-2])/(2.*delta_r) 
                                                 + xi[n+1,j]/r_grid[j]
				     	    +(3.*xi[n,j] - 4.*xi[n,j-1] + xi[n, j-2])/(2.*delta_r)  
				     	    + xi[n,j]/r_grid[j]) )
			Pi_residual[j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
                                        + 0.5 * ( (3.*Pi[n+1,j] - 4.*Pi[n+1,j-1] + Pi[n+1, j-2])/(2.*delta_r)
                                                 + Pi[n+1,j]/r_grid[j]
                                                 +(3.*Pi[n,j] - 4.*Pi[n,j-1] + Pi[n, j-2])/(2.*delta_r)
                                                 + Pi[n,j]/r_grid[j]) )
		elif(j == N-2):
                        xi_residual[j] = ( (xi[n+1,j]-xi[n,j])/delta_t
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
                        Pi_residual[j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
                                          -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
				          -0.5*(4./2.)*beta[n,j]*Pi[n,j]/psi[n,j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
                                          -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
                                          -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
                                          -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
                                          -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
                                          -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
                                          -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
                                          -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
				          -0.5*(4./2.)*beta[n+1,j]*Pi[n+1,j]/psi[n+1,j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                          -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
                                          -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
                                          -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
                                          -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
                                          -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                          -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
                                         )
		else:
			xi_residual[j] = ( (xi[n+1,j]-xi[n,j])/delta_t
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
				         	  +(epsilon/(16.*delta_t))*(xi[n,j-2]-4*xi[n,j-1]+6*xi[n,j]-4*xi[n,j+1]+xi[n,j+2])
				     	)
			Pi_residual[j] = ( (Pi[n+1,j]-Pi[n,j])/delta_t
					     -0.5*(2./3.)*Pi[n,j]*beta[n,j]/r_grid[j]
					     -0.5*(4./2.)*beta[n,j]*Pi[n,j]/psi[n,j]*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
					     -0.5*(1./3.)*Pi[n,j]*(beta[n,j+1]-beta[n,j-1])/(2.*delta_r)
					     -0.5*beta[n,j]*(Pi[n,j+1]-Pi[n,j-1])/(2.*delta_r)
					     -0.5*alpha[n,j]/psi[n,j]**2.*(xi[n,j+1]-xi[n,j-1])/(2.*delta_r)
					     -0.5*2.*alpha[n,j]*xi[n,j]/(psi[n,j]**2.*r_grid[j])
					     -0.5*2.*alpha[n,j]*xi[n,j]/psi[n,j]**3.*(psi[n,j+1]-psi[n,j-1])/(2.*delta_r)
					     -0.5*xi[n,j]/psi[n,j]**2.*(alpha[n,j+1]-alpha[n,j-1])/(2.*delta_r)
					     -0.5*(2./3.)*Pi[n+1,j]*beta[n+1,j]/r_grid[j]
					     -0.5*(4./2.)*beta[n+1,j]*Pi[n+1,j]/psi[n+1,j]*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                             -0.5*(1./3.)*Pi[n+1,j]*(beta[n+1,j+1]-beta[n+1,j-1])/(2.*delta_r)
                                             -0.5*beta[n+1,j]*(Pi[n+1,j+1]-Pi[n+1,j-1])/(2.*delta_r)
                                             -0.5*alpha[n+1,j]/psi[n+1,j]**2.*(xi[n+1,j+1]-xi[n+1,j-1])/(2.*delta_r)
                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/(psi[n+1,j]**2.*r_grid[j])
                                             -0.5*2.*alpha[n+1,j]*xi[n+1,j]/psi[n+1,j]**3.*(psi[n+1,j+1]-psi[n+1,j-1])/(2.*delta_r)
                                             -0.5*xi[n+1,j]/psi[n+1,j]**2.*(alpha[n+1,j+1]-alpha[n+1,j-1])/(2.*delta_r)
					     +(epsilon/(16.*delta_t))*(Pi[n,j-2]-4*Pi[n,j-1]+6*Pi[n,j]-4*Pi[n,j+1]+Pi[n,j+2])
					    )
	return [xi_residual, Pi_residual]

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
#				   else psi[i%N]**5./(12.*alpha[i%N]**2.) * ( (beta[i%N-1]-beta[i%N+1])/(delta_r*r_grid[i%N])  
#									     + 2.*beta[i%N]/(r_grid[i%N]**2.)) if j==i+N
#				   else psi[i%N]**5./(12.*alpha[i%N]**2.) * ( (beta[i%N-1]-beta[i%N+1])/(2.*delta_r**2.)  
#									     + 1.*beta[i%N]/(delta_r*r_grid[i%N])) if j==i+N-1
#				   else psi[i%N]**5./(12.*alpha[i%N]**2.) * ( (beta[i%N+1]-beta[i%N-1])/(2.*delta_r**2.) 
#									     - 1.*beta[i%N]/(delta_r*r_grid[i%N])) if j==i+N+1
#				   else psi[i%N]**5./(12.) * ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r)
#                                       - beta[i%N]/r_grid[i%N])**2. * (-2.*alpha[i%N]**(-3.)) if j==i+2*N
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
#				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
#					  *(-psi[i%N]**(-2.))*6.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r)
#					 +(-2.*psi[i%N]**(-3.))*12.*np.pi*alpha[i%N]*xi[i%N]*Pi[i%N]
#				        ) if j==i-N
#				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
#                                          *(-6.)/(2.*delta_r*psi[i%N]) 
#					) if j==i-N-1
#				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
#                                          *(6.)/(2.*delta_r*psi[i%N])
#					) if j==i-N+1
#				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
#                                          *-1.*(alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)*(-alpha[i%N]**(-2.))
#					 + 12.*np.pi*xi[i%N]*Pi[i%N]/psi[i%N]**2.
#					) if j==i+N
#				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
#                                          *-1.*(-1.)/(2.*delta_r*alpha[i%N])
#					) if j==i+N-1
#				   else ( ((beta[i%N+1]-beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N])
#                                          *-1.*(1.)/(2.*delta_r*alpha[i%N])
#                                        ) if j==i+N+1
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
#				   else ( (alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)
#					   *2.*(psi[i%N+1]-psi[i%N-1])/(2.*delta_r)*(-psi[i%N]**(-2.))
#					 - 1./alpha[i%N]*(2.*(4.*psi[i%N]**3.)/3. 
#					   * ( (beta[i%N+1] - beta[i%N-1])/(2.*delta_r) - beta[i%N]/r_grid[i%N] )**2. )
#					) if j==i-2*N
#				   else ( (alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)*(2.*-1.)/(2.*delta_r*psi[i%N])
#					) if j==i-2*N-1
#				   else ( (alpha[i%N+1]-alpha[i%N-1])/(2.*delta_r)*(2.* 1.)/(2.*delta_r*psi[i%N])
#					) if j==i-2*N+1
#				   else ( -1./alpha[i%N]* (2.*psi[i%N]**4./3.)
#					   *( (beta[i%N-1]-beta[i%N+1])/(delta_r*r_grid[i%N])
#					     +(2.*beta[i%N])/r_grid[i%N]**2.  )
#					) if j==i-N
#				   else ( -1./alpha[i%N]* (2.*psi[i%N]**4./3.)
#                                           *( (beta[i%N-1]-beta[i%N+1])/(2.*delta_r**2.)   
#                                             +(beta[i%N])/(delta_r*r_grid[i%N])  )
#					) if j==i-N-1
#				   else ( -1./alpha[i%N]* (2.*psi[i%N]**4./3.)
#                                           *( (beta[i%N+1]-beta[i%N-1])/(2.*delta_r**2.)        
#                                             -(beta[i%N])/(delta_r*r_grid[i%N])  )
#					) if j==i-N+1
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


def solve_elliptics_first_ts(f_n, xi, Pi, r_grid, correction_weight=1.):

	N = np.shape(r_grid)[0]
	print 'initial:', np.mean(f_n[0:N]), np.mean(f_n[N:2*N]), np.mean(f_n[2*N:3*N])
	
	print 'solve nonlinear system w. correction_weight =', correction_weight
	res = residual(f_n, xi, Pi, r_grid) * correction_weight
	print 'iteration = 0', 'max(residual) =', np.amax(res/correction_weight)
	tolerance    = 1e-8
	iteration    = 0
	max_iter     = 200
	
	while(np.abs(np.amax(res)) > tolerance and iteration < max_iter):
		iteration   = iteration + 1
		res = residual(f_n, xi, Pi, r_grid) * correction_weight
		jacobi = jacobian(f_n, xi, Pi, r_grid)
#		inv_jacobian = inv_matrix(jacobi)
#		f_n = f_n - np.dot(inv_jacobian, res)
		diffvector = np.linalg.solve(jacobi, -res)
		f_n = f_n + diffvector
		print 'iteration =', iteration, 'max(residual) =', np.amax(np.abs(res)/correction_weight)

	return f_n


def Newton_iteration(xi, Pi, psi, beta, alpha, r_grid, n, delta_t, epsilon, correction_weight=1, PSI_EVOL=False):
	N = np.shape(r_grid)[0]
	delta_r = 1./N

#	correction_weight = 1. #TODO: later remove this make argument to function

	f_n          = np.zeros(3*N)
	f_n[0:N]     = psi[n, :]
	f_n[N:2*N]   = beta[n, :]
	f_n[2*N:3*N] = alpha[n, :]

	elliptic_res = np.zeros(3*N)
	elliptic_res = residual(f_n, xi[n, :], Pi[n, :], r_grid) * correction_weight 

#	print 'Newton max(residual) =', np.amax(np.abs(elliptic_res)/correction_weight)

	jacobi = jacobian(f_n, xi[n, :], Pi[n, :], r_grid)
	diffvector = np.linalg.solve(jacobi, -elliptic_res)

	if(PSI_EVOL == True): #only update beta, alpha; keep psi unchanged
		f_n[N:3*N] = f_n[N:3*N] + diffvector[N:3*N]
	else: #update psi, beta, alpha
		f_n = f_n + diffvector #now contains updated psi, beta, alpha at timestep n

	return [f_n, elliptic_res/correction_weight]

#print solve_elliptics(f_n, xi, Pi, r_grid)
