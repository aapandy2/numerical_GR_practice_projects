from sympy import *
import numpy as np

#TODO: 1.) need to thoroughly test this for trial equations.
#TODO: 2.) the way r is being handled is not correct; it's being treated
#          as an analytic variable; should be okay for small grid spacing
#TODO: 3.) make all numbers have trailing decimal

#define symbols
r, t             = symbols('r t')
delta_r, delta_t = symbols('delta_r delta_t')

xi_dot = Function('xi_dot') (r)
Pi_dot = Function('Pi_dot') (r)
alpha  = Function('alpha') (r)
Pi     = Function(' Pi') (r)
psi    = Function('psi') (r)
beta   = Function('beta') (r)
xi     = Function('xi') (r)

xi_prime    = symbols('xi\'')
alpha_prime = symbols('alpha\'')
beta_prime  = symbols('beta\'')
Pi_prime    = Symbol('Pi\'')
psi_prime   = symbols('psi\'')


#functions to format equation
def remove_fns(expr):
	for fn in expr.atoms(Function):
		expr = expr.subs(fn, symbols(str(fn).replace('(r)', '')))

	return expr


def print_terms(expr):
	for mul in expr.atoms(Mul):
		print 
		if(mul == expr):
			print 'full expression', mul
		else:
			print mul
	return 0.


def remove_deriv(expr):
	for deriv in expr.atoms(Derivative):
		strderiv = str(deriv)
		noderiv = strderiv.rsplit(',')[0] #strip off the ', r)' part
		noderiv = noderiv.replace('Derivative(', '')
		primed = noderiv + '\''
		expr = expr.subs(deriv, symbols(primed))
	return expr


def apply_format(expr):
	expr = expr.expand()
	newexpr = remove_fns(expr)
	newexpr = remove_deriv(newexpr)
	return newexpr


def display_formatted(L, R):
	L_formatted = apply_format(L)
	R_formatted = apply_format(R)
	
	print L_formatted, '=', R_formatted, '\n'

	return [L_formatted, R_formatted]
	

#assumes formatted expr
def append_indices(expr_formatted):
	for atom in expr_formatted.atoms():
		if( ((str(atom).replace('-', '')).replace('.', '')).isdigit() != True and str(atom) != 'r' and str(atom) != 't'): #filter out integers
			expr_formatted = expr_formatted.subs(atom, symbols( str(atom) + '[n][j%N]' ) )
		elif( (str(atom).replace('-', '')).isdigit() != True and str(atom) == 'r' ):
			expr_formatted = expr_formatted.subs(atom, symbols('r_grid[j%N]'))

	return expr_formatted

#indexed_L = append_indices(L_formatted)
#indexed_R = append_indices(R_formatted)
#
#print indexed_L, '=', indexed_R, '\n'

#TODO: in first if statement, not sure this will behave correctly if there are multiple terms with time derivatives
def CN_time_deriv(expr_indexed):
	for atom in expr_indexed.atoms():
		if((str(atom).replace('-', '')).isdigit() != True):
			atomstr = str(atom)
	
			if(atomstr.find('_dot') > 0):
				atomstr = atomstr.replace('_dot', '')
				atomstr_adv = atomstr.replace('[n', '[n+1')
		
				#need to define dummy variables and substitute them
				atomstr_sum = '(x - y)/(2 * dt)'
		
				atom_sympy  = sympify(atomstr_sum)
				atom_sympy  = atom_sympy.subs('x', symbols(atomstr_adv))
				atom_sympy  = atom_sympy.subs('y', symbols(atomstr))
				atom_sympy  = atom_sympy.subs('dt', delta_t)
	#			print atom_sympy
				expr_indexed = expr_indexed.subs(atom, atom_sympy)
	return expr_indexed


def centered_derivs(expr_indexed):
	for atom in expr_indexed.atoms():
		if(atom != expr_indexed and (str(atom).replace('-', '')).isdigit() != True):
                	        atomstr = str(atom)
				if(atomstr.find('\'') > 0):
                               		atomstr = atomstr.replace('\'', '')
                               		atomstr_fwd = atomstr.replace('j%N]', 'j%N+1]')
					atomstr_bwd = atomstr.replace('j%N]', 'j%N-1]')
 
                              		#need to define dummy variables and substitute them
                               		atomstr_sum = '(x - y)/(2 * dr)'

                               		atom_sympy  = sympify(atomstr_sum)
                               		atom_sympy  = atom_sympy.subs('x', symbols(atomstr_fwd))
                               		atom_sympy  = atom_sympy.subs('y', symbols(atomstr_bwd))
					atom_sympy  = atom_sympy.subs('dr', delta_r)
                               		expr_indexed = expr_indexed.subs(atom, atom_sympy)
	return expr_indexed


def fwd_derivs(expr_indexed):
        for atom in expr_indexed.atoms():
                if(atom != expr_indexed and (str(atom).replace('-', '')).isdigit() != True):
                                atomstr = str(atom)
                                if(atomstr.find('\'') > 0):
                                        atomstr = atomstr.replace('\'', '')
					atomstr_cent = atomstr
                                        atomstr_fwd1 = atomstr.replace('j%N]', 'j%N+1]')
                                        atomstr_fwd2 = atomstr.replace('j%N]', 'j%N+2]')

                                        #need to define dummy variables and substitute them
                                        atomstr_sum = '(- 3 * x + 4 * y - z)/(2 * dr)'

                                        atom_sympy  = sympify(atomstr_sum)
                                        atom_sympy  = atom_sympy.subs('x', symbols(atomstr_cent))
                                        atom_sympy  = atom_sympy.subs('y', symbols(atomstr_fwd1))
					atom_sympy  = atom_sympy.subs('z', symbols(atomstr_fwd2))
                                        atom_sympy  = atom_sympy.subs('dr', delta_r)
                                        expr_indexed = expr_indexed.subs(atom, atom_sympy)
        return expr_indexed

def bwd_derivs(expr_indexed):
        for atom in expr_indexed.atoms():
                if(atom != expr_indexed and (str(atom).replace('-', '')).isdigit() != True):
                                atomstr = str(atom)
                                if(atomstr.find('\'') > 0):
                                        atomstr = atomstr.replace('\'', '')
                                        atomstr_cent = atomstr
                                        atomstr_bwd1 = atomstr.replace('j%N]', 'j%N-1]')
                                        atomstr_bwd2 = atomstr.replace('j%N]', 'j%N-2]')

                                        #need to define dummy variables and substitute them
                                        atomstr_sum = '(3 * x - 4 * y + z)/(2 * dr)'

                                        atom_sympy  = sympify(atomstr_sum)
                                        atom_sympy  = atom_sympy.subs('x', symbols(atomstr_cent))
                                        atom_sympy  = atom_sympy.subs('y', symbols(atomstr_bwd1))
                                        atom_sympy  = atom_sympy.subs('z', symbols(atomstr_bwd2))
                                        atom_sympy  = atom_sympy.subs('dr', delta_r)
                                        expr_indexed = expr_indexed.subs(atom, atom_sympy)
        return expr_indexed


#CN_L = CN_time_deriv(indexed_L)
#CN_R = centered_derivs(indexed_R)
#
#print CN_L, '!=', CN_R, '[RHS needs to be time-averaged]\n'

#needs indexed expression
def CN_advanced(expr_indexed):
	for atom in expr_indexed.atoms():
		#one of the atoms is the full expression rather than a term; also remove numbers
		if(atom != expr_indexed and (str(atom).replace('-', '')).isdigit() != True):
			atomstr = str(atom)
			atomstr = atomstr.replace('[n', '[n+1') #advanced timestep
			expr_indexed = expr_indexed.subs(atom, symbols( atomstr ) )
	return expr_indexed

#needs L and R after indexing and explicitly writing derivatives
def CN_full_expr(L_indexed_derivs, R_indexed_derivs):
	RHS_adv = CN_advanced(R_indexed_derivs)/2.

        L = L_indexed_derivs - RHS_adv
	R = R_indexed_derivs/2.

	for atom in L.atoms(Mul):
                if(atom != L and str(atom).find('[n]') > 0):
                        L = L - atom
                        R = R - atom


        return [L, R]


def CN_result(L, R, space_deriv='centered', print_mode='no_output'):
	print '\n' + space_deriv + '\n'	

	L_formatted = apply_format(L)
	R_formatted = apply_format(R)

	indexed_L = append_indices(L_formatted)
	indexed_R = append_indices(R_formatted)	

	CN_L = CN_time_deriv(indexed_L)

	if(space_deriv == 'centered'):
		CN_R = centered_derivs(indexed_R)
	elif(space_deriv == 'forward'):
		CN_R = fwd_derivs(indexed_R)
	elif(space_deriv == 'backward'):
		CN_R = bwd_derivs(indexed_R)

	CN_L = CN_L.expand()
	CN_R = CN_R.expand()

	L, R = CN_full_expr(CN_L, CN_R)	

#	L = L.simplify()
#	R = R.simplify()

	if(print_mode == 'normal'):
		print L, '=', R, '\n'

	elif(print_mode == 'piecewise'):
		for atom in L.atoms(Mul):
			print atom
		print '='
		for atom in R.atoms(Mul):
			print atom

	return [L, R]


def print_coefficients(L, R, symbols_list):
	for symbol in symbols_list:

		if(symbol.find('[n+1]') > 0):
			print '\n' + symbol + ' coeff:', L.coeff(symbols(symbol))
		elif(symbol.find('[n]') > 0):
			print '\n' + symbol + ' coeff:', R.coeff(symbols(symbol))

	return 0

def debug_mode(L, R):

        print L, '=', R, '\n'

        L_formatted, R_formatted = display_formatted(L, R)

        indexed_L = append_indices(L_formatted)
        indexed_R = append_indices(R_formatted)

        print indexed_L, '=', indexed_R, '\n'

	CN_L = CN_time_deriv(indexed_L)
	CN_R = centered_derivs(indexed_R)
	
	print CN_L, '!=', CN_R, '[RHS needs to be time-averaged]\n'

	#note: CN_result takes in original L, R
	CL_L_final, CN_R_final = CN_result(L, R)

	print CL_L_final, '\n=\n', CN_R_final, '\n'

        return 0.


#type in equation

#print '------------equation-1------------------------'
#
#
#R = 0
#L = xi_dot - diff( alpha*Pi/psi**2 + beta*xi, r)
#R = solve((L-R), xi_dot)[0]
#L = xi_dot
#
#print L, '=', R, '\n'
#
##display_formatted(L, R)
#
#L_final, R_final = CN_result(L, R, space_deriv='backward')
#
#print_coefficients(L_final, R_final, ['Pi[n+1][j%N+2]', 'Pi[n+1][j%N+1]', 'Pi[n+1][j%N]', 'Pi[n+1][j%N-1]', 'Pi[n+1][j%N-2]', 'xi[n+1][j%N+2]', 'xi[n+1][j%N+1]', 'xi[n+1][j%N]', 'xi[n+1][j%N-1]', 'xi[n+1][j%N-2]'])
#
#print '---------------------------------------------------------------'
#
#print_coefficients(L_final, R_final, ['Pi[n][j%N+2]', 'Pi[n][j%N+1]', 'Pi[n][j%N]', 'Pi[n][j%N-1]', 'Pi[n][j%N-2]', 'xi[n][j%N+2]', 'xi[n][j%N+1]', 'xi[n][j%N]', 'xi[n][j%N-1]', 'xi[n][j%N-2]'])

print '------------equation-2------------------------'

R = 0
L = Pi_dot - 1./(r**2. * psi**4.) * diff(r**2. * psi**4. * (beta*Pi + alpha*xi/psi**2.), r) + 2./3. * Pi * (beta.diff(r) + 2.*beta/r * (1. + 3. * r * psi.diff(r) / psi))
R = solve((L-R), Pi_dot)[0]
L = Pi_dot


#debug_mode(L, R)

L_final, R_final = CN_result(L, R, space_deriv='backward')

print_coefficients(L_final, R_final, ['Pi[n+1][j%N+2]', 'Pi[n+1][j%N+1]', 'Pi[n+1][j%N]', 'Pi[n+1][j%N-1]', 'Pi[n+1][j%N-2]', 'xi[n+1][j%N+2]', 'xi[n+1][j%N+1]', 'xi[n+1][j%N]', 'xi[n+1][j%N-1]', 'xi[n+1][j%N-2]'])

print '---------------------------------------------------------------'

print_coefficients(L_final, R_final, ['Pi[n][j%N+2]', 'Pi[n][j%N+1]', 'Pi[n][j%N]', 'Pi[n][j%N-1]', 'Pi[n][j%N-2]', 'xi[n][j%N+2]', 'xi[n][j%N+1]', 'xi[n][j%N]', 'xi[n][j%N-1]', 'xi[n][j%N-2]'])
