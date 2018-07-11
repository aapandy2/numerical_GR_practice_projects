from sympy import *
import numpy as np

#define symbols
r, t   = symbols('r t')
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


#type in equation
R = 0
L = xi_dot - diff( alpha*Pi/psi**2 + beta*xi, r)
R = solve((L-R), xi_dot)[0]
L = xi_dot

#print L, '=', R


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


#L_formatted = apply_format(L)
#R_formatted = apply_format(R)
#
#print L_formatted, '=', R_formatted, '\n'
	

#assumes formatted expr
def append_indices(expr_formatted):
	for atom in expr_formatted.atoms():
		if( (str(atom).replace('-', '')).isdigit() != True): #filter out integers
			expr_formatted = expr_formatted.subs(atom, symbols( str(atom) + '[n][j]' ) )

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
				atomstr_sum = '(x - y)/2'
		
				atom_sympy  = sympify(atomstr_sum)
				atom_sympy  = atom_sympy.subs('x', symbols(atomstr_adv))
				atom_sympy  = atom_sympy.subs('y', symbols(atomstr))
	#			print atom_sympy
				expr_indexed = expr_indexed.subs(atom, atom_sympy)
	return expr_indexed


def centered_derivs(expr_indexed):
	for atom in expr_indexed.atoms():
		if(atom != expr_indexed and (str(atom).replace('-', '')).isdigit() != True):
                	        atomstr = str(atom)
				if(atomstr.find('\'') > 0):
                               		atomstr = atomstr.replace('\'', '')
                               		atomstr_fwd = atomstr.replace('j]', 'j+1]')
					atomstr_bwd = atomstr.replace('j]', 'j-1]')
 
                              		#need to define dummy variables and substitute them
                               		atomstr_sum = '(x - y)/2'

                               		atom_sympy  = sympify(atomstr_sum)
                               		atom_sympy  = atom_sympy.subs('x', symbols(atomstr_fwd))
                               		atom_sympy  = atom_sympy.subs('y', symbols(atomstr_bwd))
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
	for atom in L_indexed_derivs.atoms(Mul):
		if(atom != L_indexed_derivs and str(atom).find('[n]') > 0):
			L_indexed_derivs = L_indexed_derivs - atom
			R_indexed_derivs = R_indexed_derivs + atom

        L = L_indexed_derivs - CN_advanced(R_indexed_derivs)/2
	R = R_indexed_derivs

        return [L, R]


def display_CN_result(L, R):
	L_formatted = apply_format(L)
	R_formatted = apply_format(R)

	indexed_L = append_indices(L_formatted)
	indexed_R = append_indices(R_formatted)	

	CN_L = CN_time_deriv(indexed_L)
	CN_R = centered_derivs(indexed_R)

	L, R = CN_full_expr(CN_L, CN_R)	

	print L, '=', R, '\n'

	return 0


#L_final, R_final = CN_full_expr(CN_L, CN_R)
#
#print L_final, '=', R_final, '\n'

#display_CN_result(L, R)

R = 0
L = Pi_dot - 1/(r**2 * psi**4) * diff(r**2 * psi**4 * (beta*Pi + alpha*xi/psi**2), r) + 2/3 * Pi * (beta.diff(r) + 2*beta/r * (1 + 3 * r * psi.diff(r) / psi))
R = solve((L-R), Pi_dot)[0]
L = Pi_dot
#print L, '=', R
print apply_format(L), '=', apply_format(R), '\n'

display_CN_result(L, R)

