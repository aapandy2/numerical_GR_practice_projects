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
		#print str(deriv).find(',')
		strderiv = str(deriv)
		noderiv = strderiv.rsplit(',')[0] #strip off the ', r)' part
		noderiv = noderiv.replace('Derivative(', '')
#		print noderiv
		primed = noderiv + '\''
		expr = expr.subs(deriv, symbols(primed))
	return expr


def apply_format(expr):
	expr = expr.expand()
	newexpr = remove_fns(expr)
	newexpr = remove_deriv(newexpr)
	return newexpr


print apply_format(L), '=', apply_format(R), '\n'

L_formatted = apply_format(L)
R_formatted = apply_format(R)
	

#print L_formatted.atoms()
#print R_formatted.atoms(Mul)


#assumes formatted expr
def append_indices(expr_formatted):
	for atom in expr_formatted.atoms():
		if( (str(atom).replace('-', '')).isdigit() != True): #filter out integers
			expr_formatted = expr_formatted.subs(atom, symbols( str(atom) + '[n][j]' ) )
	return expr_formatted

print append_indices(R_formatted)

print append_indices(L_formatted)


#R = 0
#L = Pi_dot - 1/(r**2 * psi**4) * diff(r**2 * psi**4 * (beta*Pi + alpha*xi/psi**2), r) + 2/3 * Pi * (beta.diff(r) + 2*beta/r * (1 + 3 * r * psi.diff(r) / psi))
#R = solve((L-R), Pi_dot)[0]
#L = Pi_dot
##print L, '=', R
#print apply_format(L), '=', apply_format(R), '\n'

