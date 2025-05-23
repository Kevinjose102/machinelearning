import sympy

J, w = sympy.symbols('J,w')

J = w ** 3

dJ_dw = sympy.diff(J) # returns the derivative
sympy.pprint(dJ_dw) # pretty print lol
print(dJ_dw.subs([(w,2)])) # substitues the value