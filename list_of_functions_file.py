import sympy as sp

x, y = sp.symbols('x y')

list_of_2D_functions = [
    (2*x + 1, "linear"),
    (x**2 - 1, "quadratic"),
    (x**3 - x, "cubic"),
    (x**4 - 2*x**2, "quartic"),
    (sp.sin(x), "sinusoidal"),
    (sp.cos(x), "cosinusoidal"),
    (sp.exp(x), "exponential"),
]

list_of_3D_functions = [
    (2*x + 3*y + 1, "linear"),
    (x*y + y**2 + x**2, "quadratic"),
    (x*y**2, "cubic"),
    (x**4 + y**4 + 2*x**2*y**2 - 2*x**2, "double-well"),
    (sp.exp(x) + sp.exp(y), "exponential"),
    (sp.sin(x) + sp.cos(y), "trigonometric"),
    (sp.log(x + 2) + sp.log(y + 2), "logarithmic"),
    (sp.exp(-(x/y)**2/2)/(sp.sqrt(2*sp.pi)*y), "gaussian"), # x = theta, y = sigma
]