import sympy as sp

x, y = sp.symbols('x y')

list_of_2D_functions = [
    (2*x + 1, "linear"),
    (x**2 - 1, "quadratic"),
    (x**3 - x, "cubic"),
    (x**4 - 2*x**2, "quartic"),
    (sp.sin(x), "sinusoidal"),
    (sp.cos(x), "cosinusoidal"),
]

list_of_3D_functions = [
    (2*x + 3*y + 1, "linear"),
    (x*y + y**2 + x**2, "quadratic"),
    (x*y**2, "cubic"),
    (sp.exp(x) + sp.exp(y), "exponential"),
    (sp.sin(x) + sp.cos(y), "trigonometric"),
    (sp.log(x + 2) + sp.log(y + 2), "logarithmic")
]