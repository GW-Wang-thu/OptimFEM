import numpy as np

import numpy as np




def solve_inverse():
    from sympy import symbols, Matrix
    a, b, c, d, e, f, g, h, i  = symbols('x2 y2 x2*y2 x3 y3 x3*y3 x4 y4 x4*y4')
    A = Matrix([[a, b, c],
                [d, e, f],
                [g, h, i]])
    A_inv = A.inv()
    print(A_inv)


if __name__ == '__main__':
    solve_inverse()