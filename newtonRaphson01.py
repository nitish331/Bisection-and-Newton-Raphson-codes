import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, a, b, x0=None, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method to find the root of a function.

    Parameters:
    f  : Function whose root is to be found
    df : Derivative of the function
    a, b : Interval [a, b]
    x0 : Initial guess (if None, chooses the best guess)
    tol : Tolerance for stopping criterion
    max_iter : Maximum iterations allowed

    Returns:
    root : Approximate root found
    errors : List of errors in each iteration
    """
    # Automatically choose x0 if not provided
    if x0 is None:
        x0 = a if abs(f(a)) < abs(f(b)) else b  # Choose closer value to zero

    x = x0
    errors = []
    print("\nIteration | x_value        | f(x)          | Error")
    print("-" * 50)

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-6:  # Prevent division by zero
            print("\nDerivative too small, stopping iteration.")
            return None, errors

        x_new = x - fx / dfx  # Newton-Raphson formula
        error = abs(x_new - x)
        errors.append(error)

        print(f"{i+1:^9} | {x:.8f} | {fx:.8f} | {error:.8f}")

        if error < tol:
            break
        
        x = x_new

    # Plot the function with root
    x_vals = np.linspace(a, b, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.scatter([x], [f(x)], color='red', zorder=3, label=f'Root ≈ {x:.6f}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Newton-Raphson Method: Function & Root")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot error reduction
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(errors)), errors, marker='o', linestyle='--', color='r')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Reduction in Newton-Raphson Method')
    plt.grid()
    plt.show()

    return x, errors

# Example Function
def f(x):
    return x**3 - 6*x**2 + 11*x - 6  # Has roots at x = 1, 2, 3

# Derivative of the function
def df(x):
    return 3*x**2 - 12*x + 11

def f1(x):
    return x**3 - 2*x + 2  # Has complex roots

def df1(x):
    return 3*x**2 - 2

def f2(x):
    return x**(1/3)

def df2(x):
    return (1/3) * x**(-2/3)  # Causes problems at x = 0


# Interval [a, b] and calling the method
a , b = 0 , 4
# a1, b1 = -10, 10  # Interval
# a2,b2 = -2,2
root, errors = newton_raphson(f, df, a, b)

if root is not None:
    print(f"\nApproximate root: {root:.6f} found in {len(errors)} iterations")
else:
    print("\nNo root found.")
