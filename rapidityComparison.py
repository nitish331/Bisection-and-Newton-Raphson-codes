import numpy as np
import matplotlib.pyplot as plt

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Finds a root of f(x)=0 in [a, b] using the bisection method.
    Records the error (half the interval length) at each iteration.
    
    Returns:
        root       : Computed root.
        iterations : Number of iterations.
        errors     : List of error estimates per iteration.
        iterates   : List of midpoints (approximations) per iteration.
    """
    errors = []
    iterates = []
    # Ensure f(a) and f(b) have opposite signs.
    if f(a) * f(b) >= 0:
        raise ValueError("The function must have opposite signs at endpoints a and b.")
    
    for i in range(max_iter):
        c = (a + b) / 2.0
        iterates.append(c)
        error = (b - a) / 2.0
        errors.append(error)
        
        fc = f(c)
        if abs(fc) < tol or error < tol:
            return c, i+1, errors, iterates
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return (a + b) / 2.0, max_iter, errors, iterates

def newton_method(f, f_prime, x0, tol=1e-6, max_iter=100):
    """
    Finds a root of f(x)=0 using the Newton–Raphson method starting from x0.
    Uses the exact derivative f_prime(x) and records the error (|x_new - x|) at each iteration.
    
    Returns:
        root       : Computed root.
        iterations : Number of iterations.
        errors     : List of error estimates per iteration.
        iterates   : List of x values per iteration.
    """
    errors = []
    iterates = [x0]
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < tol:  # Prevent division by near zero.
            return None, i, errors, iterates
        x_new = x - fx / fpx
        error = abs(x_new - x)
        errors.append(error)
        iterates.append(x_new)
        if error < tol:
            return x_new, i+1, errors, iterates
        x = x_new
    return x, max_iter, errors, iterates

def compare_methods(f, f_prime, a, b, x0, tol=1e-6, max_iter=100):
    """
    Finds the root of f(x)=0 using both the bisection and Newton–Raphson methods,
    then compares:
      - The final computed roots and number of iterations.
      - The rapidity (convergence speed) by plotting error vs. iteration for each method.
      
    Parameters:
        f         : Function f(x).
        f_prime   : Its derivative f'(x) (for Newton–Raphson).
        a, b      : Interval endpoints for the bisection method.
        x0        : Initial guess for the Newton–Raphson method.
        tol       : Tolerance for convergence.
        max_iter  : Maximum number of iterations.
    
    Returns:
        A tuple (root_bisection, root_newton)
    """
    # Solve with the bisection method.
    root_bis, iter_bis, errors_bis, iterates_bis = bisection_method(f, a, b, tol, max_iter)
    
    # Solve with the Newton–Raphson method.
    root_newt, iter_newt, errors_newt, iterates_newt = newton_method(f, f_prime, x0, tol, max_iter)
    
    # Print results.
    print("Bisection Method:")
    print("  Root         =", root_bis)
    print("  Iterations   =", iter_bis)
    print("\nNewton–Raphson Method:")
    print("  Root         =", root_newt)
    print("  Iterations   =", iter_newt)
    
    # Plot rapidity (convergence) curves: error vs. iteration (using a semilog scale).
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, iter_bis+1), errors_bis, 'bo-', label='Bisection Error')
    plt.semilogy(range(1, iter_newt+1), errors_newt, 'ro-', label='Newton–Raphson Error')
    plt.xlabel('Iteration Number')
    plt.ylabel('Error (log scale)')
    plt.title('Rapidity (Convergence) Curves Comparison')
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()
    
    return root_bis, root_newt

if __name__ == '__main__':
    # Example function: f(x) = x^3 - x - 2, which has a root near 1.52138.
    f = lambda x: x**3 - x - 2
    f_prime = lambda x: 3*x**2 - 1
    
    # Define the bisection method interval and Newton–Raphson initial guess.
    a = 1
    b = 2
    x0 = 1.5
    
    # Compare the methods and plot the rapidity curves.
    compare_methods(f, f_prime, a, b, x0, tol=1e-6, max_iter=50)
