import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def create_numeric_functions_from_expr(f_expr, x):
    """
    Computes the exact derivative of f_expr with respect to x using sympy.diff,
    and returns numerical (lambdified) functions for both f and f'.
    
    Parameters:
        f_expr : sympy expression representing the function f(x).
        x      : sympy symbol.
        
    Returns:
        f       : a numerical function corresponding to f_expr.
        f_prime : a numerical function corresponding to the exact derivative of f_expr.
    """
    derivative_expr = sp.diff(f_expr, x)
    f = sp.lambdify(x, f_expr, "numpy")
    f_prime = sp.lambdify(x, derivative_expr, "numpy")
    return f, f_prime

def newton_method_exact(f, f_prime, x0, tol, max_iter=50):
    """
    Applies the Newton–Raphson method using the exact derivative (computed symbolically)
    starting from an initial guess x0.
    
    Parameters:
        f       : callable, the function f(x).
        f_prime : callable, the exact derivative f'(x).
        x0      : float, the initial guess.
        tol     : float, convergence tolerance.
        max_iter: int, maximum number of iterations.
        
    Returns:
        A tuple (root, iterations, initial_guess, converged) where:
          - root is the computed root (if converged) or None.
          - iterations is the number of iterations performed.
          - initial_guess is the starting value x0.
          - converged is a boolean indicating whether convergence was achieved.
    """
    x = x0
    for i in range(max_iter):
        deriv = f_prime(x)
        # Avoid division by zero or nearly zero derivative.
        if np.isclose(deriv, 0, atol=tol):
            return (None, i, x0, False)
        x_new = x - f(x) / deriv
        if abs(x_new - x) < tol:
            return (x_new, i+1, x0, True)
        x = x_new
    return (None, max_iter, x0, False)

def find_roots_newton_exact(f, f_prime, interval_start, interval_end, tol,
                            num_initial_guesses=100, max_iter=50):
    """
    Searches for roots of the function f in the given interval using the Newton–Raphson method
    with the exact derivative.
    
    Parameters:
        f                  : callable, the function f(x).
        f_prime            : callable, the derivative f'(x) computed exactly.
        interval_start     : float, start of the interval.
        interval_end       : float, end of the interval.
        tol                : float, convergence tolerance.
        num_initial_guesses: int, number of initial guesses to try in the interval.
        max_iter           : int, maximum iterations for each Newton–Raphson run.
        
    Returns:
        A tuple (converged_roots, divergence_info) where:
          - converged_roots is a list of tuples (initial_guess, root, iterations).
          - divergence_info is a list of tuples (initial_guess, iterations) for guesses that did not converge.
    """
    converged_roots = []
    divergence_info = []
    initial_guesses = np.linspace(interval_start, interval_end, num_initial_guesses)
    
    for x0 in initial_guesses:
        root, iterations, init, converged = newton_method_exact(f, f_prime, x0, tol, max_iter)
        if converged:
            # Avoid duplicates (within tolerance)
            if not any(np.isclose(root, r[1], atol=tol) for r in converged_roots):
                converged_roots.append((init, root, iterations))
        else:
            divergence_info.append((init, iterations))
    
    converged_roots = sorted(converged_roots, key=lambda t: t[1])
    return converged_roots, divergence_info

def solve_and_plot_newton_exact(f_expr, interval_start, interval_end, tol,
                                num_initial_guesses=100, max_iter=50):
    """
    Given a function f_expr (as a sympy expression in x), this function computes its exact derivative,
    finds all roots in the specified interval using the Newton–Raphson method, prints the results 
    (including divergence info), and plots f(x) along with the converged roots.
    
    Parameters:
        f_expr             : sympy expression representing f(x). (e.g., x**3 - x - 2)
        interval_start     : float, start of the interval.
        interval_end       : float, end of the interval.
        tol                : float, convergence tolerance.
        num_initial_guesses: int, number of initial guesses to try.
        max_iter           : int, maximum iterations for Newton–Raphson.
    """
    x = sp.symbols('x')
    # Create numerical functions for f and its exact derivative.
    f, f_prime = create_numeric_functions_from_expr(f_expr, x)
    
    # Find roots using Newton–Raphson with exact derivative.
    converged_roots, divergence_info = find_roots_newton_exact(
        f, f_prime, interval_start, interval_end, tol, num_initial_guesses, max_iter
    )
    
    # Print the results.
    print("\nNewton–Raphson Method (Exact Derivative) Results:")
    print("Function: f(x) =", f_expr)
    if converged_roots:
        print("\nConverged Roots:")
        for idx, (init_guess, root, iterations) in enumerate(converged_roots, start=1):
            print(f"  Root {idx}: {root:.8f} (Initial guess: {init_guess:.8f}, Iterations: {iterations})")
    else:
        print("  No converged roots found.")
    
    if divergence_info:
        print("\nDivergence Info (initial guesses that did not converge):")
        for idx, (init_guess, iterations) in enumerate(divergence_info, start=1):
            print(f"  Divergence {idx}: Initial guess {init_guess:.8f} did not converge in {iterations} iterations")
    
    # Plot the function.
    x_vals = np.linspace(interval_start, interval_end, 1000)
    y_vals = np.vectorize(f)(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    
    # Mark converged roots.
    for init_guess, root, iterations in converged_roots:
        plt.plot(root, f(root), 'ro', markersize=8)
        plt.annotate(f'{root:.4f}', xy=(root, f(root)), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color='red')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Newton–Raphson (Exact Derivative): f(x) = {sp.pretty(f_expr)}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Define the symbol and the function as a sympy expression.
    x = sp.symbols('x')
    # For example: f(x) = x**3 - x - 2 has a real root near x ≈ 1.52138.
    f_expr = x**3 - x - 2
    
    # Set the interval, tolerance, and other parameters.
    interval_start = -3
    interval_end = 3
    tolerance = 1e-6
    num_initial_guesses = 100
    max_iter = 50

    # Solve and plot using the Newton–Raphson method with exact differentiation.
    solve_and_plot_newton_exact(f_expr, interval_start, interval_end, tolerance,
                                num_initial_guesses, max_iter)
