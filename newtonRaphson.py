import numpy as np
import matplotlib.pyplot as plt

def newton_method(f, f_prime, x0, tol, max_iter=50):
    """
    Applies the Newton-Raphson method starting from initial guess x0.
    
    Parameters:
        f       : callable, the function whose root we are finding.
        f_prime : callable, the derivative of f.
        x0      : float, the initial guess.
        tol     : float, tolerance for convergence.
        max_iter: int, maximum number of iterations.
    
    Returns:
        A tuple (root, iterations, initial_guess, converged) where:
          - root is the computed root (if converged) or None.
          - iterations is the number of iterations taken.
          - initial_guess is the starting point x0.
          - converged is a boolean indicating if convergence was achieved.
    """
    x = x0
    for i in range(max_iter):
        derivative = f_prime(x)
        # Check if derivative is near zero to avoid division by zero
        if np.isclose(derivative, 0, atol=tol):
            return (None, i, x0, False)
        x_new = x - f(x) / derivative
        if abs(x_new - x) < tol:
            return (x_new, i+1, x0, True)
        x = x_new
    # If reached max_iter without convergence, mark as divergent.
    return (None, max_iter, x0, False)

def find_roots_newton(f, f_prime, interval_start, interval_end, tol, num_initial_guesses=100, max_iter=50):
    """
    Searches for roots of the function f in the given interval using the Newton-Raphson method.
    
    Parameters:
        f                : callable, the function whose roots are to be found.
        f_prime          : callable, the derivative of f.
        interval_start   : float, start of the interval.
        interval_end     : float, end of the interval.
        tol              : float, tolerance for convergence.
        num_initial_guesses: int, number of initial guesses to try in the interval.
        max_iter         : int, maximum iterations for Newton-Raphson.
    
    Returns:
        A tuple (converged_roots, divergence_info) where:
          - converged_roots is a list of tuples (initial_guess, root, iterations)
          - divergence_info is a list of tuples (initial_guess, iterations) for guesses that did not converge.
    """
    converged_roots = []
    divergence_info = []
    initial_guesses = np.linspace(interval_start, interval_end, num_initial_guesses)
    
    for x0 in initial_guesses:
        root, iterations, init, converged = newton_method(f, f_prime, x0, tol, max_iter)
        if converged:
            # Check for duplicates within tolerance
            if not any(np.isclose(root, r[1], atol=tol) for r in converged_roots):
                converged_roots.append((init, root, iterations))
        else:
            divergence_info.append((init, iterations))
            
    # Sort the converged roots by their value
    converged_roots = sorted(converged_roots, key=lambda t: t[1])
    
    return converged_roots, divergence_info

def solve_and_plot_newton(f, f_prime, interval_start, interval_end, tol, num_initial_guesses=100, max_iter=50):
    """
    Finds all roots of f using Newton-Raphson from multiple initial guesses, prints the results,
    and plots f(x) along with the converged roots.
    
    Parameters:
        f                : callable, the function.
        f_prime          : callable, the derivative of f.
        interval_start   : float, start of the interval.
        interval_end     : float, end of the interval.
        tol              : float, convergence tolerance.
        num_initial_guesses: int, number of initial guesses to use.
        max_iter         : int, maximum iterations for Newton-Raphson.
    """
    converged_roots, divergence_info = find_roots_newton(
        f, f_prime, interval_start, interval_end, tol, num_initial_guesses, max_iter
    )
    
    # Print the results
    print("\nNewton-Raphson Method Results:")
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
    
    # Plot the function
    x = np.linspace(interval_start, interval_end, 1000)
    y = np.vectorize(f)(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    
    # Mark converged roots
    for init_guess, root, iterations in converged_roots:
        plt.plot(root, f(root), 'ro', markersize=8)
        plt.annotate(f'{root:.4f}', xy=(root, f(root)), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color='red')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Newton-Raphson Method: Function Plot with Identified Roots')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Define the function and its derivative.
    # Example: f(x) = x^3 - x - 2, f'(x) = 3x^2 - 1.
    def f(x):
        return x**3 - x - 2

    def f_prime(x):
        return 3*x**2 - 1

    # Set the interval, tolerance, and number of initial guesses.
    interval_start = -3
    interval_end = 3
    tolerance = 1e-6
    num_initial_guesses = 100
    max_iter = 50

    # Call the Newton-Raphson solver to print results and plot.
    solve_and_plot_newton(f, f_prime, interval_start, interval_end, tolerance, num_initial_guesses, max_iter)
