import numpy as np
import matplotlib.pyplot as plt

def bisection(f, a, b, tol):
    """
    Performs the bisection method on the interval [a, b] where f(a)*f(b) <= 0.
    
    Parameters:
        f   : callable, the function for which we are finding a root.
        a   : float, the start of the interval.
        b   : float, the end of the interval.
        tol : float, the tolerance for convergence.
        
    Returns:
        The approximate root within the interval [a, b] or None if no sign change is detected.
    """
    # If one of the endpoints is a root, return it immediately.
    if np.isclose(f(a), 0, atol=tol):
        return a
    if np.isclose(f(b), 0, atol=tol):
        return b
    
    # Ensure there is a sign change
    if f(a) * f(b) > 0:
        return None
    
    while (b - a) / 2 > tol:
        mid = (a + b) / 2
        if np.isclose(f(mid), 0, atol=tol):
            return mid
        elif f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    
    return (a + b) / 2

def find_roots(f, interval_start, interval_end, tol, num_subintervals=1000):
    """
    Finds all roots of function f in the interval [interval_start, interval_end] using the bisection method.
    
    Parameters:
        f                : callable, the function for which we want to find roots.
        interval_start   : float, the start of the overall interval.
        interval_end     : float, the end of the overall interval.
        tol              : float, the tolerance for the bisection method.
        num_subintervals : int, the number of subintervals to partition the interval.
        
    Returns:
        A sorted list of unique roots found.
    """
    roots = []
    x_vals = np.linspace(interval_start, interval_end, num_subintervals + 1)
    
    # Loop through each subinterval.
    for i in range(len(x_vals) - 1):
        a, b = x_vals[i], x_vals[i + 1]
        
        # Check if either endpoint is a root.
        if np.isclose(f(a), 0, atol=tol):
            candidate = a
            if not any(np.isclose(candidate, r, atol=tol) for r in roots):
                roots.append(candidate)
            continue  # No need to check the interval further.
        if np.isclose(f(b), 0, atol=tol):
            candidate = b
            if not any(np.isclose(candidate, r, atol=tol) for r in roots):
                roots.append(candidate)
            continue
        
        # If there is a sign change, use bisection.
        if f(a) * f(b) < 0:
            root = bisection(f, a, b, tol)
            if root is not None and not any(np.isclose(root, r, atol=tol) for r in roots):
                roots.append(root)
    
    return sorted(roots)

def solve_and_plot(f, interval_start, interval_end, tol, num_subintervals=1000):
    """
    Finds all roots of a function f within an interval, prints the results,
    and plots the function along with the roots.
    
    Parameters:
        f                : callable, the function to solve.
        interval_start   : float, the start of the interval.
        interval_end     : float, the end of the interval.
        tol              : float, the tolerance for the bisection method.
        num_subintervals : int, optional, the number of subintervals for root finding.
    """
    # Find roots using the bisection method.
    roots = find_roots(f, interval_start, interval_end, tol, num_subintervals)
    
    # Print the roots in a nicely formatted output.
    print("\nRoots found in the interval [{}, {}]:".format(interval_start, interval_end))
    if roots:
        for idx, root in enumerate(roots, start=1):
            print("  Root {}: {:.8f}".format(idx, root))
    else:
        print("  No roots found.")

    # Prepare data for plotting.
    x = np.linspace(interval_start, interval_end, 1000)
    y = np.vectorize(f)(x)

    # Plot the function.
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)

    # Mark each found root on the plot.
    for root in roots:
        plt.plot(root, f(root), 'ro', markersize=8)
        plt.annotate(f'{root:.8f}', xy=(root, f(root)), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color='red')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Plot with Identified Roots')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Define the function for which to find roots.
    # For example, f(x) = x**2 + 5*x + 6 = (x + 2)*(x + 3) = 0 has roots -2 and -3.
    def f(x):
        return x**2 + 5*x + 6

    # Set the interval and tolerance.
    interval_start = -10
    interval_end = 5
    tolerance = 1e-6

    # Call the function to find roots, print them, and plot the graph.
    solve_and_plot(f, interval_start, interval_end, tolerance)
