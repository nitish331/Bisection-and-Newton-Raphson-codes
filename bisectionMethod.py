import numpy as np
import matplotlib.pyplot as plt

# Bisection Method Function
def bisection_method(f, a, b, tol=1e-4, max_iter=100):
    """
    Finds the root of a function f(x) using the Bisection Method.
    
    Parameters:
    - f: Function for which the root is to be found
    - a: Left bound of the interval
    - b: Right bound of the interval
    - tol: Tolerance (default 1e-4)
    - max_iter: Maximum iterations (default 100)

    Returns:
    - The estimated root if found, or None if no root is found
    """
    if f(a) * f(b) >= 0:
        print("Invalid interval: f(a) and f(b) must have opposite signs.")
        return None

    print(f"Starting Bisection Method for interval [{a}, {b}] with tolerance {tol}")

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint
        f_c = f(c)

        print(f"Iteration {i+1}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}, f(c) = {f_c:.6f}")

        if abs(f_c) < tol or abs(b - a) < tol:  # Convergence criteria
            print(f"Root found at x = {c:.6f} (tolerance met)")
            plot_function(f, a-3, b+3, root=c)  # Plot function with root
            return c

        if f(a) * f_c < 0:
            b = c  # Root lies in [a, c]
        else:
            a = c  # Root lies in [c, b]

    print("Maximum iterations reached without finding root.")
    return None

# Function to plot a function in a given interval
def plot_function(f, a, b, root=None):
    """
    Plots the function f(x) in the given interval [a, b].
    
    Parameters:
    - f: Function to be plotted
    - a: Left bound of the interval
    - b: Right bound of the interval
    - root: If provided, marks the root on the plot
    """
    x_vals = np.linspace(a, b , 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle="--")  # X-axis

    if root is not None:
        plt.scatter(root, f(root), color='red', label=f"Root ≈ {root:.6f}", zorder=3)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function Plot")
    plt.legend()
    plt.grid()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Define any function
    def f(x):
        return x**2 + 5*x + 6  # Function with two roots

    # Find root in given interval
    root = bisection_method(f, -5, -2.5, tol=1e-4, max_iter=50)

    # Plot function separately within the interval [-5, 6]
    # plot_function(f, -4, 0)
