import numpy as np
import matplotlib.pyplot as plt

# Bisection Method with Error Tracking
def bisection_method(f, a, b, tol=1e-6, max_iter=50):
    if f(a) * f(b) >= 0:
        print("❌ Invalid interval! f(a) and f(b) must have opposite signs.")
        return None

    print(f"🔎 Starting Bisection Method in [{a}, {b}] with tolerance {tol}\n")

    errors = []  # Store errors at each iteration
    iterations = []  # Track iteration number
    roots = []  # Store approximate root at each step

    for i in range(1, max_iter + 1):
        c = (a + b) / 2  # Midpoint
        f_c = f(c)
        error = abs(b - a) / 2  # Error estimate

        print(f"🌀 Iteration {i}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}, f(c) = {f_c:.6e}, Error = {error:.6e}")

        errors.append(error)  # Store error for plotting
        iterations.append(i)
        roots.append(c)  # Store root approximation

        if abs(f_c) < tol or error < tol:  # Convergence condition
            print(f"\n✅ Root found at x = {c:.6f} (tolerance met)")
            plot_error_graph(errors, iterations)
            return c

        if f(a) * f_c < 0:
            b = c  # Root is in [a, c]
        else:
            a = c  # Root is in [c, b]

    print("⚠️ Maximum iterations reached without finding a precise root.")
    plot_error_graph(errors, iterations)
    return None

# Function to plot error reduction graph
def plot_error_graph(errors, iterations):
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, marker='o', linestyle='--', color='red', label="Error Reduction")
    plt.xlabel("Iterations")
    plt.ylabel("Error (|b-a|/2)")
    plt.yscale("log")  # Log scale to show exponential decrease
    plt.title("Bisection Method: Error Reduction Over Iterations")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=1)
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Define the function
    def f(x):
        return x**2 + 5*x + 6  # Example function

    root = bisection_method(f, -5, -2.5, tol=1e-8, max_iter=50)
