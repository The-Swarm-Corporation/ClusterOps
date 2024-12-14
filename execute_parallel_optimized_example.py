from clusterops.execute_callables_parallel import (
    execute_parallel_optimized,
)


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def power(a, b):
    return a**b


if __name__ == "__main__":
    # List of callables with their respective arguments
    callables_with_args = [
        (add, (2, 3)),
        (multiply, (5, 4)),
        (power, (2, 10)),
    ]

    # Execute the callables in parallel
    results = execute_parallel_optimized(callables_with_args)

    # Print the results
    print("Results:", results)
