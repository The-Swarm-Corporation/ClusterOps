# ClusterOps

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![Build Status](https://img.shields.io/github/actions/workflow/status/swarms-team/clusterops/test.yml?branch=master)](https://github.com/swarms-team/clusterops/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/swarms-team/clusterops)](https://codecov.io/gh/swarms-team/clusterops)


**ClusterOps** is an enterprise-grade Python library developed and maintained by the **Swarms Team** to help you manage and execute agents on specific **CPUs** and **GPUs** across clusters. This tool enables advanced CPU and GPU selection, dynamic task allocation, and resource monitoring, making it ideal for high-performance distributed computing environments.







---

## Features

- **CPU Execution**: Dynamically assign tasks to specific CPU cores.
- **GPU Execution**: Execute tasks on specific GPUs or dynamically select the best available GPU based on memory usage.
- **Fault Tolerance**: Built-in retry logic with exponential backoff for handling transient errors.
- **Resource Monitoring**: Real-time CPU and GPU resource monitoring (e.g., free memory on GPUs).
- **Logging**: Advanced logging configuration with customizable log levels (DEBUG, INFO, ERROR).
- **Scalability**: Supports multi-GPU task execution with Ray for distributed computation.

---


## Installation


```bash
pip3 install -U clusterops
```

---

## Quick Start

The following example demonstrates how to use ClusterOps to run tasks on specific CPUs and GPUs.

```python
from clusterops import (
   list_available_cpus,
   execute_with_cpu_cores,
   list_available_gpus,
   execute_on_gpu,
   execute_on_multiple_gpus,
)

# Example function to run
def sample_task(n: int) -> int:
    return n * n


# List CPUs and execute on CPU 0
cpus = list_available_cpus()
execute_on_cpu(0, sample_task, 10)

# List CPUs and execute using 4 CPU cores
execute_with_cpu_cores(4, sample_task, 10)

# List GPUs and execute on GPU 0
gpus = list_available_gpus()
execute_on_gpu(0, sample_task, 10)

# Execute across multiple GPUs
execute_on_multiple_gpus([0, 1], sample_task, 10)

```
<!-- 
## GPU Scheduler

The GPU Scheduler is a Ray Serve deployment that manages job execution with fault tolerance, job retries, and scaling. It uses the GPUJobExecutor to execute tasks on available GPUs.

See the [GPU Scheduler](/clusterops/gpu_scheduler.py) for more details.

```python
from clusterops import gpu_scheduler


async def sample_task(n: int) -> int:
    return n * n


print(gpu_scheduler(sample_task, priority=1, n=10))

``` -->


### Executing callables in parallel

This section demonstrates how to execute multiple callables in parallel using the `execute_parallel_optimized` function from the `clusterops.execute_callables_parallel` module.

```python
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

```


---

## Configuration

ClusterOps provides configuration through environment variables, making it adaptable for different environments (development, staging, production).

### Environment Variables

- **`LOG_LEVEL`**: Configures logging verbosity. Options: `DEBUG`, `INFO`, `ERROR`. Default is `INFO`.
- **`RETRY_COUNT`**: Number of times to retry a task in case of failure. Default is 3.
- **`RETRY_DELAY`**: Initial delay in seconds before retrying. Default is 1 second.

Set these variables in your environment:

```bash
export LOG_LEVEL=DEBUG
export RETRY_COUNT=5
export RETRY_DELAY=2.0
```

-----

## Docs

---

### `list_available_cpus() -> List[int]`

**Description:**  
Lists all available CPU cores on the system.

**Returns:**  
- `List[int]`: A list of available CPU core indices.

**Raises:**  
- `RuntimeError`: If no CPUs are found.

**Example Usage:**

```python
cpus = list_available_cpus()
print(f"Available CPUs: {cpus}")
```

---

### `select_best_gpu() -> Optional[int]`

**Description:**  
Selects the GPU with the most free memory.

**Returns:**  
- `Optional[int]`: The GPU ID of the best available GPU, or `None` if no GPUs are available.

**Example Usage:**

```python
best_gpu = select_best_gpu()
print(f"Best GPU ID: {best_gpu}")
```

---

### `execute_on_cpu(cpu_id: int, func: Callable, *args: Any, **kwargs: Any) -> Any`

**Description:**  
Executes a function on a specific CPU core.

**Arguments:**  
- `cpu_id (int)`: The CPU core to run the function on.
- `func (Callable)`: The function to be executed.
- `*args (Any)`: Positional arguments for the function.
- `**kwargs (Any)`: Keyword arguments for the function.

**Returns:**  
- `Any`: The result of the function execution.

**Raises:**  
- `ValueError`: If the CPU core specified is invalid.
- `RuntimeError`: If there is an error executing the function on the CPU.

**Example Usage:**

```python
result = execute_on_cpu(0, sample_task, 10)
print(f"Result: {result}")
```

---

### `retry_with_backoff(func: Callable, retries: int = RETRY_COUNT, delay: float = RETRY_DELAY, *args: Any, **kwargs: Any) -> Any`

**Description:**  
Retries a function with exponential backoff in case of failure.

**Arguments:**  
- `func (Callable)`: The function to execute with retries.
- `retries (int)`: Number of retries. Defaults to `RETRY_COUNT`.
- `delay (float)`: Delay between retries in seconds. Defaults to `RETRY_DELAY`.
- `*args (Any)`: Positional arguments for the function.
- `**kwargs (Any)`: Keyword arguments for the function.

**Returns:**  
- `Any`: The result of the function execution.

**Raises:**  
- `Exception`: After all retries fail.

**Example Usage:**

```python
result = retry_with_backoff(sample_task, retries=5, delay=2, n=10)
print(f"Result after retries: {result}")
```

---

### `execute_with_cpu_cores(core_count: int, func: Callable, *args: Any, **kwargs: Any) -> Any`

**Description:**  
Executes a function using a specified number of CPU cores.

**Arguments:**  
- `core_count (int)`: The number of CPU cores to run the function on.
- `func (Callable)`: The function to be executed.
- `*args (Any)`: Positional arguments for the function.
- `**kwargs (Any)`: Keyword arguments for the function.

**Returns:**  
- `Any`: The result of the function execution.

**Raises:**  
- `ValueError`: If the number of CPU cores specified is invalid or exceeds available cores.
- `RuntimeError`: If there is an error executing the function on the specified CPU cores.

**Example Usage:**

```python
result = execute_with_cpu_cores(4, sample_task, 10)
print(f"Result: {result}")
```

---

### `list_available_gpus() -> List[str]`

**Description:**  
Lists all available GPUs on the system.

**Returns:**  
- `List[str]`: A list of available GPU names.

**Raises:**  
- `RuntimeError`: If no GPUs are found.

**Example Usage:**

```python
gpus = list_available_gpus()
print(f"Available GPUs: {gpus}")
```

---

### `execute_on_gpu(gpu_id: int, func: Callable, *args: Any, **kwargs: Any) -> Any`

**Description:**  
Executes a function on a specific GPU using Ray.

**Arguments:**  
- `gpu_id (int)`: The GPU to run the function on.
- `func (Callable)`: The function to be executed.
- `*args (Any)`: Positional arguments for the function.
- `**kwargs (Any)`: Keyword arguments for the function.

**Returns:**  
- `Any`: The result of the function execution.

**Raises:**  
- `ValueError`: If the GPU index is invalid.
- `RuntimeError`: If there is an error executing the function on the GPU.

**Example Usage:**

```python
result = execute_on_gpu(0, sample_task, 10)
print(f"Result: {result}")
```

---

### `execute_on_multiple_gpus(gpu_ids: List[int], func: Callable, *args: Any, **kwargs: Any) -> List[Any]`

**Description:**  
Executes a function across multiple GPUs using Ray.

**Arguments:**  
- `gpu_ids (List[int])`: The list of GPU IDs to run the function on.
- `func (Callable)`: The function to be executed.
- `*args (Any)`: Positional arguments for the function.
- `**kwargs (Any)`: Keyword arguments for the function.

**Returns:**  
- `List[Any]`: A list of results from the execution on each GPU.

**Raises:**  
- `ValueError`: If any GPU index is invalid.
- `RuntimeError`: If there is an error executing the function on the GPUs.

**Example Usage:**

```python
result = execute_on_multiple_gpus([0, 1], sample_task, 10)
print(f"Results: {result}")
```

---

### `sample_task(n: int) -> int`

**Description:**  
A sample task function that returns the square of a number.

**Arguments:**  
- `n (int)`: Input number to be squared.

**Returns:**  
- `int`: The square of the input number.

**Example Usage:**

```python
result = sample_task(10)
print(f"Square of 10: {result}")
```

---

This documentation provides a clear description of the function's purpose, arguments, return values, potential exceptions, and examples of how to use them.


---

## Contributing

We welcome contributions to ClusterOps! If you'd like to contribute, please follow these steps:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/The-Swarm-Corporation/ClusterOps.git
   cd clusterops
   ```
3. **Create a feature branch** for your changes:
   ```bash
   git checkout -b feature/new-feature
   ```
4. **Install the development dependencies**:
   ```bash
   pip install -r dev-requirements.txt
   ```
5. **Make your changes**, and be sure to include tests.
6. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```
7. **Commit your changes** and push them to GitHub:
   ```bash
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```
8. **Submit a pull request** on GitHub, and we’ll review it as soon as possible.

### Reporting Issues

If you encounter any issues, please create a [GitHub issue](https://github.com/the-swarm-corporation/clusterops/issues).


## Further Documentation

[CLICK HERE](/DOCS.md)

---

## License

ClusterOps is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For any questions, feedback, or contributions, please contact the **Swarms Team** at [kye@swarms.world](mailto:kye@swarms.world).
