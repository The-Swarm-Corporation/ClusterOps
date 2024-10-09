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

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Executing on Specific CPUs](#executing-on-specific-cpus)
  - [Executing on Specific GPUs](#executing-on-specific-gpus)
  - [Retry Logic and Fault Tolerance](#retry-logic-and-fault-tolerance)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

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

---

## Usage

### Executing on Specific CPUs

You can execute a task on a specific number of CPU cores using the `execute_with_cpu_cores()` function. It automatically adjusts CPU affinity on systems where this feature is supported.

```python
from clusterops import execute_with_cpu_cores

def sample_task(n: int) -> int:
    return n * n

# Execute the task using 4 CPU cores
result = execute_with_cpu_cores(4, sample_task, 10)
print(f"Result on 4 CPU cores: {result}")
```

### Executing on Specific GPUs

ClusterOps supports running tasks on specific GPUs or dynamically selecting the best available GPU (based on free memory).

```python
from clusterops import execute_on_gpu

def sample_task(n: int) -> int:
    return n * n

# Execute the task on GPU with ID 1
result = execute_on_gpu(1, sample_task, 10)
print(f"Result on GPU 1: {result}")

# Execute the task on the best available GPU
result_best_gpu = execute_on_gpu(None, sample_task, 10)
print(f"Result on best available GPU: {result_best_gpu}")
```

### Retry Logic and Fault Tolerance

For production environments, ClusterOps includes retry logic with exponential backoff, which retries a task in case of failures.

```python
from clusterops import retry_with_backoff, execute_on_gpu

# Run task on the best GPU with retry logic
result = retry_with_backoff(execute_on_gpu, None, sample_task, 10)
print(f"Result with retry: {result}")
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

---

## License

ClusterOps is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For any questions, feedback, or contributions, please contact the **Swarms Team** at [kye@swarms.world](mailto:kye@swarms.world).
