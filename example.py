from clusterops import (
    list_available_cpus,
    execute_with_cpu_cores,
    list_available_gpus,
    execute_on_gpu,
    execute_on_multiple_gpus,
    execute_on_cpu,
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
