# Example usage of the decorator
from clusterops.cpu_exec_decorator import run_on_cpu


@run_on_cpu
def compute_heavy_task() -> None:
    # An example task that is CPU and memory intensive
    data = [i**2 for i in range(100000000)]
    print(sum(data))
    print("Task completed.")


compute_heavy_task()
