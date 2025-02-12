from clusterops.main import (
    execute_with_cpu_cores,
    force_execute_all_cores,
    get_optimal_core_count,
    get_sys_info,
    list_available_cpus,
    get_cpu_info,
)

print(list_available_cpus())

print(get_cpu_info())

print(get_optimal_core_count())


print(get_sys_info())


def sample_task(n: int) -> int:
    return n * n


# Pass the argument as a tuple instead of a plain integer
print(execute_with_cpu_cores(4, sample_task, (10,)))

print(force_execute_all_cores(sample_task, (10,)))
