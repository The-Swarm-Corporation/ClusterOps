import os
import sys
import time
from typing import Any, Callable, List

import GPUtil
import psutil
import ray
from loguru import logger
from ray.util.multiprocessing import (
    Pool,
)  # For distributed multi-node execution

# Configurable environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RETRY_COUNT = int(os.getenv("RETRY_COUNT", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))
CPU_THRESHOLD = int(
    os.getenv("CPU_THRESHOLD", 90)
)  # CPU usage threshold for alerts

# Configure Loguru logger for detailed logging
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL.upper(),
    format="{time} | {level} | {message}",
)


def monitor_resources():
    """
    Continuously monitors CPU and GPU resources and logs alerts when thresholds are crossed.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > CPU_THRESHOLD:
            logger.warning(
                f"CPU usage exceeds {CPU_THRESHOLD}%: Current usage {cpu_usage}%"
            )

        # Monitor GPU memory usage
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            memory_usage = 100 * (
                1 - gpu.memoryFree / gpu.memoryTotal
            )
            if (
                memory_usage > 90
            ):  # Example threshold for GPU memory usage
                logger.warning(
                    f"GPU {gpu.id} memory usage exceeds 90%: Current usage {memory_usage}%"
                )

        logger.info("Resource monitoring completed.")

    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        raise


def profile_execution(
    func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Profiles the execution of a task, collecting metrics like execution time and CPU/GPU usage.

    Args:
        func (Callable): The function to profile.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution along with the collected metrics.
    """
    start_time = time.time()

    # Get initial CPU and memory usage
    initial_cpu_usage = psutil.cpu_percent()
    gpus = GPUtil.getGPUs()
    initial_gpu_memory = [gpu.memoryFree for gpu in gpus]

    # Execute the function
    result = func(*args, **kwargs)

    # Metrics after execution
    execution_time = time.time() - start_time
    final_cpu_usage = psutil.cpu_percent()
    final_gpu_memory = [gpu.memoryFree for gpu in gpus]

    logger.info(f"Task execution time: {execution_time}s")
    logger.info(
        f"CPU usage change: {initial_cpu_usage}% -> {final_cpu_usage}%"
    )
    for idx, gpu in enumerate(gpus):
        logger.info(
            f"GPU {idx} memory usage change: {initial_gpu_memory[idx]} MB -> {final_gpu_memory[idx]} MB"
        )

    return result


def distributed_execute_on_gpus(
    gpu_ids: List[int], func: Callable, *args: Any, **kwargs: Any
) -> List[Any]:
    """
    Executes a callable across multiple GPUs and nodes using Ray's distributed task scheduling.

    Args:
        gpu_ids (List[int]): The list of GPU IDs across nodes to run the function on.
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        List[Any]: A list of results from the execution on each GPU.
    """
    try:
        logger.info(
            f"Executing function across distributed GPUs {gpu_ids}."
        )

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        @ray.remote(num_gpus=1)
        def task_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Distribute the task across nodes
        pool = Pool()
        result_futures = [
            pool.apply_async(task_wrapper.remote, args=(args, kwargs))
            for gpu_id in gpu_ids
        ]
        pool.close()
        pool.join()

        results = [future.get() for future in result_futures]
        logger.info(
            f"Distributed execution on GPUs {gpu_ids} completed."
        )
        return results

    except Exception as e:
        logger.error(
            f"Error during distributed execution on GPUs {gpu_ids}: {e}"
        )
        raise


# # Example function to run
# def sample_task(n: int) -> int:
#     return n * n


# # Monitor resources during execution
# monitor_resources()

# # Profile task execution and collect metrics
# profile_execution(sample_task, 10)

# # Execute distributed across multiple GPUs
# distributed_execute_on_gpus([0, 1], sample_task, 10)
