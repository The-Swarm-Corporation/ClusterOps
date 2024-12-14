import os
import platform
import sys
import time
from typing import Any, Callable, List, Optional

import GPUtil
import psutil
import ray
from loguru import logger
from ray.util.multiprocessing import (
    Pool,
)  # For distributed multi-node execution
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

# Configurable environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RETRY_COUNT = int(os.getenv("RETRY_COUNT", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))

# Configure Loguru logger for detailed logging
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL.upper(),
    format="{time} | {level} | {message}",
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def list_available_cpus() -> List[int]:
    """
    Lists all available CPU cores.

    Returns:
        List[int]: A list of available CPU core indices.

    Raises:
        RuntimeError: If no CPUs are found.
    """
    try:
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is None or cpu_count <= 0:
            raise RuntimeError("No CPUs found.")
        logger.info(f"Available CPUs: {list(range(cpu_count))}")
        return list(range(cpu_count))
    except Exception as e:
        logger.error(f"Error listing CPUs: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def execute_with_cpu_cores(
    core_count: int, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Executes a callable using a specified number of currently unused CPU cores.

    Args:
        core_count (int): The number of CPU cores to run the function on.
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution.

    Raises:
        ValueError: If the number of CPU cores specified is invalid or exceeds available unused cores.
        RuntimeError: If there is an error executing the function on the specified CPU cores.
    """
    try:
        # Get all CPU cores
        all_cpus = list_available_cpus()

        # Find cores currently in use by checking CPU utilization
        cpu_percent_per_core = psutil.cpu_percent(
            interval=0.1, percpu=True
        )
        unused_cores = [
            cpu
            for cpu, usage in enumerate(cpu_percent_per_core)
            if usage
            < 10.0  # Consider cores with <10% usage as unused
        ]

        if not unused_cores:
            logger.warning(
                "No unused CPU cores found, falling back to all available cores"
            )
            unused_cores = all_cpus

        if core_count > len(unused_cores) or core_count <= 0:
            raise ValueError(
                f"Invalid core count: {core_count}. Available unused CPUs are {unused_cores}."
            )

        if platform.system() == "Darwin":  # macOS
            logger.warning(
                "CPU affinity is not supported on macOS. Skipping setting CPU affinity."
            )
        else:
            # Set CPU affinity to use the specified number of unused cores
            selected_cores = unused_cores[:core_count]
            logger.info(
                f"Setting CPU affinity to unused cores {selected_cores} and executing the function."
            )
            psutil.Process().cpu_affinity(selected_cores)

        result = func(*args, **kwargs)
        logger.info(
            f"Execution using {core_count} unused CPU cores completed."
        )
        return result
    except Exception as e:
        logger.error(
            f"Error executing with {core_count} CPU cores: {e}"
        )
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def execute_with_all_cpu_cores(
    func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Executes a callable using all currently unused CPU cores.

    Args:
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution.

    Raises:
        RuntimeError: If there is an error executing the function on the CPU cores.
    """
    try:
        # Get all CPU cores
        all_cpus = list_available_cpus()

        # Find cores currently in use by checking CPU utilization
        cpu_percent_per_core = psutil.cpu_percent(
            interval=0.1, percpu=True
        )
        unused_cores = [
            cpu
            for cpu, usage in enumerate(cpu_percent_per_core)
            if usage
            < 10.0  # Consider cores with <10% usage as unused
        ]

        if not unused_cores:
            logger.warning(
                "No unused CPU cores found, falling back to all available cores"
            )
            unused_cores = all_cpus

        logger.info(
            f"Found {len(unused_cores)} unused CPU cores out of {len(all_cpus)} total cores"
        )

        if platform.system() == "Darwin":  # macOS
            logger.warning(
                "CPU affinity is not supported on macOS. Skipping setting CPU affinity."
            )
        else:
            # Set CPU affinity to use all unused cores
            logger.info(
                f"Setting CPU affinity to unused cores {unused_cores} and executing the function."
            )
            psutil.Process().cpu_affinity(unused_cores)

        result = func(*args, **kwargs)
        logger.info(
            f"Execution using {len(unused_cores)} unused CPU cores completed."
        )
        return result
    except Exception as e:
        logger.error(f"Error executing with CPU cores: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def select_best_gpu() -> Optional[int]:
    """
    Selects the GPU with the most free memory.

    Returns:
        Optional[int]: The GPU ID of the best available GPU, or None if no GPUs are available.
    """
    try:
        gpus = list_available_gpus()
        best_gpu = max(gpus, key=lambda gpu: gpu["memoryFree"])
        logger.info(
            f"Selected GPU {best_gpu['id']} with {best_gpu['memoryFree']} MB free memory."
        )
        return best_gpu["id"]
    except Exception as e:
        logger.error(f"Error selecting best GPU: {e}")
        return None


def get_cpu_info():
    """
    Detects available CPU cores using multiple methods for reliability.
    Returns tuple of (physical_cores, logical_cores, available_cores)
    """
    try:
        # Physical cores (excluding hyperthreading)
        physical = psutil.cpu_count(logical=False) or 1

        # Logical cores (including hyperthreading)
        logical = psutil.cpu_count(logical=True) or physical

        # Currently available cores (accounting for system restrictions)
        try:
            available = len(psutil.Process().cpu_affinity())
        except AttributeError:  # For systems without affinity support
            available = logical

        # Sanity checks and adjustments
        physical = max(1, physical)
        logical = max(physical, logical)
        available = max(1, min(available, logical))

        return physical, logical, available

    except Exception as e:
        logger.error(f"Error detecting CPU cores: {e}")
        return 1, 1, 1  # Fallback to minimum safe values


def get_optimal_core_count(requested_cores: int = None) -> int:
    """Returns optimal number of cores to use based on system capabilities"""
    physical, logical, available = get_cpu_info()

    if requested_cores is None:
        # Use 75% of available cores by default, minimum of 1
        return max(1, int(available * 0.75))

    return max(1, min(requested_cores, available))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def retry_with_backoff(
    func: Callable,
    retries: int = RETRY_COUNT,
    delay: float = RETRY_DELAY,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Retries a callable function with exponential backoff in case of failure.

    Args:
        func (Callable): The function to execute with retries.
        retries (int): Number of retries. Defaults to RETRY_COUNT from env.
        delay (float): Delay between retries in seconds. Defaults to RETRY_DELAY from env.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution.

    Raises:
        Exception: After all retries fail.
    """
    attempt = 0
    while attempt <= retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error on attempt {attempt + 1}/{retries}: {e}"
            )
            if attempt == retries:
                logger.error(f"All {retries} retries failed.")
                raise
            attempt += 1
            time.sleep(delay * (2**attempt))  # Exponential backoff


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def execute_on_cpu(
    core_count: int, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Executes a callable using a specified number of CPU cores.

    Args:
        core_count (int): The number of CPU cores to run the function on.
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution.

    Raises:
        ValueError: If the number of CPU cores specified is invalid or exceeds available cores.
        RuntimeError: If there is an error executing the function on the specified CPU cores.
    """
    try:
        available_cpus = list_available_cpus()
        if core_count > len(available_cpus) or core_count <= 0:
            raise ValueError(
                f"Invalid core count: {core_count}. Available CPUs are {available_cpus}."
            )

        if platform.system() == "Darwin":  # macOS
            logger.warning(
                "CPU affinity is not supported on macOS. Skipping setting CPU affinity."
            )
        else:
            # Set CPU affinity to use the specified number of cores on non-macOS systems
            selected_cores = available_cpus[:core_count]
            logger.info(
                f"Setting CPU affinity to cores {selected_cores} and executing the function."
            )
            psutil.Process().cpu_affinity(selected_cores)

        result = func(*args, **kwargs)
        logger.info(
            f"Execution using {core_count} CPU cores completed."
        )
        return result
    except Exception as e:
        logger.error(
            f"Error executing with {core_count} CPU cores: {e}"
        )
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def list_available_gpus() -> List[str]:
    """
    Lists all available GPUs.

    Returns:
        List[str]: A list of available GPU names.

    Raises:
        RuntimeError: If no GPUs are found.
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            raise RuntimeError("No GPUs found.")
        gpu_names = [gpu.name for gpu in gpus]
        logger.info(f"Available GPUs: {gpu_names}")
        return gpu_names
    except Exception as e:
        logger.error(f"Error listing GPUs: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def execute_on_gpu(
    gpu_id: int, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Executes a callable on a specific GPU using Ray.

    Args:
        gpu_id (int): The GPU to run the function on.
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution.

    Raises:
        ValueError: If the GPU index is invalid.
        RuntimeError: If there is an error executing the function on the GPU.
    """
    try:
        available_gpus = list_available_gpus()
        if gpu_id >= len(available_gpus):
            raise ValueError(
                f"Invalid GPU ID: {gpu_id}. Available GPUs are {available_gpus}."
            )
        logger.info(f"Executing function on GPU {gpu_id} using Ray.")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        @ray.remote(num_gpus=1)
        def task_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        result = ray.get(task_wrapper.remote(*args, **kwargs))
        logger.info(f"Execution on GPU {gpu_id} completed.")
        return result
    except Exception as e:
        logger.error(f"Error executing on GPU {gpu_id}: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def execute_on_multiple_gpus(
    gpu_ids: List[int],
    func: Callable,
    all_gpus: bool = False,
    timeout: float = None,
    *args: Any,
    **kwargs: Any,
) -> List[Any]:
    """
    Executes a callable across multiple GPUs using Ray.

    Args:
        gpu_ids (List[int]): The list of GPU IDs to run the function on.
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        List[Any]: A list of results from the execution on each GPU.

    Raises:
        ValueError: If any GPU index is invalid.
        RuntimeError: If there is an error executing the function on the GPUs.
    """
    try:
        available_gpus = list_available_gpus()
        if any(gpu_id >= len(available_gpus) for gpu_id in gpu_ids):
            raise ValueError(
                f"Invalid GPU IDs: {gpu_ids}. Available GPUs are {available_gpus}."
            )
        logger.info(
            f"Executing function across GPUs {gpu_ids} using Ray."
        )

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=False,
                *args,
                **kwargs,
            )

        @ray.remote(num_gpus=1)
        def task_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        result_futures = [
            task_wrapper.remote(*args, **kwargs) for _ in gpu_ids
        ]
        results = ray.get(result_futures, timeout)
        logger.info(f"Execution on GPUs {gpu_ids} completed.")
        return results
    except Exception as e:
        logger.error(f"Error executing on GPUs {gpu_ids}: {e}")
        raise


def distributed_execute_on_gpus(
    gpu_ids: List[int],
    func: Callable,
    *args: Any,
    num_retries: int = RETRY_COUNT,
    retry_delay: float = RETRY_DELAY,
    **kwargs: Any,
) -> List[Any]:
    """
    Executes a callable across multiple GPUs and nodes using Ray's distributed task scheduling.

    Args:
        gpu_ids (List[int]): List of GPU IDs to run the function on. Must be valid IDs.
        func (Callable): Function to execute. Must be serializable.
        *args (Any): Arguments for the callable
        num_retries (int): Number of retry attempts for failed tasks
        retry_delay (float): Delay in seconds between retries
        **kwargs (Any): Keyword arguments for the callable

    Returns:
        List[Any]: Results from execution on each GPU in order of gpu_ids

    Raises:
        ValueError: If gpu_ids is empty or contains invalid IDs
        RuntimeError: If Ray initialization or task execution fails
        TimeoutError: If execution exceeds maximum retry attempts
    """
    if not gpu_ids:
        raise ValueError("Must specify at least one GPU ID")

    available_gpus = [gpu.id for gpu in GPUtil.getGPUs()]
    invalid_gpus = [id for id in gpu_ids if id not in available_gpus]
    if invalid_gpus:
        raise ValueError(f"Invalid GPU IDs: {invalid_gpus}")

    try:
        logger.info(
            f"Executing function across distributed GPUs {gpu_ids}"
        )

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        @ray.remote(num_gpus=1)
        def task_wrapper(*task_args, **task_kwargs):
            for attempt in range(num_retries):
                try:
                    return func(*task_args, **task_kwargs)
                except Exception as e:
                    if attempt == num_retries - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}"
                    )
                    time.sleep(
                        retry_delay * (attempt + 1)
                    )  # Exponential backoff

        # Distribute tasks
        pool = Pool()
        result_futures = [
            pool.apply_async(task_wrapper.remote, args=(args, kwargs))
            for gpu_id in gpu_ids
        ]
        pool.close()
        pool.join()

        results = [future.get() for future in result_futures]
        logger.info("Distributed execution completed successfully")
        return results

    except Exception as e:
        error_msg = f"Error during distributed execution: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# # Example function to run
# def sample_task(n: int) -> int:
#     return n * n


# # List CPUs and execute on CPU 0
# cpus = list_available_cpus()
# execute_on_cpu(0, sample_task, 10)

# # List CPUs and execute using 4 CPU cores
# execute_with_cpu_cores(4, sample_task, 10)

# # List GPUs and execute on GPU 0
# gpus = list_available_gpus()
# execute_on_gpu(0, sample_task, 10)

# # Execute across multiple GPUs
# execute_on_multiple_gpus([0, 1], sample_task, 10)
