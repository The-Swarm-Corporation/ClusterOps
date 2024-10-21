import os
import platform
import sys
import time
from typing import Any, Callable, List, Optional

import GPUtil
import psutil
import ray
from loguru import logger
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
)
def execute_on_cpu(
    cpu_id: int, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Executes a callable on a specific CPU.

    Args:
        cpu_id (int): The CPU core to run the function on.
        func (Callable): The function to be executed.
        *args (Any): Arguments for the callable.
        **kwargs (Any): Keyword arguments for the callable.

    Returns:
        Any: The result of the function execution.

    Raises:
        ValueError: If the CPU core specified is invalid.
        RuntimeError: If there is an error executing the function on the CPU.
    """
    try:
        available_cpus = list_available_cpus()
        if cpu_id not in available_cpus:
            raise ValueError(
                f"Invalid CPU core: {cpu_id}. Available CPUs are {available_cpus}."
            )

        if platform.system() == "Darwin":  # macOS
            logger.warning(
                "CPU affinity is not supported on macOS. Skipping setting CPU affinity."
            )
        else:
            logger.info(f"Setting CPU affinity to core {cpu_id}.")
            psutil.Process().cpu_affinity([cpu_id])

        result = func(*args, **kwargs)
        logger.info(f"Execution on CPU {cpu_id} completed.")
        return result
    except Exception as e:
        logger.error(f"Error executing on CPU {cpu_id}: {e}")
        raise


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
def execute_with_cpu_cores(
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
