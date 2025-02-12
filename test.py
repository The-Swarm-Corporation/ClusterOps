import GPUtil
import psutil

from clusterops.main import (
    list_available_cpus,
    execute_with_cpu_cores,
    execute_with_all_cpu_cores,
    select_best_gpu,
    get_cpu_info,
    get_optimal_core_count,
    execute_on_cpu,
    list_available_gpus,
    execute_on_gpu,
    execute_on_multiple_gpus,
    distributed_execute_on_gpus,
)


def test_list_available_cpus():
    """Test listing available CPUs"""
    # Basic functionality test
    cpus = list_available_cpus()
    assert isinstance(cpus, list)
    assert len(cpus) > 0
    assert all(isinstance(cpu, int) for cpu in cpus)

    # Mock CPU count for error case
    original_cpu_count = psutil.cpu_count
    psutil.cpu_count = lambda logical: None
    try:
        error_raised = False
        try:
            list_available_cpus()
        except RuntimeError:
            error_raised = True
        assert error_raised
    finally:
        psutil.cpu_count = original_cpu_count


def test_execute_with_cpu_cores():
    """Test executing function with specific CPU cores"""

    def sample_func(x: int) -> int:
        return x * 2

    # Test with valid core count
    result = execute_with_cpu_cores(1, sample_func, 5)
    assert result == 10

    # Test with invalid core count
    error_raised = False
    try:
        execute_with_cpu_cores(-1, sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised

    # Test with excessive core count
    error_raised = False
    try:
        execute_with_cpu_cores(9999, sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised


def test_execute_with_all_cpu_cores():
    """Test executing function with all CPU cores"""

    def sample_func(x: int) -> int:
        return x * 2

    # Basic functionality test
    result = execute_with_all_cpu_cores(sample_func, 5)
    assert result == 10

    # Test with function that raises exception
    def failing_func():
        raise ValueError("Test error")

    error_raised = False
    try:
        execute_with_all_cpu_cores(failing_func)
    except Exception:
        error_raised = True
    assert error_raised


def test_select_best_gpu():
    """Test selecting best GPU"""
    if not GPUtil.getGPUs():
        # Skip test if no GPUs available
        return

    gpu_id = select_best_gpu()
    assert isinstance(gpu_id, int)
    assert gpu_id >= 0


def test_get_cpu_info():
    """Test getting CPU information"""
    physical, logical, available = get_cpu_info()

    assert isinstance(physical, int)
    assert isinstance(logical, int)
    assert isinstance(available, int)

    assert physical > 0
    assert logical >= physical
    assert available > 0
    assert available <= logical


def test_get_optimal_core_count():
    """Test getting optimal core count"""
    # Test with no requested cores
    optimal = get_optimal_core_count()
    assert isinstance(optimal, int)
    assert optimal > 0

    # Test with specific requested cores
    requested = 2
    optimal = get_optimal_core_count(requested)
    assert optimal <= requested
    assert optimal > 0

    # Test with excessive requested cores
    large_request = 9999
    optimal = get_optimal_core_count(large_request)
    assert optimal > 0
    assert optimal <= psutil.cpu_count()


def test_execute_on_cpu():
    """Test executing function on CPU"""

    def sample_func(x: int) -> int:
        return x * 2

    # Test with valid core count
    result = execute_on_cpu(1, sample_func, 5)
    assert result == 10

    # Test with invalid core count
    error_raised = False
    try:
        execute_on_cpu(-1, sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised


def test_list_available_gpus():
    """Test listing available GPUs"""
    try:
        gpus = list_available_gpus()
        assert isinstance(gpus, list)
        assert all(isinstance(gpu, str) for gpu in gpus)
    except RuntimeError:
        # This is acceptable if no GPUs are available
        pass


def test_execute_on_gpu():
    """Test executing function on GPU"""
    if not GPUtil.getGPUs():
        # Skip test if no GPUs available
        return

    def sample_func(x: int) -> int:
        return x * 2

    # Test with valid GPU ID
    result = execute_on_gpu(0, sample_func, 5)
    assert result == 10

    # Test with invalid GPU ID
    error_raised = False
    try:
        execute_on_gpu(9999, sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised


def test_execute_on_multiple_gpus():
    """Test executing function on multiple GPUs"""
    if len(GPUtil.getGPUs()) < 2:
        # Skip test if not enough GPUs available
        return

    def sample_func(x: int) -> int:
        return x * 2

    # Test with valid GPU IDs
    results = execute_on_multiple_gpus([0, 1], sample_func, 5)
    assert len(results) == 2
    assert all(result == 10 for result in results)

    # Test with invalid GPU IDs
    error_raised = False
    try:
        execute_on_multiple_gpus([9999], sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised


def test_distributed_execute_on_gpus():
    """Test distributed execution on GPUs"""
    if not GPUtil.getGPUs():
        # Skip test if no GPUs available
        return

    def sample_func(x: int) -> int:
        return x * 2

    # Test with valid GPU IDs
    results = distributed_execute_on_gpus([0], sample_func, 5)
    assert len(results) == 1
    assert results[0] == 10

    # Test with empty GPU list
    error_raised = False
    try:
        distributed_execute_on_gpus([], sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised

    # Test with invalid GPU IDs
    error_raised = False
    try:
        distributed_execute_on_gpus([9999], sample_func, 5)
    except ValueError:
        error_raised = True
    assert error_raised


def run_all_tests():
    """Run all test cases"""
    test_functions = [
        test_list_available_cpus,
        test_execute_with_cpu_cores,
        test_execute_with_all_cpu_cores,
        test_select_best_gpu,
        test_get_cpu_info,
        test_get_optimal_core_count,
        test_execute_on_cpu,
        test_list_available_gpus,
        test_execute_on_gpu,
        test_execute_on_multiple_gpus,
        test_distributed_execute_on_gpus,
    ]

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
