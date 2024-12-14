from experimental import gpu_scheduler


async def sample_task(n: int) -> int:
    return n * n


print(gpu_scheduler(sample_task, priority=1, n=10))
