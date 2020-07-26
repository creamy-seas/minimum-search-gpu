from typing import List, Callable, Tuple, Optional

import numba as nb
from numba import cuda


def gpu_check():
    if cuda.is_available():
        device = cuda.get_current_device()
        parameters = {
            "max_shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
            "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "max_block_dim_x": device.MAX_BLOCK_DIM_X,
            "max_block_dim_y": device.MAX_BLOCK_DIM_Y,
            "max_block_dim_z": device.MAX_BLOCK_DIM_Z,
            "max_grid_dim_x": device.MAX_GRID_DIM_X,
            "max_grid_dim_y": device.MAX_GRID_DIM_Y,
            "max_grid_dim_z": device.MAX_GRID_DIM_Z,
        }
        # print(f"🦑 Found device {str(device)}")
        # print(parameters)

        # print(
        # f"🦑 Max shared memory cells for int16: {device.MAX_SHARED_MEMORY_PER_BLOCK / nb.int16.bitwidth}"
        # )
        # print(
        # f"🦑 Max shared memory cells for float32: {device.MAX_SHARED_MEMORY_PER_BLOCK / nb.float32.bitwidth}"
        # )
        return parameters
    raise RuntimeError("Missing GPU")


def allocate_max_threads(
    user_defined_number: Optional[int] = None, verbose=False
) -> Tuple[int, int, int]:
    gpu_info = gpu_check()
    if verbose:
        print(
            f"""Thread parameters:
    > Max threads per block: {gpu_info['max_threads_per_block']}
    > Max threads in x: {gpu_info['max_block_dim_x']}
    > Max threads in y: {gpu_info['max_block_dim_y']}
    > Max threads in z: {gpu_info['max_block_dim_z']}"""
        )
    max_threads_approximation = int(gpu_info["max_threads_per_block"] ** (1 / 3))
    if user_defined_number is not None:
        max_threads_approximation = user_defined_number

    max_thread_allocation = (
        min(max_threads_approximation, gpu_info["max_block_dim_x"]),
        min(max_threads_approximation, gpu_info["max_block_dim_y"]),
        min(max_threads_approximation, gpu_info["max_block_dim_z"]),
    )
    print(f"🐳 {'Allocating':<20} THREADS_PER_BLOCK = {max_thread_allocation}")

    return max_thread_allocation


def verify_blocks_per_grid(blocks_per_grid: Tuple, verbose=False) -> bool:
    gpu_info = gpu_check()

    if verbose:
        print(
            f"""Block parameters:
    > Max blocks in x: {gpu_info['max_grid_dim_x']}
    > Max blocks in y: {gpu_info['max_grid_dim_y']}
    > Max blocks in z: {gpu_info['max_grid_dim_z']}"""
        )
    for (block_dim, max_dim) in zip(
        blocks_per_grid,
        [
            gpu_info["max_grid_dim_x"],
            gpu_info["max_grid_dim_y"],
            gpu_info["max_grid_dim_z"],
        ],
    ):
        if block_dim > max_dim:
            print("🦑 Allocating too many blocks")
            return False
    print(f"🐳 {'Verified':<20} BLOCKS_PER_GRID={blocks_per_grid}")
    return True
