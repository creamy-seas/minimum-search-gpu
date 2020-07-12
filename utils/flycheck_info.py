import numba as nb

def gpu_check():
    if nb.cuda.is_available():
        device = nb.cuda.get_current_device()
        dimensions = {
            "max_shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
            "max_block_dim_x": device.MAX_BLOCK_DIM_X,
            "max_block_dim_y": device.MAX_BLOCK_DIM_Y,
            "max_block_dim_z": device.MAX_BLOCK_DIM_Z,
            "max_grid_dim_x": device.MAX_GRID_DIM_X,
            "max_grid_dim_y": device.MAX_GRID_DIM_Y,
            "max_grid_dim_z": device.MAX_GRID_DIM_Z,
        }
        print(f"🦑 Found device {str(device)}")
        print(dimensions)

        print(f"🦑 Max shared memory cells for int16: {device.MAX_SHARED_MEMORY_PER_BLOCK / nb.int16.bitwidth}")
        print(f"🦑 Max shared memory cells for float32: {device.MAX_SHARED_MEMORY_PER_BLOCK / nb.float32.bitwidth}")
        return dimensions
    raise RuntimeError("Missing GPU")
