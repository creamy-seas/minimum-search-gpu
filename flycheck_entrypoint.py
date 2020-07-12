import math

from utils.info import gpu_check

gpu_info = gpu_check()
max_shared_memory_per_block = (
    gpu_info["max_shared_memory_per_block"]
)
# print(f"🦑 Max shared memory cells for int16: {max_shared_memory_per_block}")
# pi = math.pi

# # Parameters for simulation ###################################################
# NUMBER_OF_PHI_POINTS = 5
# NUMBER_OF_FIELD_POINTS = 50
# ALPHA = 1
# LOWER = -0.5
# UPPER = 1.5

# lr_array = np.linspace(LOWER * 2 * pi, UPPER * 2 * pi, NUMBER_OF_FIELD_POINTS)
# phixx_array = np.linspace(-pi, pi, NUMBER_OF_PHI_POINTS)

# THREADS_PER_BLOCK = (8, 10, 10)


# for L in range(0, 1):
#     print(f"🐳 Evaluated potential for L={L}")


#     kernel[NUMBER_OF_FIELD_POINTS, THREADS_PER_BLOCK](
#         DEVICE_phixx_array, DEVICE_lr_array, L, ALPHA, DEVICE_potential_array
#     )
#     # potential_array =
#     # DEVICE_potential_array_to_minimize = DEVICE_potential_array.copy_to_host()
#     print("Original")
#     print(DEVICE_potential_array.copy_to_host()[0][0][0])
#     print(DEVICE_potential_array.copy_to_host()[0][0][1])
