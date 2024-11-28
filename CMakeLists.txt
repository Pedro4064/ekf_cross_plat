# cmake_minimum_required(VERSION 3.13)
# project(STM32_EKF_Project)

# # Set the C standard
# set(CMAKE_C_STANDARD 11)

# # Add the main source file to a library
# add_library(STM32_EKF STATIC src/EstimationSystem.c)

# # Add the CMSIS-DSP library from the relative path
# add_subdirectory(lib/CMSIS-DSP/Source)

# # Link CMSIS-DSP to your target
# target_link_libraries(STM32_EKF PRIVATE CMSISDSP)

# # Add CMSIS-DSP and CMSIS-Core include directories
# # target_include_directories(CMSISDSP PUBLIC
# #     ${CMAKE_SOURCE_DIR}/lib/CMSIS_5/CMSIS/Core/Include  # CMSIS-Core must come first
# # )

# target_include_directories(STM32_EKF PUBLIC
#     ${CMAKE_SOURCE_DIR}/lib/CMSIS-DSP/Include          # CMSIS-DSP headers
#     ${CMAKE_SOURCE_DIR}/lib/CMSIS-DSP/PrivateInclude   # DSP private headers (if required)
#     ${CMAKE_SOURCE_DIR}/inc/                           # Your project include headers
# )

cmake_minimum_required(VERSION 3.13)
project(ARM_Math_Project)

# Set the C standard
set(CMAKE_C_STANDARD 11)

# Add the main source file
add_executable(ARM_Math_Project src/main.c)

# Add the CMSIS-DSP library from the relative path
add_subdirectory(lib/CMSIS-DSP/Source)

# Link CMSIS-DSP to your target
target_link_libraries(ARM_Math_Project PRIVATE CMSISDSP)

# Add CMSIS-DSP and CMSIS-Core include directories
# target_include_directories(CMSISDSP PUBLIC
#     ${CMAKE_SOURCE_DIR}/lib/CMSIS_5/CMSIS/Core/Include  # CMSIS-Core must come first
#     # ${CMAKE_SOURCE_DIR}/lib/CMSIS-DSP/Include          # CMSIS-DSP headers
#     # ${CMAKE_SOURCE_DIR}/lib/CMSIS-DSP/PrivateInclude   # DSP private headers
# )

target_include_directories(ARM_Math_Project PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/CMSIS-DSP/Include
    ${CMAKE_SOURCE_DIR}/lib/CMSIS-DSP/PrivateInclude # Include DSP's private headers if required
)