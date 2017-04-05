
cmake_minimum_required(VERSION 2.8)
project(cuRANSAC)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_VERBOSE_MAKEFILE ON)

# cuRANSAC library requirements
find_package(CUDA 6.5 REQUIRED)

# Additional requirements for application
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(VTK REQUIRED) 
pkg_search_module(GLFW REQUIRED glfw3)

include_directories(${VTK_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/)


#Workaround for GCC / G++ 5.3 and CUDA 
if( (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC) AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.4)
	message(FATAL_ERROR "Current CUDA versions do not support GCC > 5.4. Compilation is likely to fail. TIP: Your distro may have package compatible compilers with CUDA, if so they may be enabled by the following environment variables: \nexport CC=/opt/cuda/bin/gcc\nexport CXX=/opt/cuda/bin/g++ ")
endif()

message(AUTHOR_WARNING "On some configurations, compiler flags are automatically forwarded to the CUDA compiler. If errors related to std appear try editing the CMakeLists.txt according to the comments therein.")

if( (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC) AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)	
	
	# NOTE 1: On e.g. Arch Linux with CUDA 8 the --std=c++11 flag gets propagated to nvcc automatically so use the following instead:
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=sm_30;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES;-D__STRICT_ANSI__)
	#NOTE 2: You may need to specify a lower/higher compute_XX and code=sm_XX version, depending on your GPU
	
	# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --std=c++11 -gencode arch=compute_30,code=sm_30;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES;-D__STRICT_ANSI__)
else()
	# see NOTE 1, above
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=sm_30;-D_FORCE_INLINES) 
	# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --std=c++11 -gencode arch=compute_30,code=sm_30;-D_FORCE_INLINES)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -march=native")
	
cuda_add_library(graph_tools graph_tools.cu)
cuda_add_library(RANSAC RANSAC.cu)

add_executable(cuRANSAC read_graphs.cpp readfile.cpp)

target_link_libraries(cuRANSAC ${VTK_LIBRARIES} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} graph_tools RANSAC)
