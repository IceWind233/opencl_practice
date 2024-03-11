cmake_minimum_required(VERSION 3.0)
project(DLTest)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)

# Find OpenCL
set(OpenCL_DIR "D:/OpenCL-SDK/install")
set(OpenCL_INCLUDE_DIRS "${OpenCL_DIR}/include")
find_package(OpenCL REQUIRED PATHS ${OpenCL_DIR})
include_directories(${OpenCL_INCLUDE_DIRS})
message(STATUS "OpenCL_INCLUDE_DIRS: ${OpenCL_INCLUDE_DIRS}")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
# Add your source files here
add_executable(
	DLTest 
	main.cpp
	"src/Utils.cpp"
	"include/Utils.hpp"
	"src/Conv.cpp"
	"include/Conv.hpp"
)

# Link against OpenDL and OpenCV libraries
target_link_libraries(DLTest OpenCL::OpenCL ${OpenCV_LIBS})
