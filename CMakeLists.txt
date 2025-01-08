cmake_minimum_required(VERSION 3.10)
project(CPDL)

# Force use of C++17 and libc++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")
endif()

# Add option for debug mode
option(DEBUG_MODE "Enable debug mode with additional visualization" ON)

# Add debug definition if enabled
if(DEBUG_MODE)
    add_definitions(-DDEBUG_MODE)
endif()

# Add source files
add_executable(CPDL src/engine.cpp)

# Include directories
target_include_directories(CPDL PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Set compiler features
target_compile_features(CPDL PRIVATE cxx_std_17)

# Add testing
enable_testing()
add_test(NAME ValueTests COMMAND CPDL) 

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif() 