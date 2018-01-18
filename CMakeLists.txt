project("NDTensor")
cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    set( CMAKE_BUILD_TYPE Release)
endif()

if(NOT CUDA_COMPUTE_CAPABILITY)
    set( CUDA_COMPUTE_CAPABILITY "52")
endif()

add_definitions(-std=c++11)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest QUIET)

if (${GTEST_FOUND})

    file( GLOB
            TEST_SOURCES
            "./Test/*.cpp"
            )

    cuda_add_executable(Test
            ${TEST_SOURCES}
            )
    target_link_libraries(Test ${GTEST_BOTH_LIBRARIES} pthread glog)
    target_include_directories(Test PUBLIC ${PROJECT_SOURCE_DIR}/Include ${EIGEN3_INCLUDE_DIR})

endif()

set( NDTensor_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include/" )
set( NDTensor_INCLUDE_DIRS ${NDTensor_INCLUDE_DIR} )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NDTensorConfig.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/NDTensorConfig.cmake
                @ONLY ESCAPE_QUOTES)

install_files(/include/NDT .h 
	Include/NDT/TensorBase 
	Include/NDT/Tensor
	Include/NDT/TensorView
	Include/NDT/TupleHelpers)
install_files(/share/NDT .cmake NDTensorConfig)
