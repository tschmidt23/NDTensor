project("NDTensor")
cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    set( CMAKE_BUILD_TYPE Release)
endif()

if(NOT CUDA_COMPUTE_CAPABILITY)
    set( CUDA_COMPUTE_CAPABILITY "52")
endif()

add_definitions(-std=c++11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --expt-relaxed-constexpr" )


find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest QUIET)

file(GLOB
        NDT_SOURCES
        "./Src/*.cpp"
        "./Src/*.cu")

include_directories(${PROJECT_SOURCE_DIR}/Include ${EIGEN3_INCLUDE_DIR})
cuda_add_library(NDT ${NDT_SOURCES})
message(STATUS "${PROJECT_SOURCE_DIR}/Include")
target_include_directories(NDT PUBLIC ${PROJECT_SOURCE_DIR}/Include ${EIGEN3_INCLUDE_DIR})

if (${GTEST_FOUND})

    file( GLOB
            TEST_SOURCES
            "./Test/*.cpp"
            )

    cuda_add_executable(Test
            ${TEST_SOURCES}
            )
    target_link_libraries(Test ${GTEST_BOTH_LIBRARIES} pthread glog NDT)
    target_include_directories(Test PUBLIC ${PROJECT_SOURCE_DIR}/Include ${EIGEN3_INCLUDE_DIR})

endif()

set( NDTensor_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include/" )
set( NDTensor_INCLUDE_DIRS ${NDTensor_INCLUDE_DIR} )
set( NDTensor_LIBRARY "NDT" )
set( NDTensor_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib/" )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NDTensorConfig.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/NDTensorConfig.cmake
                @ONLY ESCAPE_QUOTES)

install(DIRECTORY Include/ DESTINATION include)

#install cmake config
install_files(/share .cmake NDTensorConfig)

install(TARGETS NDT DESTINATION lib)
