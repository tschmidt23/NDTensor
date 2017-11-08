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