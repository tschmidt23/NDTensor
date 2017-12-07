#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(InterpolationTest, Test1D) {

    float data[] = { 0.0f, 1.0f, 2.0f, 3.0f, 5.0f };

    Tensor<1, float, HostResident> tensor(5, data);

    for (float i = 0.0f; i < 3.0f; i += 0.1f) {

        ASSERT_NEAR(i, tensor.Interpolate(i), 1e-5);

    }

    ASSERT_NEAR(3.5f, tensor.Interpolate(3.25f), 1e-5);
    ASSERT_NEAR(4.0f, tensor.Interpolate(3.5f),  1e-5);
    ASSERT_NEAR(4.5f, tensor.Interpolate(3.75f), 1e-5);

}

TEST(InterpolationTest, Test2D) {

    float data[] = { 0.0f, 1.0f, 2.0f, 3.0f,
                     3.0f, 5.0f, 4.0f, 0.0f,
                     6.0f, 7.0f,-2.0f,-1.0f };

    Tensor<2, float, HostResident> tensor( { 4, 3}, data);

    for (float x = 0.0f; x < 3.0f; x += 0.1f) {

        // test both float and int versions
        ASSERT_NEAR(x, tensor.Interpolate(x, 0), 1e-5);
        ASSERT_NEAR(x, tensor.Interpolate(x, 0.f), 1e-5);

    }

    for (float y = 0.0f; y < 2.0f; y += 0.1f) {

        // test both float and int versions
        ASSERT_NEAR(3 * y, tensor.Interpolate(0, y), 1e-5);
        ASSERT_NEAR(3 * y, tensor.Interpolate(0.f, y), 1e-5);

    }

    ASSERT_NEAR(7.0f, tensor.Interpolate(1,2), 1e-5);

    ASSERT_NEAR(1.125f, tensor.Interpolate(2.5f, 1.25f), 1e-5);

    ASSERT_NEAR(4.875f, tensor.Interpolate(0.25f, 1.5f), 1e-5);


}

}