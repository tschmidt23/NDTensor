#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(GradientTest, TestGradients2D) {

    float data[] = { 0.0f, 1.0f, 2.0f, 3.0f,
                     3.0f, 5.0f, 4.0f, 0.0f,
                     6.0f, 7.0f,-2.0f,-1.0f };

    Image<float> image( { 4, 3}, data);


    // test backward gradients

    // test integral gradient
    const Eigen::Matrix<float, 1, 2> integralBackwardGradient = image.BackwardDifference(1, 1);
    ASSERT_NEAR(integralBackwardGradient(0), 2.f, 1e-4);
    ASSERT_NEAR(integralBackwardGradient(1), 4.f, 1e-4);

    // test interpolated gradient
    const Eigen::Matrix<float, 1, 2> interpolatedBackwardGradient = image.BackwardDifference(1.5, 1.5);
    ASSERT_NEAR(interpolatedBackwardGradient(0), -1.75f, 1e-4);
    ASSERT_NEAR(interpolatedBackwardGradient(1),   0.5f, 1e-4);

    // test central gradients

    // test integral gradient
    const Eigen::Matrix<float, 1, 2> integralCentralGradient = image.CentralDifference(1, 1);
    ASSERT_NEAR(integralCentralGradient(0), 0.5f, 1e-4);
    ASSERT_NEAR(integralCentralGradient(1), 3.f, 1e-4);

    // test interpolated gradient
    const Eigen::Matrix<float, 1, 2> interpolatedCentralGradient = image.CentralDifference(1.5, 1);
    ASSERT_NEAR(interpolatedCentralGradient(0), -1.f, 1e-4);
    ASSERT_NEAR(interpolatedCentralGradient(1), 0.5f, 1e-4);

    // test forward gradients

    // test integral gradient
    const Eigen::Matrix<float, 1, 2> integralForwardGradient = image.ForwardDifference(1, 1);
    ASSERT_NEAR(integralForwardGradient(0), -1.f, 1e-4);
    ASSERT_NEAR(integralForwardGradient(1),  2.f,  1e-4);

    // test interpolated gradient
    const Eigen::Matrix<float, 1, 2> interpolatedForwardGradient = image.ForwardDifference(1.5, 0.5);
    ASSERT_NEAR(interpolatedForwardGradient(0), -0.75f, 1e-4);
    ASSERT_NEAR(interpolatedForwardGradient(1),   0.5f, 1e-4);

}


} // namespace