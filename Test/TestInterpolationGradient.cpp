#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

//TEST(InterpolationGradientTest, Test2D) {
//
//    float data[] = { 0.0f, 1.0f, 2.0f, 3.0f,
//                     3.0f, 5.0f, 4.0f, 0.0f,
//                     6.0f, 7.0f,-2.0f,-1.0f };
//
//    Image<float> image( { 4, 3}, data);
//
//    Eigen::Matrix<float, 1, 2> g = image.InterpolationGradient(0.5f, 0.5f);
//
//}

TEST(InterpolationGradientTest, Test2DVectorValued) {

    const bool gradientTypeMatch = std::is_same<internal::GradientTraits<Eigen::Vector2f,2>::GradientType,
            Eigen::Matrix<float, 2, 2> >::value;

    ASSERT_TRUE(gradientTypeMatch);

    Eigen::Vector2f data[] = { { 0.0f,  0.0f}, { 1.0f, -1.0f}, { 2.0f, -2.0f}, { 3.0f, -3.0f},
                               { 3.0f, -3.0f}, { 5.0f, -5.0f}, { 4.0f, -4.0f}, { 0.0f,  0.0f},
                               { 6.0f, -6.0f}, { 7.0f, -7.0f}, {-2.0f,  2.0f}, {-1.0f,  1.0f} };

    Image<Eigen::Vector2f> image( { 4, 3}, data);

    std::cout << std::endl;

    Eigen::Matrix<float, 2, 2> g = image.InterpolationGradient(0.5f, 0.5);

    ASSERT_NEAR(1.5f, g(0,0), 1e-4);  ASSERT_NEAR(-1.5f, g(0,1), 1e-4);
    ASSERT_NEAR(3.5f, g(1,0), 1e-4);  ASSERT_NEAR(-3.5f, g(1,1), 1e-4);

    std::cout << g << std::endl;

    g = image.InterpolationGradient(2, 1.5f);

    std::cout << g << std::endl;

}

} // namespace