#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

// TODO: fix this test case
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

    Eigen::Matrix<float, 2, 2> g = image.InterpolationGradient(0.5f, 0.5);

    ASSERT_NEAR( 1.5, g(0,0), 1e-4);  ASSERT_NEAR(3.5, g(0,1), 1e-4);
    ASSERT_NEAR(-1.5, g(1,0), 1e-4);  ASSERT_NEAR(-3.5, g(1,1), 1e-4);

    Eigen::Matrix<float, 2, 2> g2 = image.InterpolationGradient(Eigen::Vector2f(0.5f, 0.5f));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_EQ(g2(i,j), g(i,j));
        }
    }

    g = image.InterpolationGradient(0.5f, 1.25f);

    // tx = 0.5
    // ty = 0.25
    // 3.0 * (1-tx) * (1-ty) + 5.0 * tx * (1-ty) + 6.0 * (1-tx) * ty + 7.0 * tx * ty
    // -3.0 * 0.75 + 5.0 * 0.75 - 6.0 * 0.25 + 7.0 * 0.25
    // -3.0 * 0.5 - 5.0 * 0.5 + 6.0 * 0.5 + 7.0 * 0.5

    ASSERT_NEAR( 1.75, g(0, 0), 1e-4 );  ASSERT_NEAR( 2.5, g(0, 1), 1e-4 );
    ASSERT_NEAR(-1.75, g(1, 0), 1e-4 );  ASSERT_NEAR(-2.5, g(1, 1), 1e-4 );

    g2 = image.InterpolationGradient(Eigen::Vector2d(0.5, 1.25));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_EQ(g2(i,j), g(i,j));
        }
    }

    g = image.InterpolationGradient(2, 1.5f);

    ASSERT_NEAR(-3.25, g(0,0), 1e-4);  ASSERT_NEAR( -6.0, g(0,1), 1e-4);
    ASSERT_NEAR( 3.25, g(1,0), 1e-4);  ASSERT_NEAR(  6.0, g(1,1), 1e-4);

}

TEST(TransformInterpolationGradientTest, Test2DVectorValued) {

    Eigen::Vector2i data[] = { { 0,  0}, { 1, -1}, { 2, -2}, { 3, -3},
                               { 3, -3}, { 5, -5}, { 4, -4}, { 0,  0},
                               { 6, -6}, { 7, -7}, {-2,  2}, {-1,  1} };

    Image<Eigen::Vector2i> image( { 4, 3}, data);

//    struct Floater {
//
//        inline Eigen::Vector2f operator()(const Eigen::Vector2i & v) {
//            return v.cast<float>();
//        }
//
//    };

    auto floater = [](const Eigen::Vector2i & v) { return Eigen::Vector2f(v.cast<float>()); };

//    Eigen::Vector2f vf = floater(Eigen::Vector2i(-1,3));

//    Floater floater;

//    const bool typeMatch = std::is_same<decltype(floater(*data)), Eigen::Vector2f>::value;
//
//    ASSERT_TRUE(typeMatch);
//
    Eigen::Matrix<float, 2, 2> g = image.TransformInterpolationGradient(floater, 0.5f, 0.5);

    ASSERT_NEAR( 1.5, g(0,0), 1e-4);  ASSERT_NEAR(3.5, g(0,1), 1e-4);
    ASSERT_NEAR(-1.5, g(1,0), 1e-4);  ASSERT_NEAR(-3.5, g(1,1), 1e-4);

    Eigen::Matrix<float, 2, 2> g2 = image.TransformInterpolationGradient(floater, Eigen::Vector2f(0.5f, 0.5f));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            ASSERT_EQ(g2(i,j), g(i,j));
        }
    }
//
//    g = image.InterpolationGradient(0.5f, 1.25f);
//
//    // tx = 0.5
//    // ty = 0.25
//    // 3.0 * (1-tx) * (1-ty) + 5.0 * tx * (1-ty) + 6.0 * (1-tx) * ty + 7.0 * tx * ty
//    // -3.0 * 0.75 + 5.0 * 0.75 - 6.0 * 0.25 + 7.0 * 0.25
//    // -3.0 * 0.5 - 5.0 * 0.5 + 6.0 * 0.5 + 7.0 * 0.5
//
//    ASSERT_NEAR( 1.75, g(0, 0), 1e-4 );  ASSERT_NEAR( 2.5, g(0, 1), 1e-4 );
//    ASSERT_NEAR(-1.75, g(1, 0), 1e-4 );  ASSERT_NEAR(-2.5, g(1, 1), 1e-4 );
//
//    g2 = image.InterpolationGradient(Eigen::Vector2d(0.5, 1.25));
//
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 2; ++j) {
//            ASSERT_EQ(g2(i,j), g(i,j));
//        }
//    }
//
//    g = image.InterpolationGradient(2, 1.5f);
//
//    ASSERT_NEAR(-3.25, g(0,0), 1e-4);  ASSERT_NEAR( -6.0, g(0,1), 1e-4);
//    ASSERT_NEAR( 3.25, g(1,0), 1e-4);  ASSERT_NEAR(  6.0, g(1,1), 1e-4);

}


} // namespace