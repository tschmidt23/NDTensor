#include <gtest/gtest.h>

#include <NDT/Util.h>

namespace {

using namespace NDT;

TEST(UtilTest, TestExpandDims) {

    float data[] = { };

    Tensor<2, float, HostResident> tensor({2, 4}, data);

    Tensor<3, float, HostResident> expanded = ExpandDims(tensor, 0);

    EXPECT_EQ(expanded.DimensionSize(0), 1);
    EXPECT_EQ(expanded.DimensionSize(1), tensor.DimensionSize(0));
    EXPECT_EQ(expanded.DimensionSize(2), tensor.DimensionSize(1));

    expanded = ExpandDims(tensor, 1);

    EXPECT_EQ(expanded.DimensionSize(0), tensor.DimensionSize(0));
    EXPECT_EQ(expanded.DimensionSize(1), 1);
    EXPECT_EQ(expanded.DimensionSize(2), tensor.DimensionSize(1));

    expanded = ExpandDims(tensor, 2);

    EXPECT_EQ(expanded.DimensionSize(0), tensor.DimensionSize(0));
    EXPECT_EQ(expanded.DimensionSize(1), tensor.DimensionSize(1));
    EXPECT_EQ(expanded.DimensionSize(2), 1);

}

TEST(UtilTest, TestZeros) {

    ManagedTensor2i zeros2i = Zeros<2,int>({5, 3});

    ASSERT_EQ(5, zeros2i.DimensionSize(0));
    ASSERT_EQ(3, zeros2i.DimensionSize(1));

    for (int i = 0; i < zeros2i.DimensionSize(0); ++i) {
        for (int j = 0; j < zeros2i.DimensionSize(1); ++j) {
            ASSERT_EQ(0, zeros2i(i, j));
        }
    }

    ManagedTensor1f zeros1f = Zeros<float>(10);

    ASSERT_EQ(10, zeros1f.DimensionSize(0));

    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(0, zeros1f(i));
    }

}

TEST(UtilTest, TestZerosLike) {

    Tensor3d tensor({2,4,3});

    {
        ManagedTensor3d zeros3d = ZerosLike(tensor);

        ASSERT_EQ(2, zeros3d.DimensionSize(0));
        ASSERT_EQ(4, zeros3d.DimensionSize(1));
        ASSERT_EQ(3, zeros3d.DimensionSize(2));

        for (int i = 0; i < zeros3d.DimensionSize(0); ++i) {
            for (int j = 0; j < zeros3d.DimensionSize(1); ++j) {
                for (int k = 0; k < zeros3d.DimensionSize(2); ++k) {
                    ASSERT_EQ(0, zeros3d(i, j, k));
                }
            }
        }
    }

    {
        ManagedTensor3d zeros3d = ZerosLike(tensor.SubTensor({0, 1, 0}, {2, 2, 2}));

        ASSERT_EQ(2, zeros3d.DimensionSize(0));
        ASSERT_EQ(2, zeros3d.DimensionSize(1));
        ASSERT_EQ(2, zeros3d.DimensionSize(2));

        for (int i = 0; i < zeros3d.DimensionSize(0); ++i) {
            for (int j = 0; j < zeros3d.DimensionSize(1); ++j) {
                for (int k = 0; k < zeros3d.DimensionSize(2); ++k) {
                    ASSERT_EQ(0, zeros3d(i, j, k));
                }
            }
        }
    }

    {
        ManagedTensor2d zeros2d = ZerosLike(tensor.Slice<1>(2));

        ASSERT_EQ(2, zeros2d.DimensionSize(0));
        ASSERT_EQ(3, zeros2d.DimensionSize(1));

        for (int i = 0; i < zeros2d.DimensionSize(0); ++i) {
            for (int j = 0; j < zeros2d.DimensionSize(1); ++j) {
                ASSERT_EQ(0, zeros2d(i, j));
            }
        }
    }

}

TEST(UtilTest, TestMeanAndSum) {

    float data[10] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };

    Tensor1f tensor(10, data);

    ASSERT_NEAR(5.5f, Mean(tensor), 1e-4);

    ASSERT_NEAR(55.f, Sum(tensor), 1e-4);

}

TEST(UtilTest, TestDot) {

    float dataA[4] = {  1.0, -2.0, 7.5, -3.0 };
    float dataB[4] = { -2.5,  4.0, 3.0,  2.0 };

    static constexpr float expectedDotProduct = -2.5f - 8.f + 22.5f - 6.f;

    NDT::Vector<float> vectorA(4, dataA);
    NDT::Vector<float> vectorB(4, dataB);

    ASSERT_NEAR(expectedDotProduct, NDT::Dot(vectorA, vectorB), 1e-4);

}

}