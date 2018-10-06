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

TEST(UtilTest, TestOnes) {

    ManagedTensor2i ones2i = Ones<2,int>({5, 3});

    ASSERT_EQ(5, ones2i.DimensionSize(0));
    ASSERT_EQ(3, ones2i.DimensionSize(1));

    for (int i = 0; i < ones2i.DimensionSize(0); ++i) {
        for (int j = 0; j < ones2i.DimensionSize(1); ++j) {
            ASSERT_EQ(1, ones2i(i, j));
        }
    }

    ManagedTensor1f ones1f = Ones<float>(10);

    ASSERT_EQ(10, ones1f.DimensionSize(0));

    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(1.f, ones1f(i));
    }

}

TEST(UtilTest, TestOnesLike) {

    Tensor3d tensor({2,4,3});

    {
        ManagedTensor3d ones3d = OnesLike(tensor);

        ASSERT_EQ(2, ones3d.DimensionSize(0));
        ASSERT_EQ(4, ones3d.DimensionSize(1));
        ASSERT_EQ(3, ones3d.DimensionSize(2));

        for (int i = 0; i < ones3d.DimensionSize(0); ++i) {
            for (int j = 0; j < ones3d.DimensionSize(1); ++j) {
                for (int k = 0; k < ones3d.DimensionSize(2); ++k) {
                    ASSERT_EQ(1.0, ones3d(i, j, k));
                }
            }
        }
    }

    {
        ManagedTensor3d ones3d = OnesLike(tensor.SubTensor({0, 1, 0}, {2, 2, 2}));

        ASSERT_EQ(2, ones3d.DimensionSize(0));
        ASSERT_EQ(2, ones3d.DimensionSize(1));
        ASSERT_EQ(2, ones3d.DimensionSize(2));

        for (int i = 0; i < ones3d.DimensionSize(0); ++i) {
            for (int j = 0; j < ones3d.DimensionSize(1); ++j) {
                for (int k = 0; k < ones3d.DimensionSize(2); ++k) {
                    ASSERT_EQ(1.0, ones3d(i, j, k));
                }
            }
        }
    }

    {
        ManagedTensor2d ones2d = OnesLike(tensor.Slice<1>(2));

        ASSERT_EQ(2, ones2d.DimensionSize(0));
        ASSERT_EQ(3, ones2d.DimensionSize(1));

        for (int i = 0; i < ones2d.DimensionSize(0); ++i) {
            for (int j = 0; j < ones2d.DimensionSize(1); ++j) {
                ASSERT_EQ(1.0, ones2d(i, j));
            }
        }
    }

}

TEST(UtilTest, TestARange) {

    {
        ManagedVector<int> vec = ARange(5);

        ASSERT_EQ(5, vec.Length());

        for (int i = 0; i < vec.Length(); ++i) {
            ASSERT_EQ(i, vec(i));
        }
    }

    {
        ManagedVector<uint> vec = ARange<uint>(2, 8);

        ASSERT_EQ(6, vec.Length());

        for (int i = 0; i < vec.Length(); ++i) {
            ASSERT_EQ(i+2, vec(i));
        }

    }

    {
        ManagedVector<std::size_t> vec = ARange<std::size_t>(100, 500, 50);

        ASSERT_EQ(8, vec.Length());

        for (int i = 0; i < vec.Length(); ++i) {
            ASSERT_EQ(100 + 50 * i, vec(i));
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

TEST(UtilTest, TestReshape) {

    short data[16] = {  0, 1, 2, 3,
                        4, 5, 6, 7,
                        8, 9, 10, 11,
                        12, 13, 14, 15 };

    NDT::Tensor<1, short> tensor1D(16, data);

    NDT::Tensor<2, short> tensor2D = NDT::Reshape(tensor1D, Eigen::Vector2i(4, 4));

    ASSERT_EQ(4, tensor2D.DimensionSize(0));
    ASSERT_EQ(4, tensor2D.DimensionSize(1));

    NDT::Tensor<3, short> tensor3D = NDT::Reshape(tensor2D, Eigen::Vector3i(2, -1, 2));

    ASSERT_EQ(2, tensor3D.DimensionSize(0));
    ASSERT_EQ(4, tensor3D.DimensionSize(1));
    ASSERT_EQ(2, tensor3D.DimensionSize(2));

}



}