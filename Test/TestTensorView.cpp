#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(TensorViewTest, Test1D) {

    int data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    Tensor<1, int, HostResident> tensor(10, data);

    TensorView<1, int, HostResident> tensorView = tensor.SubTensor(3, 4);

    EXPECT_EQ(1, tensorView.Dimensions().rows());

    EXPECT_EQ(4, tensorView.DimensionSize(0));  EXPECT_EQ(4, tensorView.Dimensions()(0));

    EXPECT_EQ(3, tensorView(0));
    EXPECT_EQ(4, tensorView(1));
    EXPECT_EQ(5, tensorView(2));
    EXPECT_EQ(6, tensorView(3));

}

TEST(TensorViewTest, Test2D) {

    float data[] = { 0, 1, 2,
                     3, 4, 5,
                     6, 7, 8 };

    Tensor<2, float, HostResident> tensor({3, 3}, data);

    // Test SubTensor
    TensorView<2, float, HostResident> tensorView = tensor.SubTensor({1, 0}, {1, 2});

    EXPECT_EQ(2, tensorView.Dimensions().rows());

    EXPECT_EQ(1, tensorView.DimensionSize(0));  EXPECT_EQ(1, tensorView.Dimensions()(0));
    EXPECT_EQ(2, tensorView.DimensionSize(1));  EXPECT_EQ(2, tensorView.Dimensions()(1));

    EXPECT_EQ(1, tensorView(0, 0));
    EXPECT_EQ(4, tensorView(0, 1));

    tensorView = tensor.SubTensor({1,0}, {1,2});

    EXPECT_EQ(2, tensorView.Dimensions().rows());

    EXPECT_EQ(1, tensorView.DimensionSize(0));  EXPECT_EQ(1, tensorView.Dimensions()(0));
    EXPECT_EQ(2, tensorView.DimensionSize(1));  EXPECT_EQ(2, tensorView.Dimensions()(1));

    EXPECT_EQ(1, tensorView(0, 0));
    EXPECT_EQ(4, tensorView(0, 1));

    // Test Slice
    VectorView<float> row = tensor.Slice<1>(1);

    EXPECT_EQ(3, row.DimensionSize(0));

    EXPECT_EQ(3, row(0));
    EXPECT_EQ(4, row(1));
    EXPECT_EQ(5, row(2));

    ConstVectorView<float> col = static_cast<const Image<float> &>(tensor).Slice<0>(1);

    EXPECT_EQ(3, col.DimensionSize(0));

    EXPECT_EQ(1, col(0));
    EXPECT_EQ(4, col(1));
    EXPECT_EQ(7, col(2));

}

TEST(TensorViewTest, Test3D) {

    float data[] = { 0, 1, 2,
                     3, 4, 5,

                     6, 7, 8,
                     9, 10, 11,

                    11, 12, 13,
                    14, 15, 16 };

    Tensor<3, float, HostResident> tensor({3, 2, 3}, data);

    // Test SubTensor
    VolumeView<float> tensorView = tensor.SubTensor({1,0,0}, {2, 2, 2});

    EXPECT_EQ(2, tensorView.DimensionSize(0));
    EXPECT_EQ(2, tensorView.DimensionSize(1));
    EXPECT_EQ(2, tensorView.DimensionSize(2));

    EXPECT_EQ(1, tensorView(0,0,0));
    EXPECT_EQ(2, tensorView(1,0,0));
    EXPECT_EQ(4, tensorView(0,1,0));
    EXPECT_EQ(7, tensorView(0,0,1));
    EXPECT_EQ(11, tensorView(1,1,1));

    // Test Slice
    const ImageView<float> plane = tensor.Slice<1>(1);

    EXPECT_EQ(3, plane.DimensionSize(0));
    EXPECT_EQ(3, plane.DimensionSize(1));

    EXPECT_EQ(3, plane(0, 0));
    EXPECT_EQ(5, plane(2, 0));
    EXPECT_EQ(9, plane(0, 1));
    EXPECT_EQ(16,plane(2, 2));

    // Test double slice
    ConstVectorView<float> vec = plane.Slice<0>(1);

    EXPECT_EQ(3, vec.DimensionSize(0));

    EXPECT_EQ(4, vec(0));
    EXPECT_EQ(10, vec(1));
    EXPECT_EQ(15, vec(2));

}

} // namespace