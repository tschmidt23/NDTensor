#include <gtest/gtest.h>

#include <NDT/Tensor.h>
#include <NDT/TensorView.h>

namespace {

using namespace NDT;

TEST(TensorViewTest, Test1D) {

    int data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    Tensor<1, int, HostResident> tensor(10, data);

    TensorView<1, int, HostResident> tensorView = tensor.Slice(3, 4);

    EXPECT_EQ(1, tensorView.Dimensions().rows());

    EXPECT_EQ(4, tensorView.DimensionSize(0));  EXPECT_EQ(4, tensorView.Dimensions()(0));

    EXPECT_EQ(3, tensorView(0));
    EXPECT_EQ(4, tensorView(1));
    EXPECT_EQ(5, tensorView(2));
    EXPECT_EQ(6, tensorView(3));

}

TEST(TensorViewTest, Test2D) {

    float data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    Tensor<2, float, HostResident> tensor({3, 3}, data);

    TensorView<2, float, HostResident> tensorView = tensor.Slice(1, 0, 1, 2);

    EXPECT_EQ(2, tensorView.Dimensions().rows());

    EXPECT_EQ(1, tensorView.DimensionSize(0));  EXPECT_EQ(1, tensorView.Dimensions()(0));
    EXPECT_EQ(2, tensorView.DimensionSize(1));  EXPECT_EQ(2, tensorView.Dimensions()(1));

    EXPECT_EQ(1, tensorView(0, 0));
    EXPECT_EQ(4, tensorView(0, 1));

}

} // namespace