#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(TensorDimensionsTest, DimensionsTest1) {

    Tensor<1,float,HostResident> T(5, nullptr);

    EXPECT_EQ(1, T.Dimensions().rows());

    EXPECT_EQ(5, T.DimensionSize(0));  EXPECT_EQ(5, T.Dimensions()(0));  EXPECT_EQ(5, T.Length());

    EXPECT_EQ(5, T.Count());

    const auto strides = T.Strides();
    EXPECT_EQ(1, strides(0));

}

TEST(TensorDimensionsTest, DimensionsTest2) {

    Tensor<2,int,DeviceResident> T({6, 3}, nullptr);

    EXPECT_EQ(2, T.Dimensions().rows());

    EXPECT_EQ(6, T.DimensionSize(0));  EXPECT_EQ(6, T.Dimensions()(0));  EXPECT_EQ(6, T.Width());
    EXPECT_EQ(3, T.DimensionSize(1));  EXPECT_EQ(3, T.Dimensions()(1));  EXPECT_EQ(3, T.Height());

    EXPECT_EQ(6 * 3, T.Count());

    const auto strides = T.Strides();
    EXPECT_EQ(1, strides(0));
    EXPECT_EQ(6, strides(1));

}

TEST(TensorDimensionsTest, DimensionsTest3) {

    Tensor<3,double,HostResident,true> T({7,15,8}, nullptr);

    EXPECT_EQ(3, T.Dimensions().rows());

    EXPECT_EQ( 7, T.DimensionSize(0));  EXPECT_EQ( 7, T.Dimensions()(0));
    EXPECT_EQ(15, T.DimensionSize(1));  EXPECT_EQ(15, T.Dimensions()(1));
    EXPECT_EQ( 8, T.DimensionSize(2));  EXPECT_EQ( 8, T.Dimensions()(2));

    EXPECT_EQ(7 * 15 * 8, T.Count());

    const auto strides = T.Strides();
    EXPECT_EQ(1, strides(0));
    EXPECT_EQ(7, strides(1));
    EXPECT_EQ(7 * 15, strides(2));

}

TEST(TensorDimensionsTest, DimensionsTest4) {

    Tensor<4,uint,DeviceResident,true> T({1,9,5,3}, nullptr);

    EXPECT_EQ(4, T.Dimensions().rows());

    EXPECT_EQ( 1, T.DimensionSize(0));  EXPECT_EQ( 1, T.Dimensions()(0));
    EXPECT_EQ( 9, T.DimensionSize(1));  EXPECT_EQ( 9, T.Dimensions()(1));
    EXPECT_EQ( 5, T.DimensionSize(2));  EXPECT_EQ( 5, T.Dimensions()(2));
    EXPECT_EQ( 3, T.DimensionSize(3));  EXPECT_EQ( 3, T.Dimensions()(3));

    EXPECT_EQ(1 * 9 * 5 * 3, T.Count());

    const auto strides = T.Strides();
    EXPECT_EQ(1, strides(0));
    EXPECT_EQ(1, strides(1));
    EXPECT_EQ(1 * 9, strides(2));
    EXPECT_EQ(1 * 9 * 5, strides(3));

}

} // namespace