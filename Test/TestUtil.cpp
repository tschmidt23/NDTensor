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

}