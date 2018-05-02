#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(TensorSliceCopyTest, Test2D) {


    float data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    Tensor<2, float, HostResident> tensor({3, 3}, data);

    TensorView<2, float, HostResident> tensorView = tensor.SubTensor({1,1},{2,2});

    ManagedTensor<2, float, HostResident> tensorCopy({2, 2});

    tensorCopy.CopyFrom(tensorView);

    EXPECT_EQ(4, tensorCopy(0,0));
    EXPECT_EQ(5, tensorCopy(1,0));
    EXPECT_EQ(7, tensorCopy(0,1));
    EXPECT_EQ(8, tensorCopy(1,1));

    TensorView<2, float, HostResident> tensorView2 = tensorCopy.SubTensor({0,1}, {2, 1});

    TensorView<2, float, HostResident> tensorView3 = tensor.SubTensor({0,0}, {2, 1});

    tensorView3.CopyFrom(tensorView2);

    EXPECT_EQ(7, tensorView3(0,0));
    EXPECT_EQ(7, tensor(0,0));
    EXPECT_EQ(8, tensorView3(1,0));
    EXPECT_EQ(8, tensor(1,0));

}

} // namespace