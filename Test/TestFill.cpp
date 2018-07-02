#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(FillTest, TestHost) {

    float data[10];

    NDT::Tensor1f tensor(10, data);

    tensor.Fill(0.5f);

    for (int i = 0; i < tensor.Length(); ++i) {
        ASSERT_EQ(0.5f, tensor(i));
    }

}

TEST(FillTest, TestManagedHost) {

    NDT::ManagedTensor2i tensor({2, 5});

    tensor.Fill(18);

    for (int i = 0; i < tensor.Count(); ++i) {
        ASSERT_EQ(18, tensor.Data()[i]);
    }

}

TEST(FillTest, TestManagedDevice) {

    NDT::ManagedDeviceTensor3d tensor({2, 3, 4});

    tensor.Fill(-2.4);

    NDT::ManagedTensor3d hTensor(tensor.Dimensions());

    hTensor.CopyFrom(tensor);

    for (int i = 0; i < hTensor.Count(); ++i) {
        ASSERT_EQ(-2.4, hTensor.Data()[i]);
    }

}


} // namespace