#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

TEST(ManagedTensorTest, TestAlloc) {

    NDT::ManagedTensor3i tensor( { 4, 2, 6 } );

    ASSERT_EQ(4, tensor.DimensionSize(0));

    ASSERT_EQ(2, tensor.DimensionSize(1));

    ASSERT_EQ(6, tensor.DimensionSize(2));

    ASSERT_EQ(4 * 2 * 6, tensor.Count());

}

TEST(ManagedTensorTest, TestResize) {

    NDT::ManagedTensor2d tensor({3, 7});

    tensor.Resize({4, 7});

    ASSERT_EQ(4, tensor.DimensionSize(0));

    ASSERT_EQ(7, tensor.DimensionSize(1));

}

} // namespace