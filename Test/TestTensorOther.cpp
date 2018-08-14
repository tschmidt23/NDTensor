#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(TensorOtherTest, TestMinMax) {

    float data[5] = { -1.f, 4.f, -0.6f, 12.f, 0.1f };

    Vector<float> vec(5, data);

    ASSERT_EQ(-1.f, vec.Min());

    ASSERT_EQ(12.f, vec.Max());

    ASSERT_EQ(-1.f, vec.MinMax().first);

    ASSERT_EQ(12.f, vec.MinMax().second);

}


}