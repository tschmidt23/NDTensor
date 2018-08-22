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

TEST(TensorOtherTest, TestCopy) {

    unsigned short data[8] = { 14, 1, 8, 19, 0, 6, 11, 9 };

    Tensor<2, unsigned short> tensor( 4, 2, data );

    {

        NDT::ManagedTensor<2, unsigned short> hCopy = tensor.Copy();

        ASSERT_EQ(tensor.DimensionSize(0), hCopy.DimensionSize(0));
        ASSERT_EQ(tensor.DimensionSize(1), hCopy.DimensionSize(1));

        for (int j = 0; j < tensor.DimensionSize(1); ++j) {
            for (int i = 0; i < tensor.DimensionSize(0); ++i) {
                ASSERT_EQ(tensor(i, j), hCopy(i, j));
            }
        }

    }

    {

        NDT::ManagedTensor<2, unsigned short, DeviceResident> dCopy = tensor.Copy<DeviceResident>();

        ASSERT_EQ(tensor.DimensionSize(0), dCopy.DimensionSize(0));
        ASSERT_EQ(tensor.DimensionSize(1), dCopy.DimensionSize(1));

        NDT::ManagedTensor<2, unsigned short> hCopy = dCopy.Copy<HostResident>();

        ASSERT_EQ(tensor.DimensionSize(0), hCopy.DimensionSize(0));
        ASSERT_EQ(tensor.DimensionSize(1), hCopy.DimensionSize(1));

        for (int j = 0; j < tensor.DimensionSize(1); ++j) {
            for (int i = 0; i < tensor.DimensionSize(0); ++i) {
                ASSERT_EQ(tensor(i, j), hCopy(i, j));
            }
        }

    }

}

}