#include <gtest/gtest.h>

#include <NDT/Tensor.h>

namespace {

using namespace NDT;

TEST(TensorElementAccessTest, Test1D) {

    float data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    Tensor<1, float, HostResident> tensor(10, data);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(i, tensor(i));
    }

    tensor(7) = 3;

    EXPECT_EQ(3, tensor(7));

}

TEST(TensorElementAccessTest, Test2D) {

    uint64_t data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    Tensor<2, uint64_t, HostResident> tensor({3, 3}, data);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(i + 3 * j, tensor(i, j));
        }
    }

    tensor(1,2) = 13;

    EXPECT_EQ(13, tensor(1,2));

}

TEST(TensorElementAccessTest, Test3D) {

    int data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    Tensor<3, int, HostResident> tensor({2, 4, 2}, data);

    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 2; ++i) {
                EXPECT_EQ(i + 2 * (j + 4 * k), tensor(i, j, k));
            }
        }
    }

    tensor(1,2,0) = -1;

    EXPECT_EQ(-1, tensor(1,2,0));

}

TEST(TensorElementAccessTest, Test4D) {

    double data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    Tensor<4, double, HostResident> tensor({2, 2, 2, 2}, data);

    for (int l = 0; l < 2; ++l) {
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                    EXPECT_EQ(i + 2 * (j + 2 * (k + 2 * l)), tensor(i, j, k,l));
                }
            }
        }
    }

    tensor(1,0,0,1) = 2.5;

    EXPECT_EQ(2.5, tensor(1,0,0,1));

}

} // namespace