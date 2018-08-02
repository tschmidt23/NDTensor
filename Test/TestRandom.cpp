#include <gtest/gtest.h>

#include <NDT/Random.h>
#include <NDT/Util.h>

namespace {

using namespace NDT;

TEST(RandomTest, TestRandomNormal) {

    const float mean(6.7f);
    const float stdDev(0.4f);

    ManagedVector<float> values = RandomNormal<1,float>(10000, mean, stdDev);

    const float sampleMean = Mean(values);

    ASSERT_NEAR(mean, sampleMean, 1e-2);

}

TEST(RandomTest, TestRandomPermutation) {

    ManagedVector<int> permutation = RandomPermutation(100);

    // check that everything is here
    for (int i = 0; i < permutation.Length(); ++i) {
        ASSERT_LT(std::distance(permutation.Data(),
                                std::find(permutation.Data(), permutation.Data() + permutation.Length(), i)),
                  permutation.Length());
    }

    {
        int count345, count354, count435, count453, count534, count543;
        count345 = count354 = count435 = count453 = count534 = count543 = 0;

        static constexpr int nTrials = 5000;

        for (int i = 0; i < nTrials; ++i) {

            float data[3] = { 3.f, 4.f, 5.f };

            Vector<float> vec(3, data);

            RandomPermute(vec);

            if (vec(0) == 3.f) {
                if (vec(1) == 4.f) {
                    ASSERT_EQ(5.f, vec(2));
                    ++count345;
                } else {
                    ASSERT_EQ(5.f, vec(1));
                    ASSERT_EQ(4.f, vec(2));
                    ++count354;
                }
            } else if (vec(0) == 4.f) {
                if (vec(1) == 3.f) {
                    ASSERT_EQ(5.f, vec(2));
                    ++count435;
                } else {
                    ASSERT_EQ(5.f, vec(1));
                    ASSERT_EQ(3.f, vec(2));
                    ++count453;
                }
            } else {
                ASSERT_EQ(5.f, vec(0));
                if (vec(1) == 3.f) {
                    ASSERT_EQ(4.f, vec(2));
                    ++count534;
                } else {
                    ASSERT_EQ(4.f, vec(1));
                    ASSERT_EQ(3.f, vec(2));
                    ++count543;
                }
            }

        }

        ASSERT_EQ(nTrials, count345 + count354 + count435 + count453 + count534 + count543);

        static constexpr int countThreshold = 750;

        ASSERT_GE(count345, countThreshold);
        ASSERT_GE(count354, countThreshold);
        ASSERT_GE(count435, countThreshold);
        ASSERT_GE(count453, countThreshold);
        ASSERT_GE(count534, countThreshold);
        ASSERT_GE(count543, countThreshold);

    }

}

} // namespace