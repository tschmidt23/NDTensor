#include <gtest/gtest.h>

#include <NDT/Random.h>
#include <NDT/Util.h>

namespace {

using namespace NDT;

TEST(RandomTest, TestRandomNormal) {

    const float mean(6.7f);
    const float stdDev(0.4f);

    ManagedVector<float> values = RandomNormal<1,float>(Eigen::Matrix<unsigned int, 1, 1>(10000), mean, stdDev);

    const float sampleMean = Mean(values);

    std::cout << mean << " vs " << sampleMean << std::endl;

    ASSERT_NEAR(mean, sampleMean, 1e-2);

}


} // namespace