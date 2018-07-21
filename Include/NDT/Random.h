#pragma once

#include <NDT/Tensor.h>

namespace NDT {

namespace internal {

inline std::mt19937 & Generator() {
    static std::random_device randomDevice;
    static std::mt19937 generator(randomDevice());
    return generator;
}

} // namespace internal

template <uint D, typename Scalar>
inline NDT::ManagedTensor<D, Scalar> RandomNormal(const Eigen::Matrix<unsigned int,D,1> & size,
                                                  const Scalar mean = Scalar(0),
                                                  const Scalar stdDev = Scalar(1)) {

    NDT::ManagedTensor<D, Scalar> tensor(size);

    std::normal_distribution<Scalar> distribution(mean, stdDev);

    std::mt19937 & generator = internal::Generator();

    std::generate_n(tensor.Data(), tensor.Count(), [&distribution, &generator]() { return distribution(generator); });

    return tensor;

}

template <uint D, typename Scalar>
inline NDT::ManagedTensor<D, Scalar> RandomNormal(const unsigned int size,
                                                  const Scalar mean = Scalar(0),
                                                  const Scalar stdDev = Scalar(1)) {

    return RandomNormal<D, Scalar>(Eigen::Matrix<unsigned int,1,1>(size), mean, stdDev);

}

} // namespace NDT