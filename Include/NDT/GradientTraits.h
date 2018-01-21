#pragma once

#include <Eigen/Core>

namespace NDT {

namespace internal {

// in the general case the gradient is a row vector with the number of parameters
// equal to the dimensionality of the tensor
template <typename Scalar, uint D>
struct GradientTraits {

    using GradientType = Eigen::Matrix<Scalar, 1, D>;

};

// in the special case of a 1-dimensional tensor, the gradient is also a scalar
template <typename Scalar>
struct GradientTraits<Scalar, 1> {

    using GradientType = Scalar;

};

// in the special case of vector-valued data, the gradient is a matrix
template <typename Scalar, uint VecLength, int Options, uint D>
struct GradientTraits<Eigen::Matrix<Scalar,VecLength,1,Options>, D> {

    using GradientType = Eigen::Matrix<Scalar,VecLength,D,Options>;

};

} // namespace internal

} // namespace NDT