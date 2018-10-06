#pragma once

#include <type_traits>

#include <Eigen/Core>

namespace NDT {

namespace internal {


// Check if this is a vector of a particular length
template <typename Derived, int D>
struct IsVectorType {

    static constexpr bool Value = Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1;

};

// Check if this is a vector of any length
template <typename Derived>
struct IsVectorType<Derived, -1> {

    static constexpr bool Value = Eigen::internal::traits<Derived>::ColsAtCompileTime == 1;

};


template <typename Derived, int D>
struct IsRealVectorType {

    static constexpr bool Value = IsVectorType<Derived, D>::Value &&
                                  std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value;

};

template <typename Derived, int D>
struct IsIntegralVectorType {

    static constexpr bool Value = IsVectorType<Derived, D>::Value &&
                                  std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value;

};


} // namespace internal

} // namespace NDT