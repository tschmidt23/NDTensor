#pragma once

#include <Eigen/Core>

namespace NDT {

namespace internal {

// DevolvingMatrixType is a helper class used to statically define a
// matrix type, which essentially a pass-through definition to the
// Eigen::Matrix class. However, in the case of a 1 x 1 matrix of type
// Scalar, the defined Type devolves to a Scalar, skipping the Eigen
// wrapper.
template <typename Scalar, int R, int C, int Options>
struct DevolvingMatrixType {

    using Type = Eigen::Matrix<Scalar, R, C, Options>;

};

template <typename Scalar, int Options>
struct DevolvingMatrixType<Scalar, 1, 1, Options> {

    using Type = Scalar;

};


template <typename T>
struct ZeroType {
    static inline T Value() { return T(0); }
};

template <typename T>
struct OneType {
    static inline T Value() { return T(1); }
};

template <typename T, int M, int N, int O, int MR, int MC>
struct ZeroType<Eigen::Matrix<T, M, N, O, MR, MC> > {
    static inline Eigen::Matrix<T, M, N, O, MR, MC> Value() { return Eigen::Matrix<T, M, N, O, MR, MC>::Zero(); }
};

template <typename T, int M, int N, int O, int MR, int MC>
struct OneType<Eigen::Matrix<T, M, N, O, MR, MC> > {
    static inline Eigen::Matrix<T, M, N, O, MR, MC> Value() { return Eigen::Matrix<T, M, N, O, MR, MC>::Ones(); }
};



} // namespace internal

} // namespace NDT