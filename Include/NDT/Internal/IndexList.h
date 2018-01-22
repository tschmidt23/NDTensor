#pragma once

namespace NDT {

namespace internal {

template <typename T, int D>
struct IndexList {
    T head;
    IndexList<T, D - 1> tail;

    template <typename Derived>
    inline __NDT_CUDA_HD_PREFIX__ IndexList(const Eigen::MatrixBase <Derived> & indices)
            : head(indices(0)), tail(indices.template tail<D - 1>()) {}

//    inline __NDT_CUDA_HD_PREFIX__ IndexList(const Eigen::Matrix<T,D,1> & indices)
//        : head(indices(0)), tail(indices.template tail<D-1>()) { }

//    template <int D2>
//    __NDT_CUDA_HD_PREFIX__
//    inline IndexList(const Eigen::VectorBlock<const Eigen::Matrix<T,D2,1>,D> & indices))

    __NDT_CUDA_HD_PREFIX__
    inline T

    sum() const {
        return head + tail.sum();
    }

    __NDT_CUDA_HD_PREFIX__
    inline T

    product() const {
        return head * tail.product();
    }

};

template <typename T>
struct IndexList<T, 0> {

    template <typename Derived>
    __NDT_CUDA_HD_PREFIX__
    inline IndexList(const Eigen::MatrixBase <Derived> & indices) {}

    __NDT_CUDA_HD_PREFIX__
    inline T

    sum() const {
        return 0;
    }

    __NDT_CUDA_HD_PREFIX__
    inline T

    product() const {
        return 1;
    }

};

template <typename T>
inline __NDT_CUDA_HD_PREFIX__ IndexList<T, 1>

IndexList1(const T i0) {
    return {i0, IndexList < T, 0 > ()};
}

template <typename T>
inline __NDT_CUDA_HD_PREFIX__ IndexList<T, 2>

IndexList2(const T i0, const T i1) {
    return {i0, IndexList1(i1)};
}

template <typename T>
inline __NDT_CUDA_HD_PREFIX__ IndexList<T, 3>

IndexList3(const T i0, const T i1, const T i2) {
    return {i0, IndexList2(i1, i2)};
}

template <typename T>
inline __NDT_CUDA_HD_PREFIX__ IndexList<T, 4>

IndexList4(const T i0, const T i1, const T i2, const T i3) {
    return {i0, IndexList3(i1, i2, i3)};
}

} // namespace internal

} // namespace NDT