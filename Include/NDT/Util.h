#pragma once

#include <NDT/Tensor.h>

namespace NDT {

template <uint D, typename T, Residency R, bool Const>
inline Tensor<D+1,T,R,Const> ExpandDims(NDT::Tensor<D,T,R,Const> & tensor, const int index) {
    return Tensor<D+1,T,R,Const>(
            (Eigen::Matrix<unsigned int, D + 1, 1>() <<
                    tensor.Dimensions().head(index), 1, tensor.Dimensions().tail(D-index)).finished(),
        tensor.Data());
}

template <uint D, typename T>
inline ManagedTensor<D,T,HostResident> Zeros(const Eigen::Matrix<uint,D,1> & size) {

    ManagedTensor<D,T,HostResident> tensor(size);

    tensor.Fill(T(0));

    return tensor;

}

template <typename T>
inline ManagedTensor<1,T,HostResident> Zeros(const uint length) {

    return Zeros<1,T>(Eigen::Matrix<uint,1,1>(length));

}

template <typename Derived>
inline
typename std::enable_if<TensorTraits<Derived>::R == HostResident,
        ManagedTensor<TensorTraits<Derived>::D,
                      typename TensorTraits<Derived>::T,
                      HostResident> >::type ZerosLike(const TensorBase<Derived> & other) {

    using T = typename TensorTraits<Derived>::T;

    ManagedTensor<TensorTraits<Derived>::D, T, HostResident> tensor(other.Dimensions());

    tensor.Fill(T(0));

    return tensor;

}

} // namespace NDT