#pragma once

#include <NDT/Tensor.h>

#include <NDT/Internal/EigenHelpers.h>

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

    tensor.Fill(internal::ZeroType<T>::Value());

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

    tensor.Fill(internal::ZeroType<T>::Value());

    return tensor;

}

// TODO: duplicated code
template <uint D, typename T>
inline ManagedTensor<D,T,HostResident> Ones(const Eigen::Matrix<uint,D,1> & size) {

    ManagedTensor<D,T,HostResident> tensor(size);

    tensor.Fill(internal::OneType<T>::Value());

    return tensor;

}


template <typename T>
inline ManagedTensor<1,T,HostResident> Ones(const uint length) {

    return Ones<1,T>(Eigen::Matrix<uint,1,1>(length));

}

template <typename Derived>
inline
typename std::enable_if<TensorTraits<Derived>::R == HostResident,
        ManagedTensor<TensorTraits<Derived>::D,
        typename TensorTraits<Derived>::T,
        HostResident> >::type OnesLike(const TensorBase<Derived> & other) {

    using T = typename TensorTraits<Derived>::T;

    ManagedTensor<TensorTraits<Derived>::D, T, HostResident> tensor(other.Dimensions());

    tensor.Fill(internal::OneType<T>::Value());

    return tensor;

}



template <typename T = int>
inline
typename std::enable_if<std::is_integral<T>::value, ManagedVector<T> >::type
ARange(const T start, const T end, const T step = T(1)) {

    assert(end > start);

    ManagedVector<T> vec((end - start) / step);

    std::generate_n(vec.Data(), vec.Count(), [step, n = start]() mutable {
        T returnVal = n;
        n += step;
        return returnVal;
    });

    return vec;

}


template <typename T = int>
inline
typename std::enable_if<std::is_integral<T>::value, ManagedVector<T> >::type
ARange(const int end) {
    return ARange(0, end);
}

template <typename Derived>
inline
typename std::enable_if<TensorTraits<Derived>::R == HostResident,
        typename TensorTraits<Derived>::T>::type Mean(const TensorBase<Derived> & tensor) {

    using T = typename TensorTraits<Derived>::T;

    return Sum(tensor) / tensor.Count();

}


template <typename Derived>
inline
typename std::enable_if<TensorTraits<Derived>::R == HostResident,
        typename TensorTraits<Derived>::T>::type Sum(const TensorBase<Derived> & tensor) {

    using T = typename TensorTraits<Derived>::T;

    return std::accumulate(tensor.Data(), tensor.Data() + tensor.Count(), internal::ZeroType<T>::Value());

}

template <typename T, bool ConstA, bool ConstB>
inline T Dot(const Tensor<1, T, HostResident, ConstA> & vectorA,
                  const Tensor<1, T, HostResident, ConstB> & vectorB) {
    assert(vectorA.Length() == vectorB.Length());
    return std::inner_product(vectorA.Data(), vectorA.Data() + vectorA.Count(), vectorB.Data(), internal::ZeroType<T>::Value());
}

template <uint DIn, typename T, Residency R, bool Const, typename Derived>
inline
typename std::enable_if<internal::IsIntegralVectorType<Derived, -1>::Value,
        typename NDT::Tensor<Eigen::internal::traits<Derived>::RowsAtCompileTime, T, R, Const> >::type
Reshape(Tensor<DIn, T, R, Const> & tensor, const Eigen::MatrixBase<Derived> & dimensionSpec) {

    constexpr int DOut = Eigen::MatrixBase<Derived>::RowsAtCompileTime;
    using DimT = typename Tensor<DOut, T, R, Const>::DimT;

    int unspecifiedDimension = -1;
    int product = 1;

    Eigen::Matrix<DimT, DOut, 1> dimensions = dimensionSpec.template cast<DimT>();

    for (int i = 0; i < DOut; ++i) {

        if (dimensionSpec(i) < 0) {

            if (unspecifiedDimension >= 0) {
                throw std::runtime_error("only one dimension can be left unspecified");
            }

            unspecifiedDimension = i;

        } else {

            product *= dimensions(i);

        }

    }

    if (unspecifiedDimension >= 0) {

        if (tensor.Count() % product) {

            throw std::runtime_error("not evenly divisible");

        }

        dimensions(unspecifiedDimension) = tensor.Count() / product;

    } else if (product != tensor.Count()) {

        throw std::runtime_error("specified dimensions yields different tensor size (" +
            std::to_string(product) + " vs " + std::to_string(tensor.Count()) + ")");

    }

    return Tensor<DOut, T, R, Const>(dimensions, tensor.Data());

}

} // namespace NDT