#pragma once

#include <NDT/Tensor.h>
#include <NDT/TensorBase.h>

namespace NDT {

template <uint D, typename T, Residency R, bool Const>
class TensorView : public TensorBase<TensorView<D, T, R, Const> > {
public:

    using BaseT = TensorBase<TensorView<D,T,R,Const> >;
    using DimT = typename BaseT::DimT;
    using IdxT = typename BaseT::IdxT;

    friend BaseT;

    TensorView(const Tensor<D, T, R, Const> & tensor, const Eigen::Matrix<DimT,D,1,Eigen::DontAlign> & strides)
            : tensor_(tensor), strides_(strides) { }

    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0) const {
        return tensor_.Data()[d0 * strides_[0]];
    }

    template <int D2 = D, typename std::enable_if<D2 == 1 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0) {
        return tensor_.Data()[d0 * strides_[0]];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0, const IdxT d1) const {
        return tensor_.Data()[d0 * strides_[0] + d1 * strides_[1]];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0, const IdxT d1) {
        return tensor_.Data()[d0 * strides_[0] + d1 * strides_[1]];
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0, const IdxT d1, const IdxT d2) const {
        return tensor_.Data()[d0 * strides_[0] + d1 * strides_[1] + d2 * strides_[2]];
    }

    template <int D2 = D, typename std::enable_if<D2 == 3 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0, const IdxT d1, const IdxT d2) {
        return tensor_.Data()[d0 * strides_[0] + d1 * strides_[1] + d2 * strides_[2]];
    }

    template <typename Derived,
              typename std::enable_if<internal::IsIntegralVectorType<Derived, D>::Value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const Eigen::MatrixBase<Derived> & indices) {
        return tensor_.Data()[indices.dot(strides_)];
    }

    template <typename Derived,
            typename std::enable_if<internal::IsIntegralVectorType<Derived, D>::Value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const Eigen::MatrixBase<Derived> & indices) const {
        return tensor_.Data()[indices.dot(strides_)];
    }

    template <typename Derived>
    inline void CopyFrom(const TensorBase<Derived> & other) {
        internal::SliceCopier::Copy(*this, other);
    }

private:

    // -=-=-=-=-=-=- sizing functions -=-=-=-=-=-=-
    inline __NDT_CUDA_HD_PREFIX__ DimT DimensionSizeImpl(const IdxT dim) const {
        return tensor_.DimensionSize(dim);
    }

    inline __NDT_CUDA_HD_PREFIX__ const Eigen::Matrix<DimT,D,1,Eigen::DontAlign> & DimensionsImpl() const {
        return tensor_.Dimensions();
    }

    Tensor<D, T, R, Const> tensor_;
    Eigen::Matrix<DimT,D,1,Eigen::DontAlign> strides_;

};

// -=-=-=-=- traits -=-=-=-=-
template <uint D_, typename T_, Residency R_, bool Const_>
struct TensorTraits<TensorView<D_, T_, R_, Const_> > {
    static constexpr uint D = D_;
    using T = T_;
    static constexpr Residency R = R_;
    static constexpr bool Const = Const_;
};

// -=-=-=-=- typedefs -=-=-=-=-
#define __NDT_TENSOR_VIEW_TYPEDEFS___(i, type, appendix) \
    typedef TensorView<i,type,HostResident> TensorView##i##appendix; \
    typedef TensorView<i,type,DeviceResident> DeviceTensorView##i##appendix; \
    typedef TensorView<i,type,HostResident,true> ConstTensorView##i##appendix; \
    typedef TensorView<i,type,DeviceResident,true> ConstDeviceTensorView##i##appendix

#define __NDT_TENSOR_VIEW_TYPEDEFS__(type, appendix) \
    __NDT_TENSOR_VIEW_TYPEDEFS___(1, type, appendix); \
    __NDT_TENSOR_VIEW_TYPEDEFS___(2, type, appendix); \
    __NDT_TENSOR_VIEW_TYPEDEFS___(3, type, appendix); \
    __NDT_TENSOR_VIEW_TYPEDEFS___(4, type, appendix); \
    __NDT_TENSOR_VIEW_TYPEDEFS___(5, type, appendix)

__NDT_TENSOR_VIEW_TYPEDEFS__(float,f);
__NDT_TENSOR_VIEW_TYPEDEFS__(double,d);
__NDT_TENSOR_VIEW_TYPEDEFS__(int,i);
__NDT_TENSOR_VIEW_TYPEDEFS__(unsigned int,ui);
__NDT_TENSOR_VIEW_TYPEDEFS__(unsigned char,uc);

#define __NDT_TENSOR_VIEW_DIMENSIONAL_ALIAS__(dimension, alias) \
    template <typename Scalar> \
    using alias##View = TensorView<dimension,Scalar,HostResident>; \
    \
    template <typename Scalar> \
    using Device##alias##View = TensorView<dimension,Scalar,DeviceResident>; \
    \
    template <typename Scalar> \
    using Const##alias##View = TensorView<dimension,Scalar,HostResident,true>; \
    \
    template <typename Scalar> \
    using ConstDevice##alias##View = TensorView<dimension,Scalar,DeviceResident,true>


__NDT_TENSOR_VIEW_DIMENSIONAL_ALIAS__(1,Vector);
__NDT_TENSOR_VIEW_DIMENSIONAL_ALIAS__(2,Image);
__NDT_TENSOR_VIEW_DIMENSIONAL_ALIAS__(3,Volume);



} // namespace NDT