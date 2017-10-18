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

} // namespace NDT