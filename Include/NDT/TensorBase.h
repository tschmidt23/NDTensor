#pragma once

#ifndef __NDT_NO_CUDA__
#include <cuda_runtime.h>
#define __NDT_CUDA_HD_PREFIX__ __host__ __device__
#else
#define __NDT_CUDA_HD_PREFIX__
#endif // __NDT_NO_CUDA__

#include <NDT/Internal/Slicing.h>

namespace NDT {

template <typename Derived>
struct TensorTraits;

template <typename Derived>
class TensorBase {
public:

    using DimT = unsigned int;
    using IdxT = unsigned int;

    static constexpr uint D = TensorTraits<Derived>::D;
    using T = typename TensorTraits<Derived>::T;
    static constexpr bool Const = TensorTraits<Derived>::Const;

    // -=-=-=-=-=-=- sizing functions -=-=-=-=-=-=-
    inline __NDT_CUDA_HD_PREFIX__ std::size_t Count() const {
        return static_cast<const Derived *>(this)->CountImpl();
    }

    inline __NDT_CUDA_HD_PREFIX__ DimT DimensionSize(const IdxT dim) const {
        return static_cast<const Derived *>(this)->DimensionSizeImpl(dim);
    }

    inline __NDT_CUDA_HD_PREFIX__ const Eigen::Matrix<DimT, D, 1, Eigen::DontAlign> & Dimensions() const {
        return static_cast<const Derived *>(this)->DimensionsImpl();
    }

    // -=-=-=-=-=-=- data functions -=-=-=-=-=-=-
    template <typename U = T, typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T * Data() {
        return static_cast<Derived *>(this)->DataImpl();
    }

    inline __NDT_CUDA_HD_PREFIX__ const T * Data() const {
        return static_cast<const Derived *>(this)->DataImpl();
    }

    // -=-=-=-=-=-=- indexing functions -=-=-=-=-=-=-
    template <typename ... ArgTs>
    inline __NDT_CUDA_HD_PREFIX__ const T & operator()(const ArgTs ... args) const {
        return static_cast<const Derived *>(this)->Element(args...);
    }

    template <typename ... ArgTs, typename U = T, typename std::enable_if<!Const && sizeof(U),int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & operator()(const ArgTs ... args) {
        return static_cast<Derived *>(this)->Element(args...);
    }

//    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
//    inline __NDT_CUDA_HD_PREFIX__ const T & operator()(const IdxT d0) const {
//        return static_cast<const Derived *>(this)->Element(d0);
//    }
//
//    template <int D2 = D, typename std::enable_if<D2 == 1 && !Const, int>::type = 0>
//    inline __NDT_CUDA_HD_PREFIX__ T & operator()(const IdxT d0) {
//        return static_cast<Derived *>(this)->Element(d0);
//    }

    // TODO: should this be exposed?
    inline __NDT_CUDA_HD_PREFIX__ Eigen::Matrix<DimT, D, 1, Eigen::DontAlign> Strides() const {
        return static_cast<const Derived *>(this)->StridesImpl();
    }

    template <int Axis>
    inline typename TensorTraits<Derived>::SliceT Slice(const IdxT slice) {
        return typename TensorTraits<Derived>::SliceT(
                typename TensorTraits<Derived>::SliceTensorT(
                        internal::AxisDropper<D, Axis>::Drop(Dimensions()),
                        &(*this)(internal::AxisEmplacer<D, Axis>::Emplace(slice))),
                internal::AxisDropper<D, Axis>::Drop(this->Strides())
        );
    }

    template <int Axis>
    inline typename TensorTraits<Derived>::ConstSliceT Slice(const IdxT slice) const {
        return typename TensorTraits<Derived>::ConstSliceT(
                typename TensorTraits<Derived>::ConstSliceTensorT(
                        internal::AxisDropper<D, Axis>::Drop(Dimensions()),
                        &(*this)(internal::AxisEmplacer<D, Axis>::Emplace(slice))),
                internal::AxisDropper<D, Axis>::Drop(this->Strides())
        );
    }

    // -=-=-=-=-=-=- conversion to base -=-=-=-=-=-=-
    inline Derived & Downcast() {
        return static_cast<Derived &>(*this);
    }

    inline const Derived & Downcast() const {
        return static_cast<const Derived &>(*this);
    }


protected:


    TensorBase() = default;
    TensorBase(const TensorBase<Derived> &) = default;
    TensorBase<Derived> & operator=(const TensorBase<Derived> &) = default;

};

} // namespace NDT
