#pragma once

#include <NDT/Internal/IndexList.h>
#include <NDT/Internal/ToType.h>

namespace NDT {

namespace internal {

// In the general case, the dimension along with the gradient is taken is indexed with a real value,
// and the gradient is determined by the local linear slope of the interpolation.
//
// TODO: this can be made more efficient. Currently, an interpolation gradient in D dimensions samples
// 2D points (the floor and ceil of each real-valued index). However, because the gradients are linear,
// the same gradient results when considering the center value and just the floor or the ceil (but not
// both) along each dimension, requiring only D+1 samples.
template <typename Scalar, int D, int I, typename IndexIType, typename ... IdxTs,
        typename std::enable_if<std::is_floating_point<IndexIType>::value, int>::type = 0>
inline Scalar InterpolationGradientAlongOneDimension(const Scalar * data,
                                                     const Eigen::Matrix<uint,D,1> & dimensions,
                                                     const std::tuple<IdxTs...> & indices,
                                                     const TypeToType<IndexIType> /*indexTypeTag*/,
                                                     const IntToType<I> /*indexTag*/) {

    // Nota bene: this relies on the C++ default rounding, i.e. round-towards-zero. It thus implicitly
    // assumes that indices will be positive, which should always be the case.
    typename TupleTypeSubstitute<I, int, IdxTs...>::Type roundedIndices = indices;
    const Scalar before = Interpolate(data, IndexList<uint,D>(dimensions.reverse()),
    TupleReverser<std::tuple<IdxTs...> >::Reverse(roundedIndices));
//    std::cout << "Index " << I << std::endl;
//    std::cout << "before: " << std::get<I>(roundedIndices) << std::endl;
//    std::cout << before << std::endl;

    std::get<I>(roundedIndices)++;
//    std::cout << "after: " << std::get<I>(roundedIndices) << std::endl;
//    std::cout << Interpolate(data, IndexList<uint,D>(dimensions.reverse()),
//                             TupleReverser<std::tuple<IdxTs...> >::Reverse(indices)) << std::endl;
    return Interpolate(data, IndexList<uint,D>(dimensions.reverse()),
                       TupleReverser<std::tuple<IdxTs...> >::Reverse(roundedIndices)) - before;
}

// In this special case, the dimension along which the interpolation gradient is taken is represented by
// an integral value. Therefore, we fall back on central difference, in effect averaging the local
// interpolation slopes in the forward and backward direction.
template <typename Scalar, int D, int I, typename IndexIType, typename ... IdxTs,
        typename std::enable_if<std::is_integral<IndexIType>::value, int>::type = 0>
inline Scalar InterpolationGradientAlongOneDimension(const Scalar * data,
                                                     const Eigen::Matrix<uint,D,1> & dimensions,
                                                     const std::tuple<IdxTs...> & indices,
                                                     const TypeToType<IndexIType> /*indexTypeTag*/,
                                                     const IntToType<I> /*indexTag*/) {

    std::tuple<IdxTs...> indicesCopy = indices;
    std::get<I>(indicesCopy)--;
    const Scalar before = Interpolate(data, IndexList<uint,D>(dimensions.reverse()),
    TupleReverser<std::tuple<IdxTs...> >::Reverse(indicesCopy));

    std::get<I>(indicesCopy) += 2;

    return (Interpolate(data, IndexList<uint,D>(dimensions.reverse()),
                        TupleReverser<std::tuple<IdxTs...> >::Reverse(indicesCopy)) - before) / 2;
}


template <int D, int I>
struct InterpolationGradientFiller {

    template <typename DataType, typename Scalar, int Options, int R, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ static inline void Fill(const DataType * data,
                                                   const Eigen::Matrix<uint,D,1> & dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   Eigen::Matrix<Scalar, R, D, Options> & gradient) {
        gradient.template block<R, 1>(0, I) =
                InterpolationGradientAlongOneDimension(data, dimensions, indices,
                                                       TypeToType<typename TypeListIndex<I,IdxTs...>::Type>(),
                                                       IntToType<I>());
        InterpolationGradientFiller<D, I+1>::Fill(data, dimensions, indices, gradient);
    }

};

template <int D>
struct InterpolationGradientFiller<D,D> {

    template <typename DataType, typename Scalar, int Options, int R, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ static inline void Fill(const DataType * /*data*/,
                                                   const Eigen::Matrix<uint,D,1> & /*dimensions*/,
                                                   const std::tuple<IdxTs...> & /*indices*/,
                                                   Eigen::Matrix<Scalar, R, D, Options> & /*gradient*/) { }

};




// In the general case, the dimension along with the gradient is taken is indexed with a real value,
// and the gradient is determined by the local linear slope of the interpolation.
//
// TODO: this can be made more efficient. Currently, an interpolation gradient in D dimensions samples
// 2D points (the floor and ceil of each real-valued index). However, because the gradients are linear,
// the same gradient results when considering the center value and just the floor or the ceil (but not
// both) along each dimension, requiring only D+1 samples.
template <typename Transformer, typename Scalar, int D, int I, typename IndexIType, typename ... IdxTs,
          typename std::enable_if<std::is_floating_point<IndexIType>::value, int>::type = 0>
inline auto InterpolationGradientAlongOneDimension(Transformer transformer,
                                                   const Scalar * data,
                                                   const Eigen::Matrix<uint,D,1> & dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   const TypeToType<IndexIType> /*indexTypeTag*/,
                                                   const IntToType<I> /*indexTag*/) -> decltype(transformer(*data)) {

    using TransformedType = decltype(transformer(*data));

    // Nota bene: this relies on the C++ default rounding, i.e. round-towards-zero. It thus implicitly
    // assumes that indices will be positive, which should always be the case.
    typename TupleTypeSubstitute<I, int, IdxTs...>::Type roundedIndices = indices;
    const TransformedType before = TransformInterpolate(data, IndexList<uint,D>(dimensions.reverse()), transformer,
                                                        TupleReverser<std::tuple<IdxTs...> >::Reverse(roundedIndices));

    std::get<I>(roundedIndices)++;

    const TransformedType after = TransformInterpolate(data, IndexList<uint,D>(dimensions.reverse()), transformer,
                                                       TupleReverser<std::tuple<IdxTs...> >::Reverse(roundedIndices));

    const TransformedType diff = after - before;

    return diff;
}

// In this special case, the dimension along which the interpolation gradient is taken is represented by
// an integral value. Therefore, we fall back on central difference, in effect averaging the local
// interpolation slopes in the forward and backward direction.
template <typename Transformer, typename Scalar, int D, int I, typename IndexIType, typename ... IdxTs,
        typename std::enable_if<std::is_integral<IndexIType>::value, int>::type = 0>
inline auto InterpolationGradientAlongOneDimension(Transformer transformer,
                                                   const Scalar * data,
                                                   const Eigen::Matrix<uint,D,1> & dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   const TypeToType<IndexIType> /*indexTypeTag*/,
                                                   const IntToType<I> /*indexTag*/) -> decltype(transformer(*data)) {

    using TransformedType = decltype(transformer(*data));

    std::tuple<IdxTs...> indicesCopy = indices;
    std::get<I>(indicesCopy)--;
    const TransformedType before = TransformInterpolate(data, IndexList<uint,D>(dimensions.reverse()), transformer,
                                               TupleReverser<std::tuple<IdxTs...> >::Reverse(indicesCopy));

    std::get<I>(indicesCopy) += 2;

    return (TransformInterpolate(data, IndexList<uint,D>(dimensions.reverse()), transformer,
                                 TupleReverser<std::tuple<IdxTs...> >::Reverse(indicesCopy)) - before) / 2;
}

template <int D, int I>
struct TransformInterpolationGradientFiller {

    template <typename Transformer, typename DataType, typename GradientType, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ static inline void Fill(Transformer transformer,
                                                   const DataType * data,
                                                   const Eigen::Matrix<uint,D,1> & dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   GradientType & gradient) {
        gradient.template col(I) =
                InterpolationGradientAlongOneDimension(transformer, data, dimensions, indices,
                                                       TypeToType<typename TypeListIndex<I,IdxTs...>::Type>(),
                                                       IntToType<I>());
        TransformInterpolationGradientFiller<D, I+1>::Fill(transformer, data, dimensions, indices, gradient);
    }

};

template <int D>
struct TransformInterpolationGradientFiller<D,D> {

    template <typename Transformer, typename DataType, typename GradientType, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ static inline void Fill(Transformer transformer,
                                                   const DataType * /*data*/,
                                                   const Eigen::Matrix<uint,D,1> & /*dimensions*/,
                                                   const std::tuple<IdxTs...> & /*indices*/,
                                                   GradientType & /*gradient*/) { }

};

} // namespace internal

} // namespace NDT