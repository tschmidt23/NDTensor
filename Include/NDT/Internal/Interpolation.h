#pragma once

namespace NDT {

namespace internal {

template <typename Scalar>
__NDT_CUDA_HD_PREFIX__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint,0> /*dimensions*/,
                          const std::tuple<> /*remainingIndices*/) {

    return *data;

}


template <typename Scalar, typename HeadT, typename std::enable_if<!std::is_floating_point<HeadT>::value && !std::is_integral<HeadT>::value, int>::type = 0, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<HeadT, IdxTs...> remainingIndices) {

    const HeadT & firstIndex = std::get<0>(remainingIndices);
    const uint i = static_cast<double>(firstIndex); // TODO: what is going on here?
    const HeadT t = firstIndex - i;

    return (1-t)*Interpolate(data + i*dimensions.tail.product(),
                             dimensions.tail,
                             GetTail(remainingIndices))
           + t * Interpolate(data + (i+1)*dimensions.tail.product(),
                             dimensions.tail,
                             GetTail(remainingIndices));

}

template <typename Scalar, typename HeadT, typename std::enable_if<std::is_floating_point<HeadT>::value, int>::type = 0, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<HeadT, IdxTs...> remainingIndices) {

    const HeadT firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const HeadT t = firstIndex - i;

    return (1-t)*Interpolate(data + i*dimensions.tail.product(),
                             dimensions.tail,
                             GetTail(remainingIndices))
           + t * Interpolate(data + (i+1)*dimensions.tail.product(),
                             dimensions.tail,
                             GetTail(remainingIndices));

}

template <typename Scalar, typename HeadT, typename std::enable_if<std::is_integral<HeadT>::value, int>::type = 0, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint, sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<HeadT, IdxTs...> remainingIndices) {

    const HeadT firstIndex = std::get<0>(remainingIndices);

    return Interpolate(data + firstIndex*dimensions.tail.product(),
                       dimensions.tail,
                       GetTail(remainingIndices));

}


template <typename Scalar, typename ValidityCheck>
__NDT_CUDA_HD_PREFIX__
inline Scalar InterpolateValidOnly(const Scalar * data,
                                   const IndexList<uint, 0> dimensions,
                                   float & totalWeight,
                                   const float thisWeight,
                                   ValidityCheck check,
                                   const std::tuple<>) {

    if (check(*data)) {

        totalWeight += thisWeight;
        return thisWeight * (*data);

    } else {

        return 0 * (*data);

    }

}

template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline Scalar InterpolateValidOnly(const Scalar * data,
                                   const IndexList<uint, sizeof...(IdxTs)+1> dimensions,
                                   float & totalWeight,
                                   const float thisWeight,
                                   ValidityCheck check,
                                   const std::tuple<float, IdxTs...> remainingIndices) {

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const float t = firstIndex - i;

    return InterpolateValidOnly(data + i*dimensions.tail.product(),
                                dimensions.tail,
                                totalWeight,
                                thisWeight * (1-t),
                                check,
                                GetTail(remainingIndices)) +
           InterpolateValidOnly(data + (i+1)*dimensions.tail.product(),
                                dimensions.tail,
                                totalWeight,
                                thisWeight * t,
                                check,
                                GetTail(remainingIndices));

}

template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline Scalar InterpolateValidOnly(const Scalar * data,
                                   const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                   float & totalWeight,
                                   const float thisWeight,
                                   ValidityCheck check,
                                   const std::tuple<int, IdxTs...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return InterpolateValidOnly(data + firstIndex*dimensions.tail.product(),
                                dimensions.tail,
                                totalWeight,
                                thisWeight,
                                check,
                                GetTail(remainingIndices));

}


template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__ inline Scalar InterpolateValidOnly(const Scalar * data,
                                                          const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                          float & totalWeight,
                                                          ValidityCheck check,
                                                          std::tuple<IdxTs...> indices) {

    // TODO: always float?
    totalWeight = 0;

    const Scalar totalValue = InterpolateValidOnly(data, dimensions, totalWeight, 1.f,
                                                   check, indices);

    return totalValue / totalWeight;

}

// this version throws away the total weight if it is not needed
template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__ inline Scalar InterpolateValidOnly(const Scalar * data,
                                                          const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                          ValidityCheck check,
                                                          std::tuple<IdxTs...> indices) {

    // TODO: always float?
    float totalWeight;

    return InterpolateValidOnly(data, dimensions, totalWeight, check, indices);

}





template <typename Scalar, typename Transformer>
__NDT_CUDA_HD_PREFIX__
inline auto TransformInterpolate(const Scalar * data,
                                 const IndexList<uint,0> /*dimensions*/,
                                 Transformer transformer,
                                 const std::tuple<> /*remainingIndices*/)  -> decltype(transformer(*data)) {

    return transformer(*data);

}

template <typename Scalar, typename Transformer, typename HeadIdxT, typename ... TailIdxTs,
          typename std::enable_if<std::is_floating_point<HeadIdxT>::value, int>::type = 0>
__NDT_CUDA_HD_PREFIX__
inline auto TransformInterpolate(const Scalar * data,
                                 const IndexList<uint,sizeof...(TailIdxTs)+1> dimensions,
                                 Transformer transformer,
                                 const std::tuple<HeadIdxT, TailIdxTs...> remainingIndices) -> decltype(transformer(*data)) {

    const HeadIdxT firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const HeadIdxT t = firstIndex - i;

    return (1-t)*TransformInterpolate(data + i*dimensions.tail.product(),
                                      dimensions.tail,
                                      transformer,
                                      GetTail(remainingIndices))
           + t * TransformInterpolate(data + (i+1)*dimensions.tail.product(),
                                      dimensions.tail,
                                      transformer,
                                      GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename HeadIdxT, typename ... TailIdxTs,
          typename std::enable_if<std::is_integral<HeadIdxT>::value, int>::type = 0>
__NDT_CUDA_HD_PREFIX__
inline /*auto*/ typename Transformer::ReturnType TransformInterpolate(const Scalar * data,
                                                                      const IndexList<uint,sizeof...(TailIdxTs)+1> dimensions,
                                                                      Transformer transformer,
                                                                      const std::tuple<HeadIdxT, TailIdxTs...> remainingIndices) /*-> decltype(tranformer(*data))*/ {

    const HeadIdxT firstIndex = std::get<0>(remainingIndices);

    return TransformInterpolate(data + firstIndex*dimensions.tail.product(),
                                dimensions.tail,
                                transformer,
                                GetTail(remainingIndices));

}


template <typename Scalar, typename Transformer, typename ValidityCheck>
__NDT_CUDA_HD_PREFIX__
inline typename Transformer::ReturnType TransformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,0> /*dimensions*/,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<>) {

    if (check(*data)) {

        totalWeight += thisWeight;
        return thisWeight * transformer(*data);

    } else {

        return 0;

    }

}

template <typename Scalar, typename Transformer, typename ValidityCheck, typename FirstIndexT, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline
typename std::enable_if<std::is_floating_point<FirstIndexT>::value,typename Transformer::ReturnType>::type TransformInterpolateValidOnly(const Scalar * data,
                                                                                                                                         const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                                                                                                         typename Transformer::ScalarType & totalWeight,
                                                                                                                                         const typename Transformer::ScalarType thisWeight,
                                                                                                                                         Transformer transformer,
                                                                                                                                         ValidityCheck check,
                                                                                                                                         const std::tuple<FirstIndexT,IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const FirstIndexT firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const typename Transformer::ScalarType t = firstIndex - i;

    return TransformInterpolateValidOnly(data + i*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight * (1-t),
                                         transformer,
                                         check,
                                         GetTail(remainingIndices)) +
           TransformInterpolateValidOnly(data + (i+1)*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight * t,
                                         transformer,
                                         check,
                                         GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline typename Transformer::ReturnType TransformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<int, IdxTs...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return TransformInterpolateValidOnly(data + firstIndex*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight,
                                         transformer,
                                         check,
                                         GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__ inline typename Transformer::ReturnType TransformInterpolateValidOnly(const Scalar * data,
                                                                                             const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                                                             Transformer transformer,
                                                                                             ValidityCheck check,
                                                                                             std::tuple<IdxTs...> indices) {

    typename Transformer::ScalarType totalWeight(0);
    const typename Transformer::ScalarType totalValue = TransformInterpolateValidOnly(data,dimensions,totalWeight,typename Transformer::ScalarType(1),
                                                                                      transformer, check, indices);

    if (totalWeight) {
        return totalValue / totalWeight;
    }

    return 0;
}

//template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
//__NDT_CUDA_HD_PREFIX__
//inline Scalar interpolateValidOnly(const Scalar * data,
//                                   const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
//                                   float & totalWeight,
//                                   const float thisWeight,
//                                   ValidityCheck check,
//                                   const std::tuple<int, IdxTs...> remainingIndices) {

//    const int firstIndex = std::get<0>(remainingIndices);

//    return interpolateValidOnly(data + firstIndex*dimensions.tail.product(),
//                                dimensions.tail,
//                                totalWeight,
//                                thisWeight,
//                                check,
//                                GetTail(remainingIndices));

//}

//template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
//__NDT_CUDA_HD_PREFIX__ inline Scalar interpolateValidOnly(const Scalar * data,
//                                                      const IndexList<uint,sizeof...(IdxTs)> dimensions,
//                                                      ValidityCheck check,
//                                                      std::tuple<IdxTs...> indices) {

//    float totalWeight = 0.f;
//    const Scalar totalValue = interpolateValidOnly(data,dimensions,totalWeight,1.f,
//                                                   check, indices);

//    if (totalWeight) {
//        return totalValue / totalWeight;
//    }

//    return 0;
//}

template <typename Scalar, typename ValidityCheck>
__NDT_CUDA_HD_PREFIX__
inline bool validForInterpolation(const Scalar * data,
                                  const IndexList<uint,0> dimensions,
                                  ValidityCheck check,
                                  const std::tuple<> ) {

    return check(*data);

}

template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline bool validForInterpolation(const Scalar * data,
                                  const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                  ValidityCheck check,
                                  const std::tuple<float,IdxTs ...> remainingIndices) {

    const int i = std::get<0>(remainingIndices);

    return validForInterpolation(data + i*dimensions.tail.product(),
                                 dimensions.tail,
                                 check,
                                 GetTail(remainingIndices)) &&
           validForInterpolation(data + (i+1)*dimensions.tail.product(),
                                 dimensions.tail,
                                 check,
                                 GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ... IdxTs>
__NDT_CUDA_HD_PREFIX__
inline bool validForInterpolation(const Scalar * data,
                                  const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                  Transformer check,
                                  const std::tuple<int, IdxTs ...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return validForInterpolation(data + firstIndex*dimensions.tail.product(),
                                 dimensions.tail,
                                 check,
                                 GetTail(remainingIndices));

}


//template <typename Scalar,
//          typename Head,
//          typename Tail>
//struct Interpolator {

//};

//template <typename Scalar,
//          typename Tail>
//struct Interpolator<Scalar,float,Tail> {

//    typedef TypeList<float,Tail> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __NDT_CUDA_HD_PREFIX__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,IndexTypeList::Length,1> dimensions) {

//        const uint i = indices.head;
//        const float t = indices.head - i;

//        return (1-t)*Interpolator<Scalar,typename Tail::Head,typename Tail::Tail>
//                ::interpolate(indices.tail,
//                              data + i*dimensions.template head<Length-1>().prod(),
//                              dimensions.template head<Length-1>())
//               + t * Interpolator<Scalar,typename Tail::Head,typename Tail::Tail>
//                ::interpolate(indices.tail,
//                              data + (i+1)*dimensions.template head<Length-1>().prod(),
//                              dimensions.template head<Length-1>());

//    }

//};

//template <typename Scalar,
//          typename Tail>
//struct Interpolator<Scalar,int,Tail> {

//    typedef TypeList<int,Tail> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __NDT_CUDA_HD_PREFIX__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> dimensions) {

//        return Interpolator<Scalar,typename Tail::Head, typename Tail::Tail>
//                ::interpolate(indices.tail,
//                              data + indices.head*dimensions.template head<Length-1>().prod(),
//                              dimensions.template head<Length-1>());

//    }

//};

//template <typename Scalar>
//struct Interpolator<Scalar,float,NullType> {

//    typedef TypeList<float,NullType> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __NDT_CUDA_HD_PREFIX__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> /*dimensions*/) {

//        const uint i = indices.head;
//        const float t = indices.head - i;

//        return (1-t) * data[i] + t * data[i + 1];

//    }

//};

//template <typename Scalar>
//struct Interpolator<Scalar,int,NullType> {

//    typedef TypeList<int,NullType> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __NDT_CUDA_HD_PREFIX__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> /*dimensions*/) {

//        return data[indices.head];

//    }

//};

} // namespace internal

} // namespace NDT