#pragma once

#include <assert.h>

#include <iostream> // TODO

#include <Eigen/Core>

#include <cuda_runtime.h>

#include <NDT/TupleHelpers.h>

namespace NDT {

enum Residency {
    HostResident,
    DeviceResident
};

namespace internal {

// -=-=-=- const qualification -=-=-=-
template <typename T, bool ConstQualified>
struct ConstQualifier;

template <typename T>
struct ConstQualifier<T,false> {
    typedef T type;
};

template <typename T>
struct ConstQualifier<T,true> {
    typedef const T type;
};

template <typename T>
struct ConstQualifier<T *,false> {
    typedef T * type;
};

template <typename T>
struct ConstQualifier<T *,true> {
    typedef const T * type;
};

// -=-=-=- copying -=-=-=-

template <Residency DestR, Residency SrcR>
struct CopyTypeTraits;

template <>
struct CopyTypeTraits<HostResident,DeviceResident> {
    static constexpr cudaMemcpyKind copyType = cudaMemcpyDeviceToHost;
};

template <>
struct CopyTypeTraits<DeviceResident,HostResident> {
    static constexpr cudaMemcpyKind copyType = cudaMemcpyHostToDevice;
};

template <>
struct CopyTypeTraits<DeviceResident,DeviceResident> {
    static constexpr cudaMemcpyKind copyType = cudaMemcpyDeviceToDevice;
};


template <typename T, Residency DestR, Residency SrcR>
struct Copier {

    inline static void Copy(T * dst, const T * src, const std::size_t N) {
        cudaMemcpy(dst,src,N*sizeof(T),CopyTypeTraits<DestR,SrcR>::copyType);
    }

};

template <typename T>
struct Copier<T,HostResident,HostResident> {

    inline static void Copy(T * dst, const T * src, const std::size_t N) {
        std::memcpy(dst,src,N*sizeof(T));
    }

};

// -=-=-=- size equivalence checking -=-=-=-
template <bool Check>
struct EquivalenceChecker {

    template <typename DimT, uint D>
    inline static void CheckEquivalentSize(const Eigen::Matrix<DimT,D,1> & /*sizeA*/, const Eigen::Matrix<DimT,D,1> & /*sizeB*/) { }

    template <typename T>
    inline static void checkEquivalence(const T & /*A*/, const T & /*B*/) { }

};

template <>
struct EquivalenceChecker<true> {

    template <typename DimT, uint D>
    __attribute__((optimize("unroll-loops")))
    inline static void CheckEquivalentSize(const Eigen::Matrix<DimT,D,1> & sizeA, const Eigen::Matrix<DimT,D,1> & sizeB) {
        for (int d=0; d<D; ++d) {
            if (sizeA(d) != sizeB(d)) {
                throw std::runtime_error("sizes in dimension " + std::to_string(d) + " do not match: " +
                                         std::to_string(sizeA(d)) + " vs " + std::to_string(sizeB(d)));
            }
        }
    }

    template <typename T>
    inline static void checkEquivalence(const T & A, const T & B) {
        if (A != B) {
            throw std::runtime_error("not equivalent");
        }
    }

};

// -=-=-=- automatic allocation -=-=-=-
template <typename T, Residency R>
struct AutomaticAllocator;

template <typename T>
struct AutomaticAllocator<T,HostResident> {

    inline static T * allocate(const std::size_t length) {
        T * vals = new T[length];
        return vals;
    }

    inline static void deallocate(T * vec) {
        delete [] vec;
    }

};

template <typename T>
struct AutomaticAllocator<T,DeviceResident> {

    inline static T * allocate(const std::size_t length) {
        T * vals;
        cudaMalloc(&vals,length*sizeof(T));
        return vals;
    }

    inline static void deallocate(T * vec) {
        cudaFree(vec);
    }

};

// -=-=-=- generic indexing -=-=-=-
template <typename T, int D>
struct IndexList {
    T head;
    IndexList<T,D-1> tail;

    __host__ __device__
    inline IndexList(const Eigen::Matrix<T,D,1> & indices)
        : head(indices(0)), tail(indices.template tail<D-1>()) { }

//    template <int D2>
//    __host__ __device__
//    inline IndexList(const Eigen::VectorBlock<const Eigen::Matrix<T,D2,1>,D> & indices))

    __host__ __device__
    inline T sum() const {
        return head + tail.sum();
    }

    __host__ __device__
    inline T product() const {
        return head * tail.product();
    }

};

//template <typename T>
//struct IndexList<T,1> {
//    T head;

//    __host__ __device__
//    inline IndexList(const Eigen::Matrix<T,1,1> & indices)
//        : head(indices(0)) { }

//    __host__ __device__
//    inline T sum() const {
//        return head;
//    }

//    __host__ __device__
//    inline T product() const {
//        return head;
//    }

//};

template <typename T>
struct IndexList<T,0> {

    __host__ __device__
    inline IndexList(const Eigen::Matrix<T,0,1> & indices) { }

    __host__ __device__
    inline T sum() const {
        return 0;
    }

    __host__ __device__
    inline T product() const {
        return 1;
    }

};

template <typename T>
inline __host__ __device__ IndexList<T,1> IndexList1(const T i0) {
    return { i0, IndexList<T,0>() };
}

template <typename T>
inline __host__ __device__ IndexList<T,2> IndexList2(const T i0, const T i1) {
    return { i0, IndexList1(i1) };
}

template <typename T>
inline __host__ __device__ IndexList<T,3> IndexList3(const T i0, const T i1, const T i2) {
    return { i0, IndexList2(i1, i2) };
}

template <typename T>
inline __host__ __device__ IndexList<T,4> IndexList4(const T i0, const T i1, const T i2, const T i3) {
    return { i0, IndexList3(i1, i2, i3) };
}

template <typename IdxT, typename DimT, int D>
inline __host__ __device__ std::size_t OffsetXD(const IndexList<IdxT,D> dimIndices, const IndexList<DimT,D-1> dimSizes) {

    return dimIndices.head + dimSizes.head*OffsetXD(dimIndices.tail,dimSizes.tail);

}

template <typename IdxT, typename DimT>
inline __host__ __device__ std::size_t OffsetXD(const IndexList<IdxT,1> dimIndices, const IndexList<DimT,0> dimSizes) {

    return dimIndices.head;

}



//template <typename IdxT, typename DimT>
//inline __host__ __device__ std::size_t offsetXD(const IndexList<IdxT,2> dimIndices, const IndexList<DimT,1> dimSizes) {

//    return dimIndices.head + dimSizes.head*dimIndices.tail.head;

//}





// -=-=-=- interpolation -=-=-=-

//template <typename Scalar, typename ... IdxTs>
//struct Interpolator2;

//template <typename Scalar, typename ... IdxTs>
//struct Interpolator2<Scalar, float, IdxTs...> {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    __host__ __device__
//    static inline Scalar interpolate(const Scalar * data,
//                                     const Eigen::Matrix<uint,Length,1> dimensions,
//                                     float firstIndex, IdxTs ... remainingIndices) {

//        const uint i = firstIndex;
//        const float t = firstIndex - i;

//        return (1-t)*Interpolator2<Scalar, IdxTs...>::interpolate(data + i*dimensions.template head<Length-1>().prod(),
//                                                                  dimensions.template head<Length-1>(),
//                                                                  remainingIndices...)
//               + t * Interpolator2<Scalar, IdxTs...>::interpolate(data + (i+1)*dimensions.template head<Length-1>().prod(),
//                                                                  dimensions.template head<Length-1>(),
//                                                                  remainingIndices...);

//    }

//};

//template <typename Scalar, typename ... IdxTs>
//struct Interpolator2<Scalar, int, IdxTs...> {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    __host__ __device__
//    static inline Scalar interpolate(const Scalar * data,
//                                     const Eigen::Matrix<uint,Length,1> dimensions,
//                                     int firstIndex, IdxTs ... remainingIndices) {

//        return Interpolator2<Scalar, IdxTs...>::interpolate(data + firstIndex*dimensions.template head<Length-1>().prod(),
//                                                            dimensions.template head<Length-1>(),
//                                                            remainingIndices...);

//    }

//};

//template <typename Scalar>
//struct Interpolator2<Scalar> {

//    static constexpr uint Length = 0;

//    __host__ __device__
//    static inline Scalar interpolate(const Scalar * data,
//                                     const Eigen::Matrix<uint,Length,1> dimensions) {

//        return *data;

//    }

//};

//template <typename Scalar>
//__host__ __device__
//inline Scalar interpolate(const Scalar * data,
//                          const Eigen::Matrix<uint,0,1> dimensions) {

//    return *data;

//}

//template <typename Scalar, typename ... IdxTs>
//__host__ __device__
//inline Scalar interpolate(const Scalar * data,
//                          const Eigen::Matrix<uint,sizeof...(IdxTs)+1,1> dimensions,
//                          float firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    const uint i = firstIndex;
//    const float t = firstIndex - i;

//    return (1-t)*interpolate(data + i*dimensions.template head<Length-1>().prod(),
//                             dimensions.template head<Length-1>(),
//                             remainingIndices...)
//           + t * interpolate(data + (i+1)*dimensions.template head<Length-1>().prod(),
//                             dimensions.template head<Length-1>(),
//                             remainingIndices...);

//}

//template <typename Scalar, typename ... IdxTs>
//__host__ __device__
//inline Scalar interpolate(const Scalar * data,
//                          const Eigen::Matrix<uint,sizeof...(IdxTs) + 1,1> dimensions,
//                          int firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    return interpolate(data + firstIndex*dimensions.template head<Length-1>().prod(),
//                       dimensions.template head<Length-1>(),
//                       remainingIndices...);

//}


template <typename Scalar>
__host__ __device__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint,0> /*dimensions*/,
                          const std::tuple<> /*remainingIndices*/) {

    return *data;

}

template <typename Scalar,typename ... IdxTs>
__host__ __device__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<float, IdxTs...> remainingIndices) {

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const float t = firstIndex - i;

    return (1-t)*Interpolate(data + i*dimensions.tail.product(),
                             dimensions.tail,
                             GetTail(remainingIndices))
           + t * Interpolate(data + (i+1)*dimensions.tail.product(),
                             dimensions.tail,
                             GetTail(remainingIndices));

}

template <typename Scalar, typename ... IdxTs>
__host__ __device__
inline Scalar Interpolate(const Scalar * data,
                          const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<int, IdxTs...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return Interpolate(data + firstIndex*dimensions.tail.product(),
                       dimensions.tail,
                       GetTail(remainingIndices));

}


template <typename Scalar, typename ValidityCheck>
__host__ __device__
inline Scalar InterpolateValidOnly(const Scalar * data,
                                   const IndexList<uint,0> dimensions,
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
__host__ __device__
inline Scalar InterpolateValidOnly(const Scalar * data,
                                   const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                   float & totalWeight,
                                   const float thisWeight,
                                   ValidityCheck check,
                                   const std::tuple<float,IdxTs...> remainingIndices) {

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
__host__ __device__
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
__host__ __device__ inline Scalar InterpolateValidOnly(const Scalar * data,
                                                       const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                       ValidityCheck check,
                                                       std::tuple<IdxTs...> indices) {

    float totalWeight(0);
    const Scalar totalValue = InterpolateValidOnly(data,dimensions,totalWeight, 1.f,
                                                   check, indices);

    if (totalWeight) {
        return totalValue / totalWeight;
    }

    return 0 * totalValue;
}


// TODO: can this be subsumed into the original interpolate call by just having Transformer
// be the first type in the variadic parameter pack??
// the only tricky part would be deducing the return type through the recursive calls
//template <typename Scalar, typename Transformer>
//__host__ __device__
//inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
//                                                             const Eigen::Matrix<uint,0,1> dimensions,
//                                                             Transformer transformer) {

//    return transformer(*data);

//}

//template <typename Scalar, typename Transformer, typename ... IdxTs>
//__host__ __device__
//inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
//                                                             const Eigen::Matrix<uint,sizeof...(IdxTs)+1,1> dimensions,
//                                                             Transformer transformer,
//                                                             float firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    const uint i = firstIndex;
//    const float t = firstIndex - i;

//    return (1-t)*transformInterpolate(data + i*dimensions.template head<Length-1>().prod(),
//                                      dimensions.template head<Length-1>(),
//                                      transformer,
//                                      remainingIndices...)
//           + t * transformInterpolate(data + (i+1)*dimensions.template head<Length-1>().prod(),
//                                      dimensions.template head<Length-1>(),
//                                      transformer,
//                                      remainingIndices...);

//}

//template <typename Scalar, typename Transformer, typename ... IdxTs>
//__host__ __device__
//inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
//                                                             const Eigen::Matrix<uint,sizeof...(IdxTs) + 1,1> dimensions,
//                                                             Transformer transformer,
//                                                             int firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    return transformInterpolate(data + firstIndex*dimensions.template head<Length-1>().prod(),
//                                dimensions.template head<Length-1>(),
//                                transformer,
//                                remainingIndices...);

//}

template <typename Scalar, typename Transformer>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
                                                             const IndexList<uint,0> /*dimensions*/,
                                                             Transformer transformer,
                                                             const std::tuple<> /*remainingIndices*/) {

    return transformer(*data);

}

template <typename Scalar, typename Transformer, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
                                                             const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                             Transformer transformer,
                                                             const std::tuple<float, IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const typename Transformer::ScalarType t = firstIndex - i;

    return (1-t)*transformInterpolate(data + i*dimensions.tail.product(),
                                      dimensions.tail,
                                      transformer,
                                      GetTail(remainingIndices))
           + t * transformInterpolate(data + (i+1)*dimensions.tail.product(),
                                      dimensions.tail,
                                      transformer,
                                      GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
                                                             const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                             Transformer transformer,
                                                             const std::tuple<int, IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const int firstIndex = std::get<0>(remainingIndices);

    return transformInterpolate(data + firstIndex*dimensions.tail.product(),
                                dimensions.tail,
                                transformer,
                                GetTail(remainingIndices));

}


template <typename Scalar, typename Transformer, typename ValidityCheck>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
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

template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<float,IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const typename Transformer::ScalarType t = firstIndex - i;

    return transformInterpolateValidOnly(data + i*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight * (1-t),
                                         transformer,
                                         check,
                                         GetTail(remainingIndices)) +
           transformInterpolateValidOnly(data + (i+1)*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight * t,
                                         transformer,
                                         check,
                                         GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<int, IdxTs...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return transformInterpolateValidOnly(data + firstIndex*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight,
                                         transformer,
                                         check,
                                         GetTail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__host__ __device__ inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
                                                                                          const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                                                          Transformer transformer,
                                                                                          ValidityCheck check,
                                                                                          std::tuple<IdxTs...> indices) {

    typename Transformer::ScalarType totalWeight(0);
    const typename Transformer::ScalarType totalValue = transformInterpolateValidOnly(data,dimensions,totalWeight,typename Transformer::ScalarType(1),
                                                                                      transformer, check, indices);

    if (totalWeight) {
        return totalValue / totalWeight;
    }

    return 0;
}

//template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
//__host__ __device__
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
//__host__ __device__ inline Scalar interpolateValidOnly(const Scalar * data,
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
__host__ __device__
inline bool validForInterpolation(const Scalar * data,
                                   const IndexList<uint,0> dimensions,
                                   ValidityCheck check,
                                   const std::tuple<> ) {

    return check(*data);

}

template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__host__ __device__
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
__host__ __device__
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

//    __host__ __device__ static
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

//    __host__ __device__ static
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

//    __host__ __device__ static
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

//    __host__ __device__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> /*dimensions*/) {

//        return data[indices.head];

//    }

//};

enum DifferenceType {
    BackwardDifference,
    CentralDifference,
    ForwardDifference
};

template <int I, int Diff, typename ... IdxTs>
struct GradientReindex {

    __host__ __device__ inline
    static std::tuple<IdxTs...> reindex(std::tuple<IdxTs...> tuple) {
//        std::cout << std::get<I>(tuple) << " -> ";
        std::get<I>(tuple) += Diff;
//        std::cout << std::get<I>(tuple) << std::endl;
        return tuple;
    }

};

template <typename Scalar>
struct Interpolator {

    typedef Scalar InputType;
    typedef Scalar ReturnType;

    template <typename ... IdxTs>
    inline __host__ __device__ ReturnType interpolate(const InputType * data,
                                                      const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                      const std::tuple<IdxTs...> & indices) const {

        return internal::Interpolate(data,dimensions,indices);

    }

};

template <typename Transformer>
struct TransformInterpolator {

    typedef typename Transformer::InputType InputType;
    typedef typename Transformer::ReturnType ReturnType;

    inline __host__ __device__ TransformInterpolator(Transformer transformer) : transformer(transformer) { }

    template <typename ... IdxTs>
    inline __host__ __device__ ReturnType interpolate(const InputType * data,
                                                      const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                      const std::tuple<IdxTs...> & indices) const {

        return transformInterpolate(data,dimensions,transformer,indices);

    }

private:

    Transformer transformer;

};

template <typename Transformer, typename ValidityCheck>
struct TransformValidOnlyInterpolator {

    typedef typename Transformer::InputType InputType;
    typedef typename Transformer::ReturnType ReturnType;

    inline __host__ __device__ TransformValidOnlyInterpolator(Transformer transformer, ValidityCheck check)
        : transformer(transformer), check(check) { }

    template <typename ... IdxTs>
    inline __host__ __device__ typename Transformer::ReturnType interpolate(const typename Transformer::InputType * data,
                                                                            const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                                            const std::tuple<IdxTs...> & indices) const {

        return transformInterpolateValidOnly(data,dimensions,transformer,check,indices);

    }

private:

    Transformer transformer;
    ValidityCheck check;

};

template <DifferenceType Diff, typename InterpolatorType, typename Scalar, int R, int D, int Options>
struct GradientComputeCore;

// R-dimensional values, D-dimensional gradient
template <typename Scalar, typename InterpolatorType, int R, int D, int Options>
struct GradientComputeCore<BackwardDifference, InterpolatorType, Scalar, R, D, Options> {

    template <typename ... IdxTs>
    __host__ __device__ inline GradientComputeCore(const typename InterpolatorType::InputType * data,
                                                   const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   InterpolatorType interpolator)
        : interpolator(interpolator),
          center(interpolator.interpolate(data,dimensions,indices)) { }

    template <int I, typename ... IdxTs>
    __host__ __device__ inline
    Eigen::Matrix<Scalar,R,1,Options> compute(const typename InterpolatorType::InputType * data,
                                              const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                              const std::tuple<IdxTs...> & indices) const {

        return center - interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,-1,IdxTs...>::reindex(indices));

    }

private:

    InterpolatorType interpolator;
    const Eigen::Matrix<Scalar,R,1,Options> center;

};

// scalar values, D-dimensional gradient
template <typename Scalar, typename InterpolatorType, int D, int Options>
struct GradientComputeCore<BackwardDifference, InterpolatorType, Scalar, 1, D, Options> {

    template <typename ... IdxTs>
    __host__ __device__ inline GradientComputeCore(const typename InterpolatorType::InputType * data,
                                                   const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   InterpolatorType interpolator)
        : interpolator(interpolator),
          center(interpolator.interpolate(data,dimensions,indices)) { }

    template <int I, typename ... IdxTs>
    __host__ __device__ inline
    Scalar compute(const typename InterpolatorType::InputType * data,
                   const IndexList<uint,sizeof...(IdxTs)> dimensions,
                   const std::tuple<IdxTs...> & indices) const {

        return center - interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,-1,IdxTs...>::reindex(indices));

    }

private:

    InterpolatorType interpolator;
    const Scalar center;

};


// R-dimensional values, D-dimensional gradient
template <typename Scalar, typename InterpolatorType, int R, int D, int Options>
struct GradientComputeCore<CentralDifference, InterpolatorType, Scalar, R, D, Options> {

    template <typename ... IdxTs>
    __host__ __device__ inline GradientComputeCore(const Eigen::Matrix<Scalar,R,1,Options> * /*data*/,
                        const IndexList<uint,sizeof...(IdxTs)> /*dimensions*/,
                        const std::tuple<IdxTs...> & /*indices*/,
                        InterpolatorType interpolator)
        : interpolator(interpolator) { }

    template <int I, typename ... IdxTs>
    __host__ __device__ inline
    Eigen::Matrix<Scalar,R,1,Options> compute(const Eigen::Matrix<Scalar,R,1,Options> * data,
                                              const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                              const std::tuple<IdxTs...> & indices) const {

        return 0.5*(interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,1,IdxTs...>::reindex(indices)) - interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,-1,IdxTs...>::reindex(indices)));

    }

private:

    InterpolatorType interpolator;

};

template <typename Scalar, typename InterpolatorType, int D, int Options>
struct GradientComputeCore<CentralDifference, InterpolatorType, Scalar, 1, D, Options> {

    template <typename ... IdxTs>
    __host__ __device__ inline GradientComputeCore(const typename InterpolatorType::InputType * /*data*/,
                        const IndexList<uint,sizeof...(IdxTs)> /*dimensions*/,
                        const std::tuple<IdxTs...> & /*indices*/,
                        InterpolatorType interpolator)
        : interpolator(interpolator) { }

    template <int I, typename ... IdxTs>
    __host__ __device__ inline
    Scalar compute(const typename InterpolatorType::InputType * data,
                   const IndexList<uint,sizeof...(IdxTs)> dimensions,
                   const std::tuple<IdxTs...> & indices) const {

        return 0.5*(interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,1,IdxTs...>::reindex(indices)) - interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,-1,IdxTs...>::reindex(indices)));

    }

private:

    InterpolatorType interpolator;

};



template <typename Scalar, typename InterpolatorType, int R, int D, int Options>
struct GradientComputeCore<ForwardDifference, InterpolatorType, Scalar, R, D, Options> {

    template <typename ... IdxTs>
     __host__ __device__ inline GradientComputeCore(const typename InterpolatorType::InputType * data,
                                                    const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                    const std::tuple<IdxTs...> & indices,
                                                    InterpolatorType interpolator)
        : interpolator(interpolator),
          center(interpolator.interpolate(data,dimensions,indices)) { }

    template <int I, typename ... IdxTs>
    __host__ __device__ inline
    Eigen::Matrix<Scalar,R,1,Options> compute(const typename InterpolatorType::InputType * data,
                                              const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                              const std::tuple<IdxTs...> & indices) const {

        return interpolator.interpolate(data,dimensions,GradientReindex<D-1-I,1,IdxTs...>::reindex(indices)) - center;

    }

private:

    InterpolatorType interpolator;
    const Eigen::Matrix<Scalar,R,1,Options> center;

};



template <DifferenceType Diff, typename Scalar, int R, int D, int I>
struct GradientFiller {

    template <int Options, typename InterpolatorT, typename ... IdxTs>
    __host__ __device__ inline
    static void fill(Eigen::Matrix<Scalar,R,D,Options> & gradient,
                     const GradientComputeCore<Diff,InterpolatorT,Scalar,R,D,Options> & core,
                     const typename InterpolatorT::InputType * data,
                     const IndexList<uint,sizeof...(IdxTs)> dimensions,
                     const std::tuple<IdxTs...> & indices) {
        gradient.template block<R,1>(0,I) = core.template compute<I,IdxTs...>(data,dimensions,indices);
        GradientFiller<Diff,Scalar, R, D, I+1>::fill(gradient,core,data,dimensions,indices);
    }

};

template <DifferenceType Diff, typename Scalar, int R, int D>
struct GradientFiller<Diff,Scalar, R, D, D> {

    template <int Options, typename InterpolatorT, typename ... IdxTs>
    __host__ __device__ inline
    static void fill(Eigen::Matrix<Scalar,R,D,Options> & /*gradient*/,
                     const GradientComputeCore<Diff,InterpolatorT,Scalar,R,D,Options> & /*core*/,
                     const typename InterpolatorT::InputType * /*data*/,
                     const IndexList<uint,sizeof...(IdxTs)> /*dimensions*/,
                     const std::tuple<IdxTs...> & /*indices*/) { }

};

// TODO: I don't think this special case is needed
template <DifferenceType Diff, typename Scalar, int D, int I>
struct GradientFiller<Diff, Scalar,1,D,I> {

    template <int Options, typename InterpolatorT, typename ... IdxTs>
    __host__ __device__ inline
    static void fill(Eigen::Matrix<Scalar,1,D,Options> & gradient,
                     const GradientComputeCore<Diff,InterpolatorT,Scalar,1,D,Options> & core,
                     const typename InterpolatorT::InputType * data,
                     const IndexList<uint,sizeof...(IdxTs)> dimensions,
                     const std::tuple<IdxTs...> & indices) {

        gradient(I) = core.template compute<I,IdxTs...>(data,dimensions,indices);
        GradientFiller<Diff,Scalar, 1, D, I+1>::fill(gradient,core,data,dimensions,indices);

    }

};

// TODO: I don't think this special case is needed
template <DifferenceType Diff,typename Scalar, int D>
struct GradientFiller<Diff,Scalar, 1, D, D> {

    template <int Options, typename InterpolatorT, typename ... IdxTs>
    __host__ __device__ inline
    static void fill(Eigen::Matrix<Scalar,1,D,Options> & /*gradient*/,
                     const GradientComputeCore<Diff,InterpolatorT,Scalar,1,D,Options> & /*core*/,
                     const typename InterpolatorT::InputType * /*data*/,
                     const IndexList<uint,sizeof...(IdxTs)> /*dimensions*/,
                     const std::tuple<IdxTs...> & /*indices*/) { }

};


template <typename Scalar, int D, DifferenceType Diff>
struct GradientComputer {

    typedef Eigen::Matrix<Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor> GradientType;

    template <typename ... IdxTs>
    __host__ __device__ inline
    static GradientType compute(const Scalar * data,
                                const Eigen::Matrix<uint,D,1> & dimensions,
                                const std::tuple<IdxTs...> & indices) {
        GradientType gradient;
        GradientFiller<Diff,Scalar,1,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,Interpolator<Scalar>,Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::reverse(indices),
                                                                                                                                     Interpolator<Scalar>()),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::reverse(indices));
        return gradient;
    }

    template <typename Transformer, typename ... IdxTs>
    __host__ __device__ inline
    static GradientType transformCompute(const Transformer transformer,
                                         const typename Transformer::InputType * data,
                                         const Eigen::Matrix<uint,D,1> & dimensions,
                                         const std::tuple<IdxTs...> & indices) {

        GradientType gradient;
        GradientFiller<Diff,Scalar,1,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,TransformInterpolator<Transformer>,Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor>(data,
                                                                                                                                                           internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                                                                                           internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices),
                                                                                                                                                           TransformInterpolator<Transformer>(transformer)),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::reverse(indices));
        return gradient;

    }

    template <typename Transformer, typename ValidityCheck, typename ... IdxTs>
    __host__ __device__ inline
    static GradientType transformComputeValidOnly(Transformer transformer,
                                                  ValidityCheck check,
                                                  const typename Transformer::InputType * data,
                                                  const Eigen::Matrix<uint,D,1> & dimensions,
                                                  const std::tuple<IdxTs...> & indices) {

        GradientType gradient;
        GradientFiller<Diff,Scalar,1,D,0>::fill(gradient,
                                                                  GradientComputeCore<Diff,TransformValidOnlyInterpolator<Transformer,ValidityCheck>,Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor>(data,
                                                                                                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices),
                                                                                                                                                                             TransformValidOnlyInterpolator<Transformer,ValidityCheck>(transformer,check)),
                                                                  data,
                                                                  internal::IndexList<uint,D>(dimensions.reverse()),
                                                                  internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;

    }

};

template <typename Scalar, int R, int Options, int D, DifferenceType Diff>
struct GradientComputer<Eigen::Matrix<Scalar,R,1,Options>, D, Diff> {

    typedef Eigen::Matrix<Scalar,R,D,Options> GradientType;

    template <typename ... IdxTs>
    __host__ __device__ inline
    static GradientType compute(const Eigen::Matrix<Scalar,R,1,Options> * data,
                                const Eigen::Matrix<uint,D,1> & dimensions,
                                const std::tuple<IdxTs...> & indices) {
        GradientType gradient;
        GradientFiller<Diff,Scalar,R,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,Interpolator<Eigen::Matrix<Scalar,R,1,Options> >,Scalar,R,D,Options>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::reverse(indices),
                                                                                             Interpolator<Eigen::Matrix<Scalar,R,1,Options> >()),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;
    }

    template <typename Transformer, typename ... IdxTs>
    __host__ __device__ inline
    static GradientType transformCompute(const Transformer transformer,
                                         const typename Transformer::InputType * data,
                                         const Eigen::Matrix<uint,D,1> & dimensions,
                                         const std::tuple<IdxTs...> & indices) {
        Eigen::Matrix<Scalar,R,D,Options> gradient;
        GradientFiller<Diff,Scalar,R,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,TransformInterpolator<Transformer>,Scalar,R,D,Options>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::reverse(indices),
                                                                                             TransformInterpolator<Transformer>(transformer)),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;

    }

    template <typename ValidityCheck, typename Transformer, typename ... IdxTs>
    __host__ __device__ inline
    static GradientType transformComputeValidOnly(Transformer transformer,
                                                  ValidityCheck check,
                                                  const typename Transformer::InputType * data,
                                                  const Eigen::Matrix<uint,D,1> & dimensions,
                                                  const std::tuple<IdxTs...> & indices) {
        GradientType gradient;
        GradientFiller<Diff,Scalar,R,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,TransformValidOnlyInterpolator<Transformer,ValidityCheck>,Scalar,R,D,Options>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::reverse(indices),
                                                                                             TransformValidOnlyInterpolator<Transformer,ValidityCheck>(transformer,check)),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;

    }

};


template <typename ... IdxTs>
struct IndexTypePrinter {

    static inline __host__ __device__ void print() {
        std::cout << std::endl;
    }

};

template <typename ... IdxTs>
struct IndexTypePrinter<int,IdxTs...> {

    static inline __host__ __device__ void print(int v0, IdxTs... vs) {
        std::cout << "int ";
        IndexTypePrinter<IdxTs...>::print(vs...);
    }

};

template <typename ... IdxTs>
struct IndexTypePrinter<float,IdxTs...> {

    static inline __host__ __device__ void print(float v0, IdxTs... vs) {
        std::cout << "float ";
        IndexTypePrinter<IdxTs...>::print(vs...);
    }

};

} // namespace internal

template <typename Scalar>
class TypedTensorBase {

};

template <typename Scalar, Residency R>
class TypedResidentTensorBase : public TypedTensorBase<Scalar> {

};

template <uint D, typename T, Residency R = HostResident, bool Const = false>
class Tensor : public TypedResidentTensorBase<T,R> {
public:

    typedef unsigned int DimT;
    typedef unsigned int IdxT;

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    __host__ __device__ Tensor(const DimT length) : dimensions_(Eigen::Matrix<DimT,D,1>(length)), data_(nullptr) { }

    __host__ __device__ Tensor(const Eigen::Matrix<DimT,D,1> & dimensions) : dimensions_(dimensions), data_(nullptr) { }

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    __host__ __device__ Tensor(const DimT length, typename internal::ConstQualifier<T *,Const>::type data) :
        dimensions_(Eigen::Matrix<DimT,D,1>(length)), data_(data) { }

    // construct with values, not valid for managed tensors
    __host__ __device__ Tensor(const Eigen::Matrix<DimT,D,1> & dimensions,
                               typename internal::ConstQualifier<T *,Const>::type data) : dimensions_(dimensions), data_(data) { }

    // copy constructor and assignment operator, not valid for managed or const tensors
    template <bool _Const>
    __host__ __device__  Tensor(Tensor<D,T,R,_Const> & other)
        : dimensions_(other.Dimensions()), data_(other.Data()) {
        static_assert(Const || !_Const,
                      "Cannot copy-construct a non-const Tensor from a Const tensor");
    }

    template <bool _Const>
    __host__ __device__ inline Tensor<D,T,R,Const> & operator=(const Tensor<D,T,R,_Const> & other) {
        static_assert(Const || !_Const,
                      "Cannot assign a non-const Tensor from a Const tensor");
        dimensions_ = other.Dimensions();
        data_ = other.Data();
        return *this;
    }

    __host__ __device__ ~Tensor() { }

    // conversion to const tensor
    template <bool _Const = Const, typename std::enable_if<!_Const,int>::type = 0>
    inline operator Tensor<D,T,R,true>() const {
        return Tensor<D,T,R,true>( Dimensions(), Data() );
    }

    template <typename U = T,
              typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __host__ __device__ T * Data() { return data_; }

    inline __host__ __device__ const T * Data() const { return data_; }

    // -=-=-=-=-=-=- sizing functions -=-=-=-=-=-=-
    inline __host__ __device__ DimT DimensionSize(const IdxT dim) const {
        return dimensions_(dim);
    }

    inline __host__ __device__ const Eigen::Matrix<DimT,D,1,Eigen::DontAlign> & Dimensions() const {
        return dimensions_;
    }

    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ DimT Length() const {
        return dimensions_(0);
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ DimT Width() const {
        return dimensions_(0);
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ DimT Height() const {
        return dimensions_(1);
    }

    inline __host__ __device__ std::size_t Count() const {
//        return internal::count<DimT,D>(dimensions_);
        return dimensions_.prod();
    }

    inline __host__ __device__ std::size_t SizeBytes() const {
        return Count() * sizeof(T);
    }

    // -=-=-=-=-=-=- indexing functions -=-=-=-=-=-=-
    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0) const {
        return data_[d0];
    }

    template <int D2 = D, typename std::enable_if<D2 == 1 && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0) {
        return data_[d0];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1) const {
        return data_[internal::OffsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2 && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1) {
        return data_[internal::OffsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])))];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1));
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1, const IdxT d2) const {
        return data_[internal::OffsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 3 && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1, const IdxT d2) {
        return data_[internal::OffsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])))];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1),indices(2));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1),indices(2));
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) const {
        return data_[internal::OffsetXD<IdxT,DimT,4>(internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(d0,d1,d2,d3)),
                internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(dimensions_[0],dimensions_[1],dimensions_[2])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 4 && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) {
        return data_[internal::OffsetXD<IdxT,DimT,4>(internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(d0,d1,d2,d3)),
                internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(dimensions_[0],dimensions_[1],dimensions_[2])))];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1),indices(2),indices(3));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1),indices(2),indices(3));
    }

    template <int D2 = D, typename std::enable_if<D2 == 5, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3, const IdxT d4) const {
        return data_[internal::OffsetXD<IdxT,DimT,5>(internal::IndexList<IdxT,5>(Eigen::Matrix<uint,5,1>(d0,d1,d2,d3,d4)),
                internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(dimensions_[0],dimensions_[1],dimensions_[2],dimensions_[3])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 5 && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3, const IdxT d4) {
        return data_[internal::OffsetXD<IdxT,DimT,5>(internal::IndexList<IdxT,5>(Eigen::Matrix<uint,5,1>(d0,d1,d2,d3,d4)),
                internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(dimensions_[0],dimensions_[1],dimensions_[2],dimensions_[3])))];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 5 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1),indices(2),indices(3),indices(4));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 5 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1),indices(2),indices(3),indices(4));
    }




    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ DimT offset(const IdxT d0, const IdxT d1) const {
        return internal::OffsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                                               internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ DimT offset(const Eigen::MatrixBase<Derived> & indices) const {
        return offset(indices(0),indices(1));
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ DimT offset(const IdxT d0, const IdxT d1, const IdxT d2) const {
        return internal::OffsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                                               internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ DimT offset(const Eigen::MatrixBase<Derived> & indices) const {
        return offset(indices(0),indices(1),indices(2));
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ DimT offset(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) const {
        return internal::OffsetXD<IdxT,DimT,4>(internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(d0,d1,d2,d3)),
                                               internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(dimensions_[0],dimensions_[1],dimensions_[2])));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ DimT offset(const Eigen::MatrixBase<Derived> & indices) const {
        return offset(indices(0),indices(1),indices(2),indices(3));
    }

    // -=-=-=-=-=-=- interpolation functions -=-=-=-=-=-=-
//    template <typename IdxT1,
//              int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0) const {
//        return internal::interpolate(data_, dimensions_, v0);
//    }

//    template <typename IdxT1, typename IdxT2,
//              int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0, const IdxT2 v1) const {
//        return internal::interpolate(data_, dimensions_, v1, v0);
//    }

//    template <typename IdxT1, typename IdxT2, typename IdxT3,
//              int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0, const IdxT2 v1, const IdxT3 v2) const {
//        return internal::interpolate(data_, dimensions_, v2, v1, v0);
//    }

//    template <typename IdxT1, typename IdxT2, typename IdxT3, typename IdxT4,
//              int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0, const IdxT2 v1, const IdxT3 v2, const IdxT4 v3) const {
//        return internal::interpolate(data_, dimensions_, v3, v2, v1, v0);
//    }

    template <typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ T Interpolate(const IdxTs ... vs) const {
        return internal::Interpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                     internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ T Interpolate(const Eigen::MatrixBase<Derived> & v) const {
        return internal::Interpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                     VectorToTuple(v.reverse()));
    }

    template <typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ T InterpolateValidOnly(ValidityCheck check, IdxTs ... vs) const {

        return internal::InterpolateValidOnly(data_,internal::IndexList<DimT,D>(dimensions_.reverse()),
                                              check, internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));

    }

    template <typename ValidityCheck, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ T InterpolateValidOnly(ValidityCheck check, const Eigen::MatrixBase<Derived> & v) const {
        return internal::InterpolateValidOnly(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                              check,VectorToTuple(v.reverse()));
    }

    template <typename Transformer, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ typename Transformer::ReturnType transformInterpolate(Transformer transformer, const IdxTs ... vs) const {
        return internal::transformInterpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), transformer,
                                              internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));
    }

    template <typename Transformer, typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ typename Transformer::ReturnType transformInterpolateValidOnly(Transformer transformer, ValidityCheck check, IdxTs ... vs) const {

        return internal::transformInterpolateValidOnly(data_,internal::IndexList<DimT,D>(dimensions_.reverse()),
                                                       transformer, check,
                                                       internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));

    }

    template <typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ bool validForInterpolation(ValidityCheck check, const IdxTs ... vs) {
        return internal::validForInterpolation(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), check,
                                               internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));
    }

    // -=-=-=-=-=-=- bounds-checking functions -=-=-=-=-=-=-
    template <typename PosT, typename BorderT, int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ bool InBounds(const PosT d0, const BorderT border) const {
        return (d0 >= border) && (d0 <= DimensionSize(0) - 1 - border);
    }

    template <typename PosT, typename BorderT, int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ bool InBounds(const PosT d0, const PosT d1, const BorderT border) const {
        return (d0 >= border) && (d0 <= DimensionSize(0) - 1 - border) &&
               (d1 >= border) && (d1 <= DimensionSize(1) - 1 - border);
    }

    template <typename BorderT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ bool InBounds(const Eigen::MatrixBase<Derived> & point, const BorderT border) const {
        return InBounds(point(0),point(1),border);
    }

    template <typename PosT, typename BorderT, int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ bool InBounds(const PosT d0, const PosT d1, const PosT d2, const BorderT border) const {
        return (d0 >= border) && (d0 <= DimensionSize(0) - 1 - border) &&
               (d1 >= border) && (d1 <= DimensionSize(1) - 1 - border) &&
               (d2 >= border) && (d2 <= DimensionSize(2) - 1 - border);
    }

    template <typename BorderT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ bool InBounds(const Eigen::MatrixBase<Derived> & point, const BorderT border) const {
        return InBounds(point(0),point(1),point(2),border);
    }

    template <typename PosT, typename BorderT, int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ bool InBounds(const PosT d0, const PosT d1, const PosT d2, const PosT d3, const BorderT border) const {
        return (d0 >= border) && (d0 <= DimensionSize(0) - 1 - border) &&
               (d1 >= border) && (d1 <= DimensionSize(1) - 1 - border) &&
               (d2 >= border) && (d2 <= DimensionSize(2) - 1 - border) &&
               (d3 >= border) && (d3 <= DimensionSize(3) - 1 - border);
    }

    template <typename BorderT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ bool InBounds(const Eigen::MatrixBase<Derived> & point, const BorderT border) const {
        return InBounds(point(0),point(1),point(2),point(3),border);
    }

    // -=-=-=-=-=-=- gradient functions -=-=-=-=-=-=-
    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<T,D,internal::BackwardDifference>::GradientType backwardDifference(const Eigen::MatrixBase<Derived> & v) const {

        return internal::GradientComputer<T,D,internal::BackwardDifference>::compute(data_,dimensions_,VectorToTuple(v));

    }

    template <typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<T,D,internal::BackwardDifference>::GradientType backwardDifference(const IdxTs ... v) const {

//        internal::IndexTypePrinter<IdxTs...>::print(v...);

        return internal::GradientComputer<T,D,internal::BackwardDifference>::template compute<IdxTs...>(data_,dimensions_,std::tuple<IdxTs...>(v...));

    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<T,D,internal::BackwardDifference>::GradientType centralDifference(const Eigen::MatrixBase<Derived> & v) const {

        return internal::GradientComputer<T,D,internal::CentralDifference>::compute(data_,dimensions_,VectorToTuple(v));

    }

    template <typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<T,D,internal::BackwardDifference>::GradientType centralDifference(const IdxTs ... v) const {

        return internal::GradientComputer<T,D,internal::CentralDifference>::template compute<IdxTs...>(data_,dimensions_,std::tuple<IdxTs...>(v...));

    }


    template <typename Transformer,
              typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::GradientType transformBackwardDifference(Transformer transformer, const Eigen::MatrixBase<Derived> & v) const {

        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::transformCompute(transformer,data_,dimensions_,VectorToTuple(v));

    }

    template <typename Transformer,
              typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::GradientType transformBackwardDifference(Transformer transformer, const IdxTs ... v) const {

        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::template transformCompute<Transformer,IdxTs...>(transformer,data_,dimensions_,std::tuple<IdxTs...>(v...));

    }


    template <typename Transformer,
              typename ValidityCheck,
              typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::GradientType TransformBackwardDifferenceValidOnly(Transformer transformer, ValidityCheck check, const Eigen::MatrixBase<Derived> & v) const {

        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::transformComputeValidOnly(transformer,check,data_,dimensions_,VectorToTuple(v));

    }

    template <typename Transformer,
              typename ValidityCheck,
              typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0>
    inline __host__ __device__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::GradientType TransformBackwardDifferenceValidOnly(Transformer transformer, ValidityCheck check, const IdxTs ... v) const {

        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::BackwardDifference>::template transformComputeValidOnly<Transformer,ValidityCheck,IdxTs...>(transformer,check,data_,dimensions_,std::tuple<IdxTs...>(v...));

    }

//    template <typename Transformer, typename ValidityCheck, typename IdxT1,
//              int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
//    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0) {
//        typedef typename Transformer::ReturnType Transformed;
//        const Transformed center = transformInterpolateValidOnly(transformer,check,v0);
//        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1));
//    }

//    template <typename Transformer, typename ValidityCheck, typename IdxT1, typename IdxT2,
//              int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
//    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0, const IdxT2 v1) {
//        typedef typename Transformer::ReturnType Transformed;
//        const Transformed center = transformInterpolateValidOnly(transformer,check,v0,v1);
//        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1,v1),
//                                              center - transformInterpolateValidOnly(transformer,check,v0,v1-1));
//    }

//    template <typename Transformer, typename ValidityCheck, typename IdxT1, typename IdxT2, typename IdxT3,
//              int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
//    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0, const IdxT2 v1, const IdxT3 v2) {
//        typedef typename Transformer::ReturnType Transformed;
//        const Transformed center = transformInterpolateValidOnly(transformer,check,v0,v1,v2);
//        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1,v1,v2),
//                                              center - transformInterpolateValidOnly(transformer,check,v0,v1-1,v2),
//                                              center - transformInterpolateValidOnly(transformer,check,v0,v1,v2-1));
//    }

//    template <typename Transformer, typename ValidityCheck, typename IdxT1, typename IdxT2, typename IdxT3, typename IdxT4,
//              int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
//    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0, const IdxT2 v1, const IdxT3 v2, const IdxT4 v3) {
//        typedef typename Transformer::ReturnType Transformed;
//        const Transformed center = transformInterpolateValidOnly(transformer,check,v0,v1,v2,v3);
//        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1,v1,v2,v3),
//                                              center - transformInterpolateValidOnly(transformer,check,v0,v1-1,v2,v3),
//                                              center - transformInterpolateValidOnly(transformer,check,v0,v1,v2-1,v3),
//                                              center - transformInterpolateValidOnly(transformer,check,v0,v1,v2,v3-1));
//    }


    // -=-=-=-=-=-=- pointer manipulation functions -=-=-=-=-=-=-
    template <typename U = T,
              typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __host__ __device__ void SetDataPointer(T * data) { data_ = data; }

    // -=-=-=-=-=-=- copying functions -=-=-=-=-=-=-
    template <Residency R2, bool Const2, bool Check=false>
    inline void CopyFrom(const Tensor<D,T,R2,Const2> & other) {
        static_assert(!Const,"you cannot copy to a const tensor");
        internal::EquivalenceChecker<Check>::template CheckEquivalentSize<DimT,D>(Dimensions(),other.Dimensions());
        internal::Copier<T,R,R2>::Copy(data_,other.Data(),Count());
    }

protected:

    Eigen::Matrix<DimT,D,1,Eigen::DontAlign> dimensions_;
    typename internal::ConstQualifier<T *,Const>::type data_;

//public:

//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

template <Residency R>
class ResidentManagedTensorBase {

};

template <typename Scalar>
class TypedManagedTensorBase {

};

template <typename Scalar, Residency R>
class TypedResidentManagedTensorBase {

};

template <uint D, typename T, Residency R = HostResident>
class ManagedTensor : public Tensor<D,T,R,false>,
                      public TypedResidentManagedTensorBase<T,R>,
                      public ResidentManagedTensorBase<R> {
public:

    typedef typename Tensor<D,T,R,false>::DimT DimT;

    ManagedTensor() :
        Tensor<D,T,R,false>::Tensor(Eigen::Matrix<uint,D,1>::Zero(), nullptr) { }

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    ManagedTensor(const DimT length) :
        Tensor<D,T,R,false>::Tensor(length, internal::AutomaticAllocator<T,R>::allocate(length)) { }

    ManagedTensor(const Eigen::Matrix<DimT,D,1> & dimensions) :
        Tensor<D,T,R,false>::Tensor(dimensions,
                                    internal::AutomaticAllocator<T,R>::allocate(dimensions.prod())) { }

    ~ManagedTensor() {
        internal::AutomaticAllocator<T,R>::deallocate(this->data_);
    }

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    inline void Resize(const DimT length) {
        Resize(Eigen::Matrix<DimT,D,1>(length));
    }

    void Resize(const Eigen::Matrix<DimT,D,1> & dimensions) {
        internal::AutomaticAllocator<T,R>::deallocate(this->data_);
        this->data_ = internal::AutomaticAllocator<T,R>::allocate(dimensions.prod());
        this->dimensions_ = dimensions;
    }

private:

    ManagedTensor(const ManagedTensor &) = delete;
    ManagedTensor & operator=(const ManagedTensor &) = delete;

};

//namespace internal {

//typedef unsigned int DimT;

//template <bool Packed>
//struct FirstDimensionStride;

//template <>
//struct FirstDimensionStride<true> {
//    inline DimT stride() const { return 1; }
//};

//template <>
//struct FirstDimensionStride<false> {
//    inline DimT stride() const { return stride_; }
//    DimT stride_;
//};


//template <bool SourcePacked, unsigned int FirstDimension>
//struct SliceReturnValPacked {
////    using Determinant = PackingDeterminant<SourcePacked>::Determinant;
////    static constexpr bool Packed = Determinant<FirstDimension>::Packed;
//    static constexpr bool Packed = false;
//};

//template <>
//struct SliceReturnValPacked<true,0> {
//    static constexpr bool Packed = true;
//};


//} // namespace internal

//template <uint D, typename T, Residency R = HostResident, bool Const = false, bool Packed = true>
//class Tensor {
//public:

//    typedef internal::DimT DimT;
//    typedef unsigned int IndT;

//    inline __host__ __device__ DimT dimensionSize(const IndT dim) const {
//        assert(dim < D);
//        return dimensions_[dim];
//    }

//    template <unsigned int FirstDimension, unsigned int ... Rest>
//    inline __host__ __device__ Tensor<D,T,R,Const,internal::SliceReturnValPacked<Packed,FirstDimension>::Packed> slice() {

//    }

//protected:

//    std::array<DimT,D> dimensions_;
//    internal::FirstDimensionStride<Packed> firstDimensionStride_;
//    std::array<DimT,D-1> otherDimensionStrides_;
//    typename ConstQualifier<T *,Const>::type values_;

//};

// -=-=-=-=- full tensor typedefs -=-=-=-=-
#define TENSOR_TYPEDEFS_(i, type, appendix)                                       \
    typedef Tensor<i,type,HostResident> Tensor##i##appendix;                      \
    typedef Tensor<i,type,DeviceResident> DeviceTensor##i##appendix;              \
    typedef Tensor<i,type,HostResident,true> ConstTensor##i##appendix;            \
    typedef Tensor<i,type,DeviceResident,true> ConstDeviceTensor##i##appendix;    \
    typedef ManagedTensor<i,type,HostResident> ManagedTensor##i##appendix;        \
    typedef ManagedTensor<i,type,DeviceResident> ManagedDeviceTensor##i##appendix

#define TENSOR_TYPEDEFS(type, appendix)  \
    TENSOR_TYPEDEFS_(1, type, appendix); \
    TENSOR_TYPEDEFS_(2, type, appendix); \
    TENSOR_TYPEDEFS_(3, type, appendix); \
    TENSOR_TYPEDEFS_(4, type, appendix); \
    TENSOR_TYPEDEFS_(5, type, appendix)

TENSOR_TYPEDEFS(float,f);
TENSOR_TYPEDEFS(double,d);
TENSOR_TYPEDEFS(int,i);
TENSOR_TYPEDEFS(uint,ui);
TENSOR_TYPEDEFS(unsigned char,uc);

template <int D, typename Scalar>
using DeviceTensor = Tensor<D,Scalar,DeviceResident>;

template <int D, typename Scalar>
using ConstTensor = Tensor<D,Scalar,HostResident, true>;

template <int D, typename Scalar>
using ConstDeviceTensor = Tensor<D,Scalar,DeviceResident, true>;

#define TENSOR_PARTIAL_TYPEDEF_(i,residency)                                         \
    template <typename Scalar>                                                       \
    using residency##Tensor##i = Tensor<i,Scalar,residency##Resident>;               \
    template <typename Scalar>                                                       \
    using Const##residency##Tensor##i = Tensor<i,Scalar,residency##Resident,true>;   \
    template <typename Scalar>                                                       \
    using Managed##residency##Tensor##i = ManagedTensor<i,Scalar,residency##Resident>

#define TENSOR_PARTIAL_TYPEDEF(i)                  \
    TENSOR_PARTIAL_TYPEDEF_(i,Device);             \
    TENSOR_PARTIAL_TYPEDEF_(i,Host)

//template <typename Scalar>
//using DeviceTensor2 = Tensor<2,Scalar,DeviceResident>;

TENSOR_PARTIAL_TYPEDEF(1);
TENSOR_PARTIAL_TYPEDEF(2);
TENSOR_PARTIAL_TYPEDEF(3);
TENSOR_PARTIAL_TYPEDEF(4);
TENSOR_PARTIAL_TYPEDEF(5);

template <typename Scalar>
using Vector = HostTensor1<Scalar>;

template <typename Scalar>
using ManagedVector = ManagedHostTensor1<Scalar>;

template <typename Scalar>
using DeviceVector = DeviceTensor1<Scalar>;

template <typename Scalar>
using ManagedDeviceVector = ManagedDeviceTensor1<Scalar>;

template <typename Scalar>
using Image = HostTensor2<Scalar>;

template <typename Scalar>
using ManagedImage = ManagedHostTensor2<Scalar>;

template <typename Scalar>
using DeviceImage = DeviceTensor2<Scalar>;

template <typename Scalar>
using ManagedDeviceImage = ManagedDeviceTensor2<Scalar>;

template <typename Scalar>
using Volume = HostTensor3<Scalar>;

template <typename Scalar>
using ManagedVolume = ManagedHostTensor3<Scalar>;

template <typename Scalar>
using DeviceVolume = DeviceTensor3<Scalar>;

template <typename Scalar>
using ManagedDeviceVolume = ManagedDeviceTensor3<Scalar>;


} // namespace NDT
