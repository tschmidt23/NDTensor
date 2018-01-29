#pragma once

#include <assert.h>

#include <iostream> // TODO

#include <Eigen/Core>

#include <NDT/TupleHelpers.h>

#include <NDT/TensorBase.h>

#include <NDT/Internal/ConstQualifier.h>
#include <NDT/Internal/IndexList.h>
#include <NDT/Internal/Interpolation.h>
#include <NDT/Internal/InterpolationGradient.h>

namespace NDT {

enum Residency {
    HostResident,
    DeviceResident
};

} // namespace NDT

#include <NDT/Internal/Copying.h>
#include <NDT/Internal/GradientTraits.h>

namespace NDT {

namespace internal {

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
//        T * vals = std::malloc(length * sizeof(T));
        return vals;
    }

    inline static void deallocate(T * vec) {
        delete [] vec;
//        std::free(vec);
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
template <typename IdxT, typename DimT, int D, typename std::enable_if<(D > 1), int>::type = 0>
inline __NDT_CUDA_HD_PREFIX__ std::size_t OffsetXD(const IndexList<IdxT,D> dimIndices, const IndexList<DimT,D-1> dimSizes) {

    return dimIndices.head + dimSizes.head*OffsetXD(dimIndices.tail,dimSizes.tail);

}

template <typename IdxT, typename DimT>
inline __NDT_CUDA_HD_PREFIX__ std::size_t OffsetXD(const IndexList<IdxT,1> dimIndices, const IndexList<DimT,0> dimSizes) {

    return dimIndices.head;

}

template <typename BorderT>
__NDT_CUDA_HD_PREFIX__
inline bool BoundsCheck(const IndexList<uint,0> /*dimensions*/,
                        const std::tuple<> /*remainingPositions*/,
                        const BorderT /*borderLow*/,
                        const BorderT /*borderHight*/) {
    return true;
}

template <typename BorderT, typename Head, typename ... Tail>
__NDT_CUDA_HD_PREFIX__
inline bool BoundsCheck(const IndexList<uint,sizeof...(Tail)+1> dimensions,
                        const std::tuple<Head, Tail...> remainingPositions,
                        const BorderT borderLow,
                        const BorderT borderHigh) {

    const Head firstPosition = std::get<0>(remainingPositions);
    return (firstPosition >= borderLow && firstPosition <= dimensions.head - 1 - borderHigh) &&
            BoundsCheck(dimensions.tail,GetTail(remainingPositions),borderLow,borderHigh);

}

enum DifferenceType {
    BackwardDifference,
    CentralDifference,
    ForwardDifference
};

template <int I, int Diff, typename ... IdxTs>
struct GradientReindex {

    __NDT_CUDA_HD_PREFIX__ inline
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
    inline __NDT_CUDA_HD_PREFIX__ ReturnType interpolate(const InputType * data,
                                                      const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                      const std::tuple<IdxTs...> & indices) const {

        return internal::Interpolate(data,dimensions,indices);

    }

};

template <typename Transformer>
struct TransformInterpolator {

    typedef typename Transformer::InputType InputType;
    typedef typename Transformer::ReturnType ReturnType;

    inline __NDT_CUDA_HD_PREFIX__ TransformInterpolator(Transformer transformer) : transformer(transformer) { }

    template <typename ... IdxTs>
    inline __NDT_CUDA_HD_PREFIX__ ReturnType interpolate(const InputType * data,
                                                      const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                      const std::tuple<IdxTs...> & indices) const {

        return TransformInterpolate(data,dimensions,transformer,indices);

    }

private:

    Transformer transformer;

};

template <typename Transformer, typename ValidityCheck>
struct TransformValidOnlyInterpolator {

    typedef typename Transformer::InputType InputType;
    typedef typename Transformer::ReturnType ReturnType;

    inline __NDT_CUDA_HD_PREFIX__ TransformValidOnlyInterpolator(Transformer transformer, ValidityCheck check)
        : transformer(transformer), check(check) { }

    template <typename ... IdxTs>
    inline __NDT_CUDA_HD_PREFIX__ typename Transformer::ReturnType interpolate(const typename Transformer::InputType * data,
                                                                            const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                                            const std::tuple<IdxTs...> & indices) const {

        return TransformInterpolateValidOnly(data,dimensions,transformer,check,indices);

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
    __NDT_CUDA_HD_PREFIX__ inline GradientComputeCore(const typename InterpolatorType::InputType * data,
                                                   const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   InterpolatorType interpolator)
        : interpolator(interpolator),
          center(interpolator.interpolate(data,dimensions,indices)) { }

    template <int I, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline GradientComputeCore(const typename InterpolatorType::InputType * data,
                                                   const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                   const std::tuple<IdxTs...> & indices,
                                                   InterpolatorType interpolator)
        : interpolator(interpolator),
          center(interpolator.interpolate(data,dimensions,indices)) { }

    template <int I, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline GradientComputeCore(const Eigen::Matrix<Scalar,R,1,Options> * /*data*/,
                        const IndexList<uint,sizeof...(IdxTs)> /*dimensions*/,
                        const std::tuple<IdxTs...> & /*indices*/,
                        InterpolatorType interpolator)
        : interpolator(interpolator) { }

    template <int I, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline GradientComputeCore(const typename InterpolatorType::InputType * /*data*/,
                        const IndexList<uint,sizeof...(IdxTs)> /*dimensions*/,
                        const std::tuple<IdxTs...> & /*indices*/,
                        InterpolatorType interpolator)
        : interpolator(interpolator) { }

    template <int I, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
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
     __NDT_CUDA_HD_PREFIX__ inline GradientComputeCore(const typename InterpolatorType::InputType * data,
                                                    const IndexList<uint,sizeof...(IdxTs)> dimensions,
                                                    const std::tuple<IdxTs...> & indices,
                                                    InterpolatorType interpolator)
        : interpolator(interpolator),
          center(interpolator.interpolate(data,dimensions,indices)) { }

    template <int I, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline
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
    __NDT_CUDA_HD_PREFIX__ inline
    static GradientType compute(const Scalar * data,
                                const Eigen::Matrix<uint,D,1> & dimensions,
                                const std::tuple<IdxTs...> & indices) {
        GradientType gradient;
        GradientFiller<Diff,Scalar,1,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,Interpolator<Scalar>,Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices),
                                                                                                                                     Interpolator<Scalar>()),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;
    }

    template <typename Transformer, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
    static GradientType TransformCompute(const Transformer transformer,
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
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;

    }

    template <typename Transformer, typename ValidityCheck, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
    static GradientType TransformComputeValidOnly(Transformer transformer,
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
    __NDT_CUDA_HD_PREFIX__ inline
    static GradientType compute(const Eigen::Matrix<Scalar,R,1,Options> * data,
                                const Eigen::Matrix<uint,D,1> & dimensions,
                                const std::tuple<IdxTs...> & indices) {
        GradientType gradient;
        GradientFiller<Diff,Scalar,R,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,Interpolator<Eigen::Matrix<Scalar,R,1,Options> >,Scalar,R,D,Options>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices),
                                                                                             Interpolator<Eigen::Matrix<Scalar,R,1,Options> >()),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;
    }

    template <typename Transformer, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
    static GradientType TransformCompute(const Transformer transformer,
                                         const typename Transformer::InputType * data,
                                         const Eigen::Matrix<uint,D,1> & dimensions,
                                         const std::tuple<IdxTs...> & indices) {
        Eigen::Matrix<Scalar,R,D,Options> gradient;
        GradientFiller<Diff,Scalar,R,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,TransformInterpolator<Transformer>,Scalar,R,D,Options>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices),
                                                                                             TransformInterpolator<Transformer>(transformer)),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;

    }

    template <typename ValidityCheck, typename Transformer, typename ... IdxTs>
    __NDT_CUDA_HD_PREFIX__ inline
    static GradientType TransformComputeValidOnly(Transformer transformer,
                                                  ValidityCheck check,
                                                  const typename Transformer::InputType * data,
                                                  const Eigen::Matrix<uint,D,1> & dimensions,
                                                  const std::tuple<IdxTs...> & indices) {
        GradientType gradient;
        GradientFiller<Diff,Scalar,R,D,0>::fill(gradient,
                                                GradientComputeCore<Diff,TransformValidOnlyInterpolator<Transformer,ValidityCheck>,Scalar,R,D,Options>(data,
                                                                                             internal::IndexList<uint,D>(dimensions.reverse()),
                                                                                             internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices),
                                                                                             TransformValidOnlyInterpolator<Transformer,ValidityCheck>(transformer,check)),
                                                data,
                                                internal::IndexList<uint,D>(dimensions.reverse()),
                                                internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(indices));
        return gradient;

    }

};

template <DifferenceType>
struct DifferenceTypeTraits;

template <>
struct DifferenceTypeTraits<BackwardDifference> {
    static constexpr int borderLow = 1;
    static constexpr int borderHigh = 0;
};

template <>
struct DifferenceTypeTraits<CentralDifference> {
    static constexpr int borderLow = 1;
    static constexpr int borderHigh = 1;
};

template <>
struct DifferenceTypeTraits<ForwardDifference> {
    static constexpr int borderLow = 0;
    static constexpr int borderHigh = 1;
};

template <typename Derived, int D>
struct IsVectorType {

    static constexpr bool Value = Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1;

};

template <typename Derived, int D>
struct IsRealVectorType {

    static constexpr bool Value = IsVectorType<Derived,D>::Value &&
                                  std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value;

};

template <typename Derived, int D>
struct IsIntegralVectorType {

    static constexpr bool Value = IsVectorType<Derived,D>::Value &&
                                  std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value;

};

template <uint D>
struct StrideConstructor {

    static inline Eigen::Matrix<uint,D,1,Eigen::DontAlign>
    Construct(const uint soFar, const Eigen::Matrix<uint,D,1,Eigen::DontAlign> & dimensions) {

        return (Eigen::Matrix<uint,D,1,Eigen::DontAlign>() << soFar, StrideConstructor<D-1>::Construct(soFar * dimensions(0), dimensions.template tail<D-1>())).finished();

    }

};

template <>
struct StrideConstructor<1> {

    static inline Eigen::Matrix<uint,1,1,Eigen::DontAlign>
    Construct(const uint soFar, const Eigen::Matrix<uint,1,1,Eigen::DontAlign> & /*dimensions*/) {

        return Eigen::Matrix<uint,1,1,Eigen::DontAlign>(soFar);

    }

};

template <typename ... IdxTs>
struct IndexTypePrinter {

    static inline __NDT_CUDA_HD_PREFIX__ void print() {
        std::cout << std::endl;
    }

};

template <typename ... IdxTs>
struct IndexTypePrinter<int,IdxTs...> {

    static inline __NDT_CUDA_HD_PREFIX__ void print(int v0, IdxTs... vs) {
        std::cout << "int ";
        IndexTypePrinter<IdxTs...>::print(vs...);
    }

};

template <typename ... IdxTs>
struct IndexTypePrinter<float,IdxTs...> {

    static inline __NDT_CUDA_HD_PREFIX__ void print(float v0, IdxTs... vs) {
        std::cout << "float ";
        IndexTypePrinter<IdxTs...>::print(vs...);
    }

};

// -=-=-=-=-=-=-=-=-=-=-=- Tensor initializer -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename ... ArgTypes>
struct InitializeDimensions;

template <typename HeadArgType, typename ... TailArgTypes>
struct InitializeDimensions<HeadArgType, TailArgTypes...> {

    static inline Eigen::Matrix<uint, sizeof...(TailArgTypes), 1> Initialize(HeadArgType head, TailArgTypes ... tails) {
        return (Eigen::Matrix<uint,sizeof...(TailArgTypes),1>() << head, InitializeDimensions<TailArgTypes...>::Initialize(tails...)).finished();
    }

};

template <typename HeadArgType, typename TailArgType>
struct InitializeDimensions<HeadArgType, TailArgType> {

    static inline Eigen::Matrix<uint, 1, 1> Initialize(HeadArgType head, TailArgType tail) {
        return Eigen::Matrix<uint, 1, 1>(head);
    }

};

} // namespace internal

template <typename Scalar>
class TypedTensorBase {

};

template <typename Scalar, Residency R>
class TypedResidentTensorBase : public TypedTensorBase<Scalar> {

};

template <uint D, typename T, Residency R, bool Const = false>
class TensorView;

template <uint D, typename T, Residency R = HostResident, bool Const = false>
class Tensor
        : public TensorBase<Tensor<D,T,R,Const> >,
          public TypedResidentTensorBase<T,R> {
private:

    template <typename Derived>
    struct IsConvertibleToDimensions;

public:

    using BaseT = TensorBase<Tensor<D, T, R, Const> >;
    using DimT = typename BaseT::DimT;
    using IdxT = typename BaseT::IdxT;
    using DataT = T;

    friend BaseT;

//    template <typename Derived,
//              typename std::enable_if<IsConvertibleToDimensions<Derived>::Value, int>::type = 0>
//    __NDT_CUDA_HD_PREFIX__ Tensor(const Eigen::MatrixBase<Derived> & dimensions) : dimensions_(dimensions), data_(nullptr) { }
//
//    __NDT_CUDA_HD_PREFIX__ Tensor(const std::initializer_list<DimT> dimensions)
//            : dimensions_(dimensions), data_(nullptr) { }

    __NDT_CUDA_HD_PREFIX__ Tensor(const Eigen::Matrix<DimT, D, 1> & dimensions) : dimensions_(dimensions),
                                                                                  data_(nullptr) {}

    // construct with values, not valid for managed tensors
//    __NDT_CUDA_HD_PREFIX__ Tensor(const std::initializer_list<DimT> dimensions,
//                                  typename internal::ConstQualifier<T *,Const>::type data)
//            : dimensions_(dimensions), data_(data) { }

//    template <typename Derived,
//              typename std::enable_if<IsConvertibleToDimensions<Derived>::Value, int>::type = 0>
//    __NDT_CUDA_HD_PREFIX__ Tensor(const Eigen::MatrixBase<Derived> & dimensions,
//                                  typename internal::ConstQualifier<T *,Const>::type data) : dimensions_(dimensions), data_(data) { };

//    template <typename ... DimensionTypes,
//              typename std::enable_if<sizeof...(DimensionTypes) == D, int>::type = 0>
//    __NDT_CUDA_HD_PREFIX__ Tensor(DimensionTypes ... dimensions,
//                                  typename internal::ConstQualifier<T *, Const>::type data) : dimensions_(dimensions...),
//                                                                                              data_(data) {}

    template <typename ... ArgTypes,
              typename std::enable_if<sizeof...(ArgTypes) == (D+1), int>::type = 0>
    __NDT_CUDA_HD_PREFIX__ Tensor(ArgTypes ... args)
            : dimensions_(internal::InitializeDimensions<ArgTypes...>::Initialize(args...)),
              data_(std::get<D>(std::tuple<ArgTypes...>(args...))) { }

    __NDT_CUDA_HD_PREFIX__ Tensor(const Eigen::Matrix<DimT, D, 1> & dimensions,
                                  typename internal::ConstQualifier<T *, Const>::type data) : dimensions_(dimensions),
                                                                                              data_(data) {}

    // special case constructors for length-1 vectors
    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    __NDT_CUDA_HD_PREFIX__ Tensor(const DimT length) : dimensions_(Eigen::Matrix<DimT, D, 1>(length)), data_(nullptr) {}

    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    __NDT_CUDA_HD_PREFIX__ Tensor(const DimT length, typename internal::ConstQualifier<T *, Const>::type data) :
            dimensions_(Eigen::Matrix<DimT, D, 1>(length)), data_(data) {}

    // copy constructor and assignment operator, not valid for managed or const tensors
    template <bool _Const>
    __NDT_CUDA_HD_PREFIX__ Tensor(Tensor<D, T, R, _Const> & other)
            : dimensions_(other.Dimensions()), data_(other.Data()) {
        static_assert(Const || !_Const,
                      "Cannot copy-construct a non-const Tensor from a Const tensor");
    }

    template <bool _Const>
    __NDT_CUDA_HD_PREFIX__ inline Tensor<D, T, R, Const> & operator=(const Tensor<D, T, R, _Const> & other) {
        static_assert(Const || !_Const,
                      "Cannot assign a non-const Tensor from a Const tensor");
        dimensions_ = other.Dimensions();
        data_ = other.Data();
        return *this;
    }

    __NDT_CUDA_HD_PREFIX__ ~Tensor() {}

    // conversion to const tensor
    template <bool _Const = Const, typename std::enable_if<!_Const, int>::type = 0>
    inline operator Tensor<D, T, R, true>() const {
        return Tensor<D, T, R, true>(this->Dimensions(), Data());
    }

    template <typename U = T,
            typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T * Data() { return data_; }

    inline __NDT_CUDA_HD_PREFIX__ const T * Data() const { return data_; }

    // -=-=-=-=-=-=- sizing functions -=-=-=-=-=-=-
private:
    inline __NDT_CUDA_HD_PREFIX__ DimT DimensionSizeImpl(const IdxT dim) const {
        return dimensions_(dim);
    }

    inline __NDT_CUDA_HD_PREFIX__ const Eigen::Matrix<DimT, D, 1, Eigen::DontAlign> & DimensionsImpl() const {
        return dimensions_;
    }

public:
    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT Length() const {
        return dimensions_(0);
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT Width() const {
        return dimensions_(0);
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT Height() const {
        return dimensions_(1);
    }

    inline __NDT_CUDA_HD_PREFIX__ std::size_t Count() const {
//        return internal::count<DimT,D>(dimensions_);
        return dimensions_.prod();
    }

    inline __NDT_CUDA_HD_PREFIX__ std::size_t SizeBytes() const {
        return Count() * sizeof(T);
    }

    inline __NDT_CUDA_HD_PREFIX__ Eigen::Matrix<DimT, D, 1, Eigen::DontAlign> Strides() const {
        return internal::StrideConstructor<D>::Construct(1, dimensions_);
    }

    // -=-=-=-=-=-=- slicing functions -=-=-=-=-=-=-
    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline TensorView<D, T, R, Const> Slice(const IdxT start0, const DimT size0) {
        return TensorView<D, T, R, Const>(Tensor<D, T, R, Const>(size0, data_ + start0), Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline TensorView<D, T, R, true> Slice(const IdxT start0, const DimT size0) const {
        return TensorView<D, T, R, true>(Tensor<D, T, R, true>(size0, data_ + start0), Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline TensorView<D, T, R, Const> Slice(const IdxT start0, const IdxT start1, const DimT size0, const DimT size1) {
        return TensorView<D, T, R, Const>(Tensor<D, T, R, Const>({size0, size1}, &(*this)(start0, start1)), Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline TensorView<D, T, R, true>
    Slice(const IdxT start0, const IdxT start1, const DimT size0, const DimT size1) const {
        return TensorView<D, T, R, true>(Tensor<D, T, R, true>({size0, size1}, &(*this)(start0, start1)), Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline TensorView<D, T, R, Const> Slice(const IdxT start0, const IdxT start1, const IdxT start2,
                                            const DimT size0, const DimT size1, const DimT size2) {
        return TensorView<D, T, R, Const>(
                Tensor<D, T, R, Const>({size0, size1, size2}, &(*this)(start0, start1, start2)), Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline TensorView<D, T, R, true> Slice(const IdxT start0, const IdxT start1, const IdxT start2,
                                           const DimT size0, const DimT size1, const DimT size2) const {
        return TensorView<D, T, R, true>(Tensor<D, T, R, true>({size0, size1, size2}, &(*this)(start0, start1, start2)),
                                         Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline TensorView<D, T, R, Const> Slice(const IdxT start0, const IdxT start1, const IdxT start2, const IdxT start3,
                                            const DimT size0, const DimT size1, const DimT size2, const DimT size3) {
        return TensorView<D, T, R, Const>(
                Tensor<D, T, R, Const>({size0, size1, size2, size3}, &(*this)(start0, start1, start2, start3)),
                Strides());
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline TensorView<D, T, R, true> Slice(const IdxT start0, const IdxT start1, const IdxT start2, const IdxT start3,
                                           const DimT size0, const DimT size1, const DimT size2,
                                           const DimT size3) const {
        return TensorView<D, T, R, true>(
                Tensor<D, T, R, true>({size0, size1, size2, size3}, &(*this)(start0, start1, start2, start3)),
                Strides());
    }

    template <typename DerivedStart, typename DerivedSize,
              typename std::enable_if<internal::IsIntegralVectorType<DerivedStart,D>::Value &&
                                      internal::IsIntegralVectorType<DerivedSize,D>::Value, int>::type = 0>
    inline TensorView<D, T, R, Const> Slice(const Eigen::MatrixBase<DerivedStart> & start, const Eigen::MatrixBase<DerivedSize> & size) {
        return TensorView<D, T, R, Const>(Tensor<D, T, R, Const>(size, &this->Element(start)), Strides());
    }

    template <typename DerivedStart, typename DerivedSize,
            typename std::enable_if<internal::IsIntegralVectorType<DerivedStart,D>::Value &&
                                    internal::IsIntegralVectorType<DerivedSize,D>::Value, int>::type = 0>
    inline TensorView<D, T, R, true> Slice(const Eigen::MatrixBase<DerivedStart> & start, const Eigen::MatrixBase<DerivedSize> & size) const {
        return TensorView<D, T, R, true>(Tensor<D, T, R, true>(size, &this->Element(start)), Strides());
    }

    template <typename StartT = DimT, typename SizeT = DimT, int Options = 0, int D2 = D,
              typename std::enable_if<D2 == D && std::is_integral<SizeT>::value && std::is_integral<StartT>::value, int>::type = 0>
    inline TensorView<D, T, R, Const> Slice(const Eigen::Matrix<StartT, D2, 1, Options> & start, const Eigen::Matrix<SizeT, D2, 1, Options> & size) {
        return TensorView<D, T, R, Const>(Tensor<D, T, R, Const>(size, &this->Element(start)), Strides());
    }

    template <typename StartT = DimT, typename SizeT = DimT, int Options = 0, int D2 = D,
              typename std::enable_if<D2 == D && std::is_integral<SizeT>::value && std::is_integral<StartT>::value, int>::type = 0>
    inline TensorView<D, T, R, true> Slice(const Eigen::Matrix<StartT, D2, 1, Options> & start, const Eigen::Matrix<SizeT, D2, 1, Options> & size) const {
        return TensorView<D, T, R, true>(Tensor<D, T, R, true>(size, &this->Element(start)), Strides());
    }

    // -=-=-=-=-=-=- indexing functions -=-=-=-=-=-=-
    template <typename Derived,
              typename std::enable_if<internal::IsIntegralVectorType<Derived, D>::Value && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const Eigen::MatrixBase<Derived> & indices) {
        return data_[internal::OffsetXD(internal::IndexList<IdxT,D>(indices),
                                        internal::IndexList<IdxT,D-1>(dimensions_.template head<D-1>()))];
    }

    template <typename Derived,
              typename std::enable_if<internal::IsIntegralVectorType<Derived, D>::Value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const Eigen::MatrixBase<Derived> & indices) const {
        return data_[internal::OffsetXD(internal::IndexList<IdxT,D>(indices),
                                        internal::IndexList<IdxT,D-1>(dimensions_.template head<D-1>()))];
    }

    // TODO: replace all the following with variadic template-based versions
    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0) const {
        return data_[d0];
    }

    template <int D2 = D, typename std::enable_if<D2 == 1 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0) {
        return data_[d0];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0, const IdxT d1) const {
        return data_[internal::OffsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0, const IdxT d1) {
        return data_[internal::OffsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0, const IdxT d1, const IdxT d2) const {
        return data_[internal::OffsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 3 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0, const IdxT d1, const IdxT d2) {
        return data_[internal::OffsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) const {
        return data_[internal::OffsetXD<IdxT,DimT,4>(internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(d0,d1,d2,d3)),
                internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(dimensions_[0],dimensions_[1],dimensions_[2])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 4 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) {
        return data_[internal::OffsetXD<IdxT,DimT,4>(internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(d0,d1,d2,d3)),
                internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(dimensions_[0],dimensions_[1],dimensions_[2])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 5, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ const T & Element(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3, const IdxT d4) const {
        return data_[internal::OffsetXD<IdxT,DimT,5>(internal::IndexList<IdxT,5>(Eigen::Matrix<uint,5,1>(d0,d1,d2,d3,d4)),
                internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(dimensions_[0],dimensions_[1],dimensions_[2],dimensions_[3])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 5 && !Const, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T & Element(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3, const IdxT d4) {
        return data_[internal::OffsetXD<IdxT,DimT,5>(internal::IndexList<IdxT,5>(Eigen::Matrix<uint,5,1>(d0,d1,d2,d3,d4)),
                internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(dimensions_[0],dimensions_[1],dimensions_[2],dimensions_[3])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT offset(const IdxT d0, const IdxT d1) const {
        return internal::OffsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                                               internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT offset(const Eigen::MatrixBase<Derived> & indices) const {
        return offset(indices(0),indices(1));
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT offset(const IdxT d0, const IdxT d1, const IdxT d2) const {
        return internal::OffsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                                               internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT offset(const Eigen::MatrixBase<Derived> & indices) const {
        return offset(indices(0),indices(1),indices(2));
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT offset(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) const {
        return internal::OffsetXD<IdxT,DimT,4>(internal::IndexList<IdxT,4>(Eigen::Matrix<uint,4,1>(d0,d1,d2,d3)),
                                               internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(dimensions_[0],dimensions_[1],dimensions_[2])));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ DimT offset(const Eigen::MatrixBase<Derived> & indices) const {
        return offset(indices(0),indices(1),indices(2),indices(3));
    }

    // -=-=-=-=-=-=- interpolation functions -=-=-=-=-=-=-
    template <typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T Interpolate(const IdxTs ... vs) const {
        return internal::Interpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                     internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(std::tuple<IdxTs...>(vs...)));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T Interpolate(const Eigen::MatrixBase<Derived> & v) const {
        return internal::Interpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                     VectorToTuple(v.reverse()));
    }

    template <typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T InterpolateValidOnly(ValidityCheck check, IdxTs ... vs) const {

        return internal::InterpolateValidOnly(data_,internal::IndexList<DimT,D>(dimensions_.reverse()),
                                              check, internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(std::tuple<IdxTs...>(vs...)));

    }

    template <typename ValidityCheck, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ T InterpolateValidOnly(ValidityCheck check, const Eigen::MatrixBase<Derived> & v) const {
        return internal::InterpolateValidOnly(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                              check,VectorToTuple(v.reverse()));
    }

    template <typename Transformer, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename Transformer::ReturnType TransformInterpolate(Transformer transformer, const IdxTs ... vs) const /*- > decltype(Transformer::operator())*/ {
        return internal::TransformInterpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), transformer,
                                              internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(std::tuple<IdxTs...>(vs...)));
    }

    template <typename Transformer, typename Derived,
            typename std::enable_if<internal::IsRealVectorType<Derived,D>::Value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename Transformer::ReturnType TransformInterpolate(Transformer transformer, const Eigen::MatrixBase<Derived> & v) const /*- > decltype(Transformer::operator())*/ {
        return internal::TransformInterpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), transformer,
                                              VectorToTuple(v.reverse()));
    }

    template <typename Transformer, typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename Transformer::ReturnType TransformInterpolateValidOnly(Transformer transformer, ValidityCheck check, IdxTs ... vs) const {

        return internal::TransformInterpolateValidOnly(data_,internal::IndexList<DimT,D>(dimensions_.reverse()),
                                                       transformer, check,
                                                       internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(std::tuple<IdxTs...>(vs...)));

    }

    // -=-=-=-=-=-=- interpolation derivative functions -=-=-=-=-=-=-
    template <typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientTraits<T,D>::GradientType
    InterpolationGradient(const IdxTs ... vs) const {
        typename internal::GradientTraits<T,D>::GradientType gradient;
        internal::InterpolationGradientFiller<D, 0>::Fill(data_, dimensions_, std::tuple<IdxTs...>(vs...), gradient);
        return gradient;
    }

    template <typename Derived,
              typename std::enable_if<internal::IsVectorType<Derived,D>::Value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientTraits<T,D>::GradientType
    InterpolationGradient(const Eigen::MatrixBase<Derived> & v) const {
        typename internal::GradientTraits<T,D>::GradientType gradient;
        internal::InterpolationGradientFiller<D, 0>::Fill(data_, dimensions_, VectorToTuple(v), gradient);
        return gradient;
    }

    template <typename Transformer, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientTraits<decltype(std::declval<Transformer>()(std::declval<T>())),D>::GradientType
    TransformInterpolationGradient(Transformer transformer, const IdxTs ... vs) const {
        using TransformedType = decltype(transformer(*data_));
        typename internal::GradientTraits<TransformedType,D>::GradientType gradient;
        internal::TransformInterpolationGradientFiller<D, 0>::Fill(transformer, data_, dimensions_, std::tuple<IdxTs...>(vs...), gradient);
        return gradient;
    }

    template <typename Transformer, typename Derived,
              typename std::enable_if<internal::IsVectorType<Derived,D>::Value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientTraits<decltype(std::declval<Transformer>()(std::declval<T>())),D>::GradientType
    TransformInterpolationGradient(Transformer transformer, const Eigen::MatrixBase<Derived> & v) const {
        using TransformedType = decltype(transformer(*data_));
        typename internal::GradientTraits<TransformedType,D>::GradientType gradient;
        internal::TransformInterpolationGradientFiller<D, 0>::Fill(transformer, data_, dimensions_, VectorToTuple(v), gradient);
        return gradient;
    }

    // -=-=-=-=-=-=-  -=-=-=-=-=-=-

    // TODO: is this needed?
    template <typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ bool validForInterpolation(ValidityCheck check, const IdxTs ... vs) {
        return internal::validForInterpolation(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), check,
                                               internal::TupleReverser<std::tuple<IdxTs...> >::Reverse(std::tuple<IdxTs...>(vs...)));
    }

    // -=-=-=-=-=-=- bounds-checking functions -=-=-=-=-=-=-
    template <typename PosHead, typename ... PosTail,
              typename std::enable_if<sizeof...(PosTail) == (D-1) && std::is_fundamental<PosHead>::value, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ bool InBounds(PosHead head, PosTail... tail) const {
        return internal::BoundsCheck(internal::IndexList<DimT,D>(dimensions_),
                                     std::tuple<PosHead,PosTail...>(head,tail...), 0, 0);
    }

    template <typename ... PosTs,
              typename std::enable_if<sizeof...(PosTs) == (D+1), int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ bool InBounds(PosTs... pos) const {
        const std::tuple<PosTs...> posTuple(pos...);
        const auto border = std::get<sizeof...(PosTs)-1>(posTuple);
        return internal::BoundsCheck(internal::IndexList<DimT,D>(dimensions_),
                                     internal::TupleSubset(posTuple, typename internal::IntegerList<0,sizeof...(PosTs)-2>::Type()),
                                     border, border);
    }

    template <typename BorderT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ bool InBounds(const Eigen::MatrixBase<Derived> & pos, BorderT border) const {
        return internal::BoundsCheck(internal::IndexList<DimT,D>(dimensions_),VectorToTuple(pos),border,border);
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ bool InBounds(const Eigen::MatrixBase<Derived> & pos) const {
        return internal::BoundsCheck(internal::IndexList<DimT,D>(dimensions_),VectorToTuple(pos),0,0);
    }

#define __NDT_TENSOR_DIFFERENCE_BOUNDS_DEFINITION__(DiffType) \
    template <typename PosHead, typename ... PosTail, \
              typename std::enable_if<sizeof...(PosTail) == (D-1) && std::is_fundamental<PosHead>::value, int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ bool In##DiffType##DifferenceBounds(PosHead head, PosTail... tail) const { \
        return internal::BoundsCheck(internal::IndexList<DimT,D>(dimensions_), \
                                     std::tuple<PosHead,PosTail...>(head,tail...), \
                                     internal::DifferenceTypeTraits<internal::DiffType##Difference>::borderLow, \
                                     internal::DifferenceTypeTraits<internal::DiffType##Difference>::borderHigh); \
    } \
    \
    template <typename Derived, \
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D && \
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ bool In##DiffType##DifferenceBounds(const Eigen::MatrixBase<Derived> & pos) const { \
        return internal::BoundsCheck(internal::IndexList<DimT,D>(dimensions_),VectorToTuple(pos), \
                                     internal::DifferenceTypeTraits<internal::DiffType##Difference>::borderLow, \
                                     internal::DifferenceTypeTraits<internal::DiffType##Difference>::borderHigh); \
    }

    __NDT_TENSOR_DIFFERENCE_BOUNDS_DEFINITION__(Backward)
    __NDT_TENSOR_DIFFERENCE_BOUNDS_DEFINITION__(Central)
    __NDT_TENSOR_DIFFERENCE_BOUNDS_DEFINITION__(Forward)

    // -=-=-=-=-=-=- gradient functions -=-=-=-=-=-=-
#define __NDT_TENSOR_DIFFERENCE_DEFINITION__(DiffType) \
    template <typename Derived, \
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D && \
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientTraits<T,D>::GradientType DiffType##Difference(const Eigen::MatrixBase<Derived> & v) const { \
        \
        return internal::GradientComputer<T,D,internal::DiffType##Difference>::compute(data_,dimensions_,VectorToTuple(v)); \
        \
    } \
    \
    template <typename ... IdxTs, \
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientTraits<T,D>::GradientType DiffType##Difference(const IdxTs ... v) const { \
        \
        return internal::GradientComputer<T,D,internal::DiffType##Difference>::template compute<IdxTs...>(data_,dimensions_,std::tuple<IdxTs...>(v...)); \
        \
    } \
    \
    template <typename Transformer, \
              typename Derived, \
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D && \
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::GradientType Transform##DiffType##Difference(Transformer transformer, const Eigen::MatrixBase<Derived> & v) const { \
        \
        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::TransformCompute(transformer,data_,dimensions_,VectorToTuple(v)); \
        \
    } \
    \
    template <typename Transformer, \
              typename ... IdxTs, \
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::GradientType Transform##DiffType##Difference(Transformer transformer, const IdxTs ... v) const { \
        \
        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::template TransformCompute<Transformer,IdxTs...>(transformer,data_,dimensions_,std::tuple<IdxTs...>(v...)); \
        \
    } \
    \
    template <typename Transformer, \
              typename ValidityCheck, \
              typename Derived, \
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D && \
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::GradientType Transform##DiffType##DifferenceValidOnly(Transformer transformer, ValidityCheck check, const Eigen::MatrixBase<Derived> & v) const { \
        \
        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::TransformComputeValidOnly(transformer,check,data_,dimensions_,VectorToTuple(v)); \
        \
    } \
    \
    template <typename Transformer, \
              typename ValidityCheck, \
              typename ... IdxTs, \
              typename std::enable_if<sizeof...(IdxTs) == D,int>::type = 0> \
    inline __NDT_CUDA_HD_PREFIX__ typename internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::GradientType Transform##DiffType##DifferenceValidOnly(Transformer transformer, ValidityCheck check, const IdxTs ... v) const { \
        \
        return internal::GradientComputer<typename Transformer::ReturnType,D,internal::DiffType##Difference>::template TransformComputeValidOnly<Transformer,ValidityCheck,IdxTs...>(transformer,check,data_,dimensions_,std::tuple<IdxTs...>(v...)); \
        \
    }

    __NDT_TENSOR_DIFFERENCE_DEFINITION__(Backward)
    __NDT_TENSOR_DIFFERENCE_DEFINITION__(Central)
    __NDT_TENSOR_DIFFERENCE_DEFINITION__(Forward)

    // -=-=-=-=-=-=- pointer manipulation functions -=-=-=-=-=-=-
    template <typename U = T,
              typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __NDT_CUDA_HD_PREFIX__ void SetDataPointer(T * data) { data_ = data; }

    // -=-=-=-=-=-=- copying functions -=-=-=-=-=-=-
    template <Residency R2, bool Const2, bool Check=false>
    inline void CopyFrom(const Tensor<D,T,R2,Const2> & other) {
        static_assert(!Const,"you cannot copy to a const tensor");
        internal::EquivalenceChecker<Check>::template CheckEquivalentSize<DimT,D>(this->Dimensions(),other.Dimensions());
        internal::Copier<T,R,R2>::Copy(data_,other.Data(),Count());
    }

    template <Residency R2, bool Const2, bool Check=false>
    inline void CopyFrom(const TensorView<D,T,R2,Const2> & view) {
        static_assert(!Const,"you cannot copy to a const tensor");
        internal::EquivalenceChecker<Check>::template CheckEquivalentSize<DimT,D>(this->Dimensions(),view.Dimensions());
        internal::SliceCopier::Copy(*this, view);
    }

protected:

    Eigen::Matrix<DimT,D,1,Eigen::DontAlign> dimensions_;
    typename internal::ConstQualifier<T *,Const>::type data_;

//public:

//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    // TODO: I think this is duplicated
    template <typename Derived>
    struct IsConvertibleToDimensions {
        static constexpr bool Value = Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value;
    };

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

    ManagedTensor(ManagedTensor && other)
        : Tensor<D,T,R,false>(other.dimensions_, other.data_) {
        other.dimensions_ = Eigen::Matrix<DimT,D,1>::Zero();
        other.data_ = nullptr;
    }

    ManagedTensor & operator=(ManagedTensor && other) {
        this->dimensions_ = other.dimensions_;
        other.dimensions_ = Eigen::Matrix<DimT,D,1>::Zero();
        this->data_ = other.data_;
        other.data_ = nullptr;
    }

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

// -=-=-=-=- traits -=-=-=-=-
template <uint D_, typename T_, Residency R_, bool Const_>
struct TensorTraits<Tensor<D_, T_, R_, Const_> > {
    static constexpr uint D = D_;
    using T = T_;
    static constexpr Residency R = R_;
    static constexpr bool Const = Const_;
};

// -=-=-=-=- full tensor typedefs -=-=-=-=-
#define __NDT_TENSOR_TYPEDEFS___(i, type, appendix)                                       \
    typedef Tensor<i,type,HostResident> Tensor##i##appendix;                      \
    typedef Tensor<i,type,DeviceResident> DeviceTensor##i##appendix;              \
    typedef Tensor<i,type,HostResident,true> ConstTensor##i##appendix;            \
    typedef Tensor<i,type,DeviceResident,true> ConstDeviceTensor##i##appendix;    \
    typedef ManagedTensor<i,type,HostResident> ManagedTensor##i##appendix;        \
    typedef ManagedTensor<i,type,DeviceResident> ManagedDeviceTensor##i##appendix

#define __NDT_TENSOR_TYPEDEFS__(type, appendix)  \
    __NDT_TENSOR_TYPEDEFS___(1, type, appendix); \
    __NDT_TENSOR_TYPEDEFS___(2, type, appendix); \
    __NDT_TENSOR_TYPEDEFS___(3, type, appendix); \
    __NDT_TENSOR_TYPEDEFS___(4, type, appendix); \
    __NDT_TENSOR_TYPEDEFS___(5, type, appendix)

__NDT_TENSOR_TYPEDEFS__(float,f);
__NDT_TENSOR_TYPEDEFS__(double,d);
__NDT_TENSOR_TYPEDEFS__(int,i);
__NDT_TENSOR_TYPEDEFS__(uint,ui);
__NDT_TENSOR_TYPEDEFS__(unsigned char,uc);

template <int D, typename Scalar>
using DeviceTensor = Tensor<D,Scalar,DeviceResident>;

template <int D, typename Scalar>
using ConstTensor = Tensor<D,Scalar,HostResident, true>;

template <int D, typename Scalar>
using ConstDeviceTensor = Tensor<D,Scalar,DeviceResident, true>;

#define __NDT_TENSOR_PARTIAL_TYPEDEF___(i,residency)                                         \
    template <typename Scalar>                                                       \
    using residency##Tensor##i = Tensor<i,Scalar,residency##Resident>;               \
    template <typename Scalar>                                                       \
    using Const##residency##Tensor##i = Tensor<i,Scalar,residency##Resident,true>;   \
    template <typename Scalar>                                                       \
    using Managed##residency##Tensor##i = ManagedTensor<i,Scalar,residency##Resident>

#define __NDT_TENSOR_PARTIAL_TYPEDEF__(i)                  \
    __NDT_TENSOR_PARTIAL_TYPEDEF___(i,Device);             \
    __NDT_TENSOR_PARTIAL_TYPEDEF___(i,Host)

//template <typename Scalar>
//using DeviceTensor2 = Tensor<2,Scalar,DeviceResident>;

__NDT_TENSOR_PARTIAL_TYPEDEF__(1);
__NDT_TENSOR_PARTIAL_TYPEDEF__(2);
__NDT_TENSOR_PARTIAL_TYPEDEF__(3);
__NDT_TENSOR_PARTIAL_TYPEDEF__(4);
__NDT_TENSOR_PARTIAL_TYPEDEF__(5);

#define __NDT_DIMENSIONAL_ALIAS__(dimension,alias) \
    template <typename Scalar> \
    using alias = HostTensor##dimension<Scalar>; \
    \
    template <typename Scalar> \
    using Managed##alias = ManagedHostTensor##dimension<Scalar>; \
    \
    template <typename Scalar> \
    using Device##alias = DeviceTensor##dimension<Scalar>; \
    \
    template <typename Scalar> \
    using ManagedDevice##alias = ManagedDeviceTensor##dimension<Scalar>; \
    \
    template <typename Scalar> \
    using Const##alias = ConstHostTensor##dimension<Scalar>; \
    \
    template <typename Scalar> \
    using ConstDevice##alias = ConstDeviceTensor##dimension<Scalar>

__NDT_DIMENSIONAL_ALIAS__(1,Vector);
__NDT_DIMENSIONAL_ALIAS__(2,Image);
__NDT_DIMENSIONAL_ALIAS__(3,Volume);


} // namespace NDT

#include <NDT/TensorView.h>

#undef __NDT_CUDA_HD_PREFIX__
