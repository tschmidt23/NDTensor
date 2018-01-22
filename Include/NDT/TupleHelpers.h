#pragma once

#include <tuple>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define __NDT_CUDA_HD_PREFIX__ __host__ __device__
#else
#define __NDT_CUDA_HD_PREFIX__
#endif // __CUDACC__


#include <Eigen/Core>

namespace NDT {

namespace internal {

template <int... N>
struct IntegerListConstructor {
    template <int M>
    struct PushBack {
        typedef IntegerListConstructor<N...,M> Type;
    };
};

template <int Min, int Max>
struct IntegerList {
    typedef typename IntegerList<Min,Max-1>::Type::template PushBack<Max>::Type Type;
};

template <int Min>
struct IntegerList<Min,Min> {
    typedef IntegerListConstructor<Min> Type;
};

template <int...Indices, typename Tuple>
__NDT_CUDA_HD_PREFIX__ inline auto TupleSubset(const Tuple & tuple, IntegerListConstructor<Indices...>)
    -> decltype(std::make_tuple(std::get<Indices>(tuple)...)) {
    return std::make_tuple(std::get<Indices>(tuple)...);
}

template <typename Head>
__NDT_CUDA_HD_PREFIX__
inline std::tuple<> GetTail(const std::tuple<Head> & /*t*/) {
    return std::tuple<>();
}

template <typename Head, typename ... Tail>
__NDT_CUDA_HD_PREFIX__
inline std::tuple<Tail...> GetTail(const std::tuple<Head,Tail...> & t) {

    return TupleSubset(t, typename IntegerList<1,sizeof...(Tail)>::Type());

}

template <typename Tail>
__NDT_CUDA_HD_PREFIX__
inline std::tuple<> GetHead(const std::tuple<Tail> & /*t*/) {
    return std::tuple<>();
}

template <typename ... Head, typename Tail>
__NDT_CUDA_HD_PREFIX__
inline std::tuple<Head...> GetHead(const std::tuple<Head...,Tail> & t) {

    return TupleSubset(t, typename IntegerList<0,sizeof...(Head)-1>::Type());

}

template <typename TupleT>
struct TupleReverser;

template <typename T>
struct TupleReverser<std::tuple<T> > {
    using Type = std::tuple<T>;

    static __NDT_CUDA_HD_PREFIX__ inline Type Reverse(const std::tuple<T> & t) {
        return t;
    }

};

template <typename Head, typename ... Tail>
struct TupleReverser<std::tuple<Head,Tail...> > {

    using HeadTuple = std::tuple<Head>;
    using TailTuple = typename TupleReverser<std::tuple<Tail...> >::Type;

    using Type = decltype(std::tuple_cat(std::declval<TailTuple>(), std::declval<HeadTuple>()));

    static __NDT_CUDA_HD_PREFIX__ inline Type Reverse(const std::tuple<Head,Tail...> & t) {

        return std::tuple_cat(TupleReverser<std::tuple<Tail...> >::Reverse(GetTail(t)), std::tuple<Head>(std::get<0>(t)));

    }

};

template <typename QueryType, std::size_t CheckIndex, typename ... TupleTypes>
struct TupleIndexHunter;

template <typename QueryType, std::size_t CheckIndex, typename FirstType, typename ... TupleTypes>
struct TupleIndexHunter<QueryType,CheckIndex,FirstType,TupleTypes...> {

    static constexpr auto Index = TupleIndexHunter<QueryType,CheckIndex+1,TupleTypes...>::Index;

};

template <typename QueryType, std::size_t CheckIndex, typename ... TupleTypes>
struct TupleIndexHunter<QueryType,CheckIndex,QueryType,TupleTypes...> {

    static constexpr auto Index = CheckIndex;

};

template <std::size_t Index, typename ... TupleTypes>
struct DeviceExecutableTupleCopy {

    static inline __NDT_CUDA_HD_PREFIX__ void Copy(std::tuple<TupleTypes...> & destination,
                                                const std::tuple<TupleTypes...> & source) {
        std::get<Index>(destination) = std::get<Index>(source);
        DeviceExecutableTupleCopy<Index-1,TupleTypes...>::Copy(destination,source);
    }

};

template <typename ... TupleTypes>
struct DeviceExecutableTupleCopy<std::numeric_limits<std::size_t>::max(),TupleTypes...> {

    static inline __NDT_CUDA_HD_PREFIX__ void Copy(std::tuple<TupleTypes...> & /*destination*/,
                                                const std::tuple<TupleTypes...> & /*source*/) {
//        std::get<0>(destination) = std::get<0>(source);
    }

};


template <typename Scalar, int D>
struct TupledType {

    using HeadTuple = std::tuple<Scalar>;
    using TailTuple = typename TupledType<Scalar,D-1>::Type;

    using Type = decltype(std::tuple_cat(std::declval<HeadTuple>(), std::declval<TailTuple>()));

};

template <typename Scalar>
struct TupledType<Scalar, 1> {

    using Type = std::tuple<Scalar>;

};

/*
 * The TypeListIndex struct allows for extracting the type at a given index in a variadic type list.
 * For example, TypeListIndex<1,float,int,double>::Type is equivalent to float.
 */
template <int I, typename ... TypleTypes>
struct TypeListIndex;

template <int I, typename HeadType, typename ... TailTypes>
struct TypeListIndex<I,HeadType,TailTypes...> {

    using Type = typename TypeListIndex<I-1, TailTypes...>::Type;

};

template <typename HeadType, typename ... TailTypes>
struct TypeListIndex<0,HeadType,TailTypes...> {

    using Type = HeadType;

};

/*
 * The TupleTypeSubstitute struct allows substituting one of the types in the type list of a tuple.
 * For example, TupleTypeSubstitute<1,int,float,float,float>::Type is equivalent to
 * std::tuple<float,int,float>.
 *
 * WARNING: will generate compile-time errors if SubsitutionIndex is less than 0 or greater than or
 * equal to the length of the list.
 */
template <int SubstitutionIndex, typename SubstitutionType, typename ...TupleTypes>
struct TupleTypeSubstitute;

template <int SubstitutionIndex, typename SubstitutionType, typename HeadType, typename ... TailTypes>
struct TupleTypeSubstitute<SubstitutionIndex,SubstitutionType,HeadType,TailTypes...> {

    using HeadTupleType = std::tuple<HeadType>;
    using TailTupleType = typename TupleTypeSubstitute<SubstitutionIndex-1, SubstitutionType, TailTypes...>::Type;

    using Type = decltype(std::tuple_cat(std::declval<HeadTupleType>(), std::declval<TailTupleType>()));

};

template <typename SubstitutionType, typename HeadType, typename ... TailTypes>
struct TupleTypeSubstitute<0,SubstitutionType,HeadType,TailTypes...> {

    using HeadTupleType = std::tuple<SubstitutionType>;
    using TailTupleType = std::tuple<TailTypes...>;

    using Type = decltype(std::tuple_cat(std::declval<HeadTupleType>(), std::declval<TailTupleType>()));

};

} // namespace internal

template <typename QueryType, typename ... TupleTypes>
__NDT_CUDA_HD_PREFIX__ inline
QueryType & GetByType(std::tuple<TupleTypes...> & tuple) {

    return std::get<internal::TupleIndexHunter<QueryType,0,TupleTypes ...>::Index>(tuple);

}

template <typename QueryType, typename ... TupleTypes>
__NDT_CUDA_HD_PREFIX__ inline
const QueryType & GetByType(const std::tuple<TupleTypes...> & tuple) {

    return std::get<internal::TupleIndexHunter<QueryType,0,TupleTypes ...>::Index>(tuple);

}

template <typename ... TupleTypes>
__NDT_CUDA_HD_PREFIX__ inline
void Copy(std::tuple<TupleTypes...> & destination, const std::tuple<TupleTypes...> & source) {
    internal::DeviceExecutableTupleCopy<sizeof...(TupleTypes)-1,TupleTypes...>::Copy(destination,source);
}


template <typename Derived,
          typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 1 &&
                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
__NDT_CUDA_HD_PREFIX__ inline
typename internal::TupledType<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime>::Type
VectorToTuple(const Eigen::MatrixBase<Derived> & vector) {

    return std::tuple<typename Eigen::internal::traits<Derived>::Scalar>(vector(0));

}

template <typename Derived,
          typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime >= 2 &&
                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
__NDT_CUDA_HD_PREFIX__ inline
typename internal::TupledType<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime>::Type
VectorToTuple(const Eigen::MatrixBase<Derived> & vector) {

    return std::tuple_cat(std::tuple<typename Eigen::internal::traits<Derived>::Scalar>(vector(0)),
                          VectorToTuple(vector.template tail<Eigen::internal::traits<Derived>::RowsAtCompileTime-1>()));

}

} // namespace NDT

#undef __NDT_CUDA_HD_PREFIX__
