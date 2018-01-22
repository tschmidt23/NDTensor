#pragma once

namespace NDT {

namespace internal {

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

} // namespace internal

} // namespace NDT