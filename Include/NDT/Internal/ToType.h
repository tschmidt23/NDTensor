#pragma once

namespace NDT {

namespace internal {

template <typename T>
struct TypeToType {

    using Type = T;

};

template <int I>
struct IntToType {

    static constexpr int Int = I;

};

} // namespace internal

} // namespace NDT