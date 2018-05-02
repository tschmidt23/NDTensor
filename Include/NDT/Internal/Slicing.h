#pragma once

namespace NDT {

namespace internal {

template <uint D>
struct StrideConstructor {

    static inline Eigen::Matrix<uint, D, 1, Eigen::DontAlign>
    Construct(const uint soFar, const Eigen::Matrix<uint, D, 1, Eigen::DontAlign> & dimensions) {

        return (Eigen::Matrix<uint, D, 1, Eigen::DontAlign>() << soFar, StrideConstructor<D - 1>::Construct(
                soFar * dimensions(0), dimensions.template tail<D - 1>())).finished();

    }

};

template <>
struct StrideConstructor<1> {

    static inline Eigen::Matrix<uint, 1, 1, Eigen::DontAlign>
    Construct(const uint soFar, const Eigen::Matrix<uint, 1, 1, Eigen::DontAlign> & /*dimensions*/) {

        return Eigen::Matrix<uint, 1, 1, Eigen::DontAlign>(soFar);

    }

};

template <int D, int Axis>
struct AxisDropper {

    static inline Eigen::Matrix<uint, D-1, 1, Eigen::DontAlign>
    Drop(const Eigen::Matrix<uint, D, 1, Eigen::DontAlign> & dimensions) {
        return (Eigen::Matrix<uint, D-1, 1, Eigen::DontAlign>() <<
            dimensions.template head<Axis>(), dimensions.template tail<D-Axis-1>()).finished();
    }

};

template <int D>
struct AxisDropper<D, 0> {

    static inline Eigen::Matrix<uint, D-1, 1, Eigen::DontAlign>
    Drop(const Eigen::Matrix<uint, D, 1, Eigen::DontAlign> & dimensions) {
        return dimensions.template tail<D-1>();
    }

};

//template <int D>
//struct AxisDropper<D, MinusOne<D>::Value> {
//
//    static inline Eigen::Matrix<uint, D-1, 1, Eigen::DontAlign>
//    Drop(const Eigen::Matrix<uint, D, 1, Eigen::DontAlign> & dimensions) {
//        return dimensions.template head<D-1>();
//    }
//
//};

template <int D, int Axis>
struct AxisEmplacer {

    static inline Eigen::Matrix<uint, D, 1, Eigen::DontAlign>
    Emplace(const uint val) {
        return (Eigen::Matrix<uint, D, 1, Eigen::DontAlign>() <<
                Eigen::Matrix<uint, Axis, 1, Eigen::DontAlign>::Zero(),
                val,
                Eigen::Matrix<uint, D-Axis-1, 1, Eigen::DontAlign>::Zero()).finished();
    }

};

} // namespace internal

} // namespace NDT