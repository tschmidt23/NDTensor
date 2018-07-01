#pragma once

#include <NDT/Tensor.h>

namespace NDT {

template <uint D, typename T, Residency R, bool Const>
Tensor<D+1,T,R,Const> ExpandDims(NDT::Tensor<D,T,R,Const> & tensor, const int index) {
    return Tensor<D+1,T,R,Const>(
            (Eigen::Matrix<unsigned int, D + 1, 1>() <<
                    tensor.Dimensions().head(index), 1, tensor.Dimensions().tail(D-index)).finished(),
        tensor.Data());
};

} // namespace NDT