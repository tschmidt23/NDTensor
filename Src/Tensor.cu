#include <NDT/Tensor.h>

#ifndef __NDT_NO_CUDA__

namespace NDT {

namespace internal {

template <typename T>
void Filler<T, DeviceResident>::Fill(T * data, const std::size_t N, const T & value) {
    thrust::fill_n(thrust::device_ptr<T>(data), N, value);
};

template class Filler<float, DeviceResident>;
template class Filler<double, DeviceResident>;
template class Filler<int, DeviceResident>;
template class Filler<unsigned int, DeviceResident>;
template class Filler<long long int, DeviceResident>;
template class Filler<unsigned long long int, DeviceResident>;

} // namespace internal

} // namespace NDT

#endif // __NDT_NO_CUDA__
