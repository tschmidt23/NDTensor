#pragma once

namespace NDT {

namespace internal {

template <Residency DestR, Residency SrcR>
struct CopyTypeTraits;

#ifndef __NDT_NO_CUDA__
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
#endif // __NDT_NO_CUDA__

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

template <int I, typename DestDerived, typename SrcDerived>
struct SliceCopyLoop {

    template <int D = TensorTraits<DestDerived>::D,
            typename IndicesDerived,
            typename std::enable_if<D == TensorTraits<SrcDerived>::D &&
                                    std::is_same<typename TensorTraits<DestDerived>::T,
                                            typename TensorTraits<SrcDerived>::T>::value &&
                                    !TensorTraits<DestDerived>::Const &&
                                    Eigen::internal::traits<IndicesDerived>::RowsAtCompileTime == D-1-I &&
                                    Eigen::internal::traits<IndicesDerived>::ColsAtCompileTime == 1,
                    int>::type = 0>
    inline static void Copy(TensorBase<DestDerived> & dest, const TensorBase<SrcDerived> & src,
                            const Eigen::MatrixBase<IndicesDerived> & indices) {

        for (int i = 0; i < dest.DimensionSize(I); ++i) {

            SliceCopyLoop<I-1, DestDerived, SrcDerived>::Copy(dest, src, (Eigen::Matrix<uint,D-I,1>() << i, indices).finished());

        }

    }

};

template <typename DestDerived, typename SrcDerived>
struct SliceCopyLoop<0, DestDerived, SrcDerived> {

    template <int D = TensorTraits<DestDerived>::D,
            typename IndicesDerived,
            typename std::enable_if<D == TensorTraits<SrcDerived>::D &&
                                    std::is_same<typename TensorTraits<DestDerived>::T,
                                            typename TensorTraits<SrcDerived>::T>::value &&
                                    !TensorTraits<DestDerived>::Const &&
                                    Eigen::internal::traits<IndicesDerived>::RowsAtCompileTime == D-1 &&
                                    Eigen::internal::traits<IndicesDerived>::ColsAtCompileTime == 1,
                    int>::type = 0>
    inline static void Copy(TensorBase<DestDerived> & dest, const TensorBase<SrcDerived> & src,
                            const Eigen::MatrixBase<IndicesDerived> & indices) {

        using T = typename TensorTraits<DestDerived>::T;
        static constexpr Residency DestR = TensorTraits<DestDerived>::R;
        static constexpr Residency SrcR = TensorTraits<SrcDerived>::R;

        Copier<T, DestR, SrcR>::Copy(&dest( (Eigen::Matrix<uint,D,1>() << 0, indices ).finished() ),
                                     &src( (Eigen::Matrix<uint,D,1>() << 0, indices).finished() ),
                                     dest.DimensionSize(0));

    }

};

struct SliceCopier {

    template <typename DestDerived, typename SrcDerived>
    inline static void Copy(TensorBase<DestDerived> & dest, const TensorBase<SrcDerived> & src) {

        SliceCopyLoop<TensorTraits<DestDerived>::D-1, DestDerived, SrcDerived>::Copy(dest, src, Eigen::Matrix<uint,0,1>());

    }

};

} // namespace internal

} // namespace NDT