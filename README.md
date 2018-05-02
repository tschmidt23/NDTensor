# NDTensor

NDTensor is a C++ header library for N-Dimensional tensors. The templated `Tensor` class implements a number of functions for interacting with tensors, most of which are written generically (i.e. they are generated at compile time given the known dimensionality and type of the tensor).

A `Tensor` can be used to reference memory managed elsewhere, as in the following example:

```
std::vector<float> tensorData(width * height);
NDT::Tensor<2,float> image({width,height}, tensorData.data());
```

### Managed Memory

The library also provides a `ManagedTensor` class which automatically allocates memory in the constructor given the tensor dimensions, and deallocates the memory in the destructor:

```
NDT::ManagedTensor<2,float> image({width,height});
```

The `ManagedTensor` class both derives from and provides implicit conversion to a `Tensor` of the same time and dimensionality, such that a `ManagedTensor` class can be safely passed to a function expecting either a pointer or reference to a `Tensor` or an actual `Tensor` object. Copy construction and assignment for `ManagedTensor` are disabled, but move construction and assignment are available and transfer ownership of the allocated memory.

### Aliases

There are aliases for low-dimensional tensors according to more commonly used names, i.e. `Tensor<1,T>` is also known as `Vector<T>`,  `Tensor<2,T>` is also known as `Image<T>` and `Tensor<3,T>` is also known as `Volume<T>`. The corresponding aliases also exist for `ManagedTensor`.

### Element Access

The `Tensor` class overloads `operator()`, which is called with a number of parameters depending on the tensor dimensionality, or an Eigen vector with the corresponding length. For example:

```
NDT::ManagedVector<float> vector(10);
vector(5) = 0.f;

NDT::ManagedImage<Eigen::Vector3f> image({10,10});
image(5,5) = Eigen::Vector3f(0,0,0);
image(Eigen::Vector2i(3,3)) = Eigen::Vector3f(0,0,0);

NDT::ManagedVolume<uint> volume({10,10,10});
volume(5,5,5) = 0;
volume(Eigen::Vector3i(3,3,3)) = 0;
```

The `Interpolate` function can be used to do linear interpolation:

```
NDT::ManagedImage<float> image({10,10});
std::cout << image.Interpolate(4.3f, 2.1f) << std::endl;
```

### GPU Memory

If compiled with CUDA support, a third tensor allows a `Tensor` or a `ManagedTensor` to reference GPU memory. For `ManagedTensor`, memory is automatically allocated on the host or device depending on this template parameter, but for `Tensor`, it is up to the library user to ensure that the pointer passed into the constructor is in the appropriate parameter space. Copying between host and device is easy, as shown in the following example:

```
// allocate memory
NDT::ManagedVolume<int> hVolume({4,4,4});
NDT::ManagedDeviceVolume<int> dVolume({4,4,4});

// fill the host volume
...

// copy from host to device
dVolume.CopyFrom(hVolume);

```

Normal rules for element access on host / device apply, i.e. host memory can only be accessed from code executing on the host and device memory can only be accessed from a CUDA kernel executing on the device.

### Slicing and Tensor Views

The `Tensor` class assumes packed memory, such that the stride of the first dimension is the size of one element and the total size of the memory referenced is the product of the dimensions times the size of one element. The `TensorView` class, on the other hand, decouples the stride and dimensions of the tensor. A `TensorView` is created using the `SubTensor` member function of the `Tensor` class.

```
float data[16] = {  0,  1,  2,  3,
                    4,  5,  6,  7,
                    8,  9, 10, 11,
                   12, 13, 14, 15 };

NDT::Image<float> image({4,4}, data);

NDT::TensorView<2,float> imageView(2, 1,  2, 2);
std::cout << imageView(0,0) << std::endl; // prints 6
std::cout << imageView(1,1) << std::endl; // prints 11
```

One particularly useful use case is in copying to/from a part of a tensor:

```
NDT::ManagedVector<float> vector(10);
NDT::ManagedVector<float> smallerVector(2);
smallerVector(0) = 3.f;
smallerVector(1) = 4.f;

vector.SubTensor( 3, 2 ).CopyFrom(smallerVector);

NDT::ManagedImage<int> image({100,100});
NDT::ManagedImage<int> patch({10,10});

patch.CopyFrom(image.SubTensor(17, 25,  10, 10));


```
