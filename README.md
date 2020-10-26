# DeepJulia
A very simple DL package in Julia. The API is heavily inspired by the one and only [PyTorch](pytorch.org) library.


## Installation

```Julia
using Pkg

pkg"add https://github.com/taraspiotr/DeepJulia"
pkg"precompile"

using DeepJulia
```

## Example

Example scripts can be found in the `examples/` directory.

## Remarks

This package was created as a part of the assignment for the COMP0090 - Introduction to Deep Learning @ UCL.

This is my first project in Julia and it is more than certain that there are a lot of design flaws. Nevertheless I hope this package can have some value for those picking up deep learning / Julia.

I will be very grateful for every issue / comment / PR.

## What's implemented

- [x] Tensor
- [x] Autograd: +, *, /, .+ (for same shape 2-dim arrays), logistic
- [x] NN: NNModule, LinearLayer, Activation, SigmoidActivation, ModuleList
- [x] Optim: SGD
- [x] Loss: LogLoss (without the log trick), MSE
- [ ] Autograd: everything else, especially broadcasting
- [ ] CUDA support
- [ ] NN: Convolutions, RNN, Transformer, ...
- [ ] Optim: Adam, ...
- [ ] ...
