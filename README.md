# DeepJulia
A very simple DL package in Julia. The API is heavily inspired by the one and only [PyTorch](pytorch.org) library.

This package was created as a part of the assignment for the COMP0090 - Introduction to Deep Learning @ UCL.

This is my first project in Julia and it is more than certain that there are a lot of design flaws. Nevertheless I hope this package can have some value for those picking up deep learning / Julia.

I will be very grateful for every issue / comment / PR.


## Installation

```Julia
using Pkg

pkg"add https://github.com/taraspiotr/DeepJulia"
pkg"precompile"

using DeepJulia
```

## Example

```
using DeepJulia

lr = 1e-2
momentum = 0.9
num_epochs = 10
batch_size = 8
D = 100

model = ModuleList([
    LinearLayer(D, D ÷ 2),
    SigmoidActivation(),
    LinearLayer(D ÷ 2, 1),
    SigmoidActivation(),
])

loss = MSE()
optim = SGD(params(model), lr, momentum)

input = Tensor(rand(batch_size, D); requires_grad=false)
output = Tensor(rand(batch_size, 1); requires_grad=false)

for i=1:num_epochs
    zerograd!(optim)
    l = get_loss(loss, output, forward(model, input))
    backward!(l)
    DeepJulia.step(optim)
    println("Epoch $(i), loss = $(l.values[1])")
end
```

Example scripts can be found in the `examples/` directory.

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
