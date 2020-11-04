module DeepJulia

export Device,
gpu,
cpu,
to,

# tensor
# here we implement the autograd
# and implement basic operations on Tensors

Tensor,
size,
getindex,
length,
adjoint,
show,
to!,
zerograd!,
backward!,
sum,
+,
-,
*,
/,
broadcasted,

# functional
# here we keep any useful functions
# with implemented gradients

logistic,
logloss,

# loss
# wrapper on losses

Loss,
LogLoss,
MSE,
get_loss,

# modules
# modules for building Neural Networks

NNModule,
LinearLayer,
Activation,
SigmoidActivation,
ModuleList,
forward,
params,

# optim
# optimizers for NNModules

Optimizer,
SGD,
step,

# dataset

Dataset,
length,
batchify,
FashionMNIST,
shuffle!,

# init
# initialization functions

xavier_uniform!,
zero!

include("device.jl")
include("tensor.jl")
include("functional.jl")
include("loss.jl")
include("modules.jl")
include("optim.jl")
include("dataset.jl")
include("init.jl")

end # module
