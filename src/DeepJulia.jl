module DeepJulia

export Device,
gpu,
cpu,
to,

# tensor

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

logistic,
logloss,

# loss
Loss,
LogLoss,
MSE
get_loss,

# modules

NNModule,
LinearLayer,
Activation,
SigmoidActivation,
ModuleList,
forward,
params,

# optim

Optimizer,
SGD,
step,

# dataset

Dataset,
length,
batchify,
FashionMNIST,
shuffle!

include("device.jl")
include("tensor.jl")
include("functional.jl")
include("loss.jl")
include("modules.jl")
include("optim.jl")
include("dataset.jl")

end # module
