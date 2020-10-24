module DeepJulia

export Device,
gpu,
cpu,
ArrayOrCuArray,
to,

# variable

Variable,
zerograd!,
to!,

# loss
Loss,
LogLoss,
MSE,
get_loss,
get_grad,

# modules

NNModule,
LinearLayer,
forward!,
backward!,
params,
Activation,
SigmoidActivation,
ModuleList,

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

include("cuda.jl")
include("variable.jl")
include("loss.jl")
include("modules.jl")
include("optim.jl")
include("dataset.jl")

end # module
