using Random

abstract type NNModule end

forward!(m::NNModule, x) = throw("unimplemented")
backward!(m::NNModule, grad) = throw("unimplemented")
params(m::NNModule) = throw("unimplemented")
zerograd!(m::NNModule) = throw("unimplemented")
to!(m::NNModule, device::Device) = throw("unimplemented")

mutable struct LinearLayer <: NNModule
    W::Variable
    b::Variable
    
    input::ArrayOrCuArray
    
    LinearLayer(W::ArrayOrCuArray, b::ArrayOrCuArray) = new(
        Variable(W),
        Variable(b),
        Matrix{Real}(undef, 0, 0),
    )
end
    
LinearLayer(input_dim::Integer, output_dim::Integer) = LinearLayer(
    randn(input_dim, output_dim) ./ 100,
    zeros(output_dim),
)

function forward!(m::LinearLayer, X)
    m.input = X
    X * m.W.values .+ m.b.values'
end

function backward!(m::LinearLayer, grad)
    m.W.grad += m.input' * grad
    m.b.grad += dropdims(sum(grad, dims=1), dims=1)
    
    grad * m.W.values'
end

params(m::LinearLayer) = [m.W, m.b]

function zerograd!(m::LinearLayer)
    for param ∈ params(m)
        zerograd!(param)
    end
end

function to!(m::LinearLayer, device::Device)
    for param ∈ params(m)
        to!(param, device)
    end
end

mutable struct Activation <: NNModule
    activation::Function
    grad::Function
    
    input::AbstractMatrix{<:Real}
    
    Activation(activation::Function, grad::Function) = new(
        activation,
        grad,
        Matrix{Real}(undef, 0, 0),
    )
end


function forward!(m::Activation, X)
    m.input = X
    m.activation(X)
end

function backward!(m::Activation, grad)
    grad .* m.grad(m.input)
end

params(m::Activation) = []
zerograd!(m::Activation) = nothing
to!(m::Activation, device::Device) = m

logistic(x) = 1 ./ (1 .+ exp.(-x))
logistic_grad(x) = logistic(x) .* (1 .- logistic(x))
SigmoidActivation() = Activation(logistic, logistic_grad)

struct ModuleList <: NNModule
    modules::Array{NNModule}
end

function forward!(m::ModuleList, X)
    for mod ∈ m.modules
        X = forward!(mod, X)
    end
    X
end

function backward!(m::ModuleList, grad)
    for mod ∈ m.modules[end:-1:1]
        grad = backward!(mod, grad)
    end
    grad
end

params(m::ModuleList) = vcat([params(mod) for mod ∈ m.modules]...)
        
function zerograd!(m::ModuleList)
    for mod ∈ m.modules
        zerograd!(mod)
    end
end

function to!(m::ModuleList, device::Device)
    for mod ∈ m.modules
        to!(mod, device)
    end
end
