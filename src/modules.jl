using Random

abstract type NNModule end

forward(m::NNModule, x) = throw("unimplemented")
params(m::NNModule) = throw("unimplemented")

function to!(m::NNModule, device::Device)
    for p ∈ params(m)
        to!(p, device)
    end
end

struct LinearLayer <: NNModule
    W::Tensor
    b::Tensor
    
    function LinearLayer(input_dim::Integer, output_dim::Integer)
        a = sqrt(1 / input_dim)
        new(
            Tensor(rand(input_dim, output_dim) * 2a .- a),
            Tensor(zeros(1, output_dim) * 2a .- a),
        )
    end
end

forward(m::LinearLayer, x) = x * m.W .+ m.b
params(m::LinearLayer) = [m.W, m.b]

struct Activation <: NNModule
    activation::Function
end


forward(m::Activation, x) = m.activation(x)
params(m::Activation) = []

SigmoidActivation() = Activation(logistic)

struct ModuleList <: NNModule
    modules::Array{NNModule}
end

function forward(m::ModuleList, x)
    for mod ∈ m.modules
        x = forward(mod, x)
    end
    x
end

params(m::ModuleList) = vcat([params(mod) for mod ∈ m.modules]...)
