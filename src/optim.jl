import Base: step


abstract type Optimizer end

"""
    step(optim)

Run a single optimization step.
"""
step(optim::Optimizer) = throw("unimplemented")

to!(optim::Optimizer, device::Device) = throw("unimplemented")

"""
    zerograd!(optim)

Run zerograd! on every parameters optimized by optim.
"""
function zerograd!(optim::Optimizer)
    for p ∈ optim.params
        zerograd!(p)
    end
end

struct SGD <: Optimizer
    params::Vector{Tensor}
    lr::Real
    momentum::Real
    velocities::Vector{AbstractArray}
    
    SGD(params, lr) = new(params, lr, 0.0, Vector{Matrix{Real}}(undef, size(params, 1)))
    SGD(params, lr, momentum) = new(params, lr, momentum, [to(zeros(size(p.values)), p.device) for p ∈ params])
end

function step(optim::SGD)
    for (param, prev_vel) ∈ zip(optim.params, optim.velocities)
        vel =  -param.grad
        if optim.momentum > 0
            vel .+= optim.momentum * prev_vel
            prev_vel .= deepcopy(vel)
        end
        param.values += optim.lr * vel
    end
end

function to!(optim::SGD, device::Device)
    for (i, vel) ∈ enumerate(optim.velocities)
        optim.velocities[i] = to(vel, device)
    end
end
