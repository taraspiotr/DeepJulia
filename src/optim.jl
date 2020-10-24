abstract type Optimizer end

step(optim::Optimizer) = throw("unimplemented")
to!(optim::Optimizer, device::Device) = throw("unimplemented")

struct SGD <: Optimizer
    params::Vector{Variable}
    lr::Real
    momentum::Real
    velocities::Vector{ArrayOrCuArray}
    
    SGD(params, lr) = new(params, lr, 0.0, Vector{Matrix{Real}}(undef, size(params, 1)))
    SGD(params, lr, momentum) = new(params, lr, momentum, [to(zeros(size(p.values)), device(p.values)) for p ∈ params])
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
