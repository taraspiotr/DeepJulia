import Base: size, getindex, sum, +, *, -, /, adjoint, show, broadcasted, length


struct Dependency
    tensor::Any
    grad_fn::Function
end

mutable struct Tensor{T,N}
    values::AbstractArray{T,N}
    grad::Union{AbstractArray,Nothing}
    dependencies::Vector{Dependency}
    requires_grad::Bool
    device::Device
    
    function Tensor(values::AbstractArray{T,N}; grad=nothing, dependencies=nothing, requires_grad=true, device=cpu) where T where N
        values = to(values, device)
        grad = requires_grad ? to(isnothing(grad) ? zeros(size(values)) : grad, device) : nothing
        dependencies = isnothing(dependencies) ? Vector{Dependency}() : dependencies
        
        new{T,N}(values, grad, dependencies, requires_grad, device)
    end
end

size(t::Tensor, args...; kwargs...) = size(t.values, args...; kwargs...)
getindex(t::Tensor, args...; kwargs...) = getindex(t.values, args...; kwargs...)
length(t::Tensor) = prod(size(t))
adjoint(t::Tensor) = Tensor(t.values', grad=t.grad', dependencies=[Dependency(t, x -> x')], device=t.device)

function show(io::IO, t::Tensor)
    println("$(typeof(t)), requires_grad = $(t.requires_grad)")
    show(io, t.values)
end

function show(io::IO, m::MIME"text/plain", t::Tensor)
    println("$(typeof(t)), requires_grad = $(t.requires_grad)")
    show(io, m, t.values)
end

function to!(t::Tensor, device::Device)
    t.values = to(t.values, device)
    if t.requires_grad
        t.grad = to(t.values, device)
    end
    t.device = device
end

function zerograd!(t::Tensor)
    fill!(t.grad, 0)
end

function backward!(t::Tensor, grad::Union{AbstractArray,Nothing}=nothing)
    @assert t.requires_grad "Called backward! on a Tensor that doesn't require grad"
    grad = isnothing(grad) ? to(ones(size(t.grad)), t.device) : grad
    t.grad .+= grad
    for dep ∈ t.dependencies
        if dep.tensor.requires_grad
            backward!(dep.tensor, dep.grad_fn(t.grad))
        end
    end
end

sum(t::Tensor) = Tensor(
        [sum(t.values)];
        requires_grad=t.requires_grad,
        dependencies=t.requires_grad ? [Dependency(t, x -> x .* to(ones(size(t.grad)), t.device))] : nothing,
        device=t.device
)

function +(t1::Tensor, t2::Tensor)
    values = t1.values + t2.values
    if t1.requires_grad || t2.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(t1, x -> x),
            Dependency(t2, x -> x)
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=t1.device
    )
end

-(t::Tensor) = Tensor(-t.values; dependencies=[Dependency(t, x -> -x)], device=t.device)
-(t1::Tensor, t2::Tensor) = t1 + (-t2)

function *(r::Number, t::Tensor)
    Tensor(
        t.values * r;
        requires_grad=t.requires_grad,
        dependencies=t.requires_grad ? [Dependency(t, x -> r * x)] : nothing,
        device=t.device
    )
end

*(t::Tensor, r::Number) = r * t
/(t::Tensor, r::Number) = t * (1 / r)

function *(t1::Tensor{T,2}, t2::Tensor{T,2}) where T
    values = t1.values * t2.values
    if t1.requires_grad || t2.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(t1, x -> x * t2.values'),
            Dependency(t2, x -> t1.values' * x)
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=t1.device
    )
end

function broadcasted(+, t1::Tensor{T,2}, t2::Tensor{T,1}) where T
    values = t1.values .+ t2.values
    if t1.requires_grad || t2.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(t1, x -> x),
            Dependency(t2, x -> to(dropdims(sum(x, dims=2), dims=2), t2.device))
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=t1.device
    )
end

broadcasted(+, t1::Tensor{T,1}, t2::Tensor{T,2}) where T = t2 .+ t1

function broadcasted(+, t1::Tensor{T,2}, t2::Tensor{T,2}) where T
    @assert size(t2, 1) == 1 "At the moment only t2 of size (1, N)"
    values = t1.values .+ t2.values
    if t1.requires_grad || t2.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(t1, x -> x),
            Dependency(t2, x -> to(sum(x, dims=1), t2.device))
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=t1.device
    )
end
