
import Base: size, getindex, sum, +, *, adjoint, show

using CUDA

@enum Device begin
    cpu
    gpu
end

function to(A::AbstractArray, device::Device)
    if device == cpu
        return Array(A)
    elseif device == gpu
        return CuArray(A)
    else
        throw(ArgumentError("Unrecognized device"))
    end
end

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
adjoint(t::Tensor) = Tensor(t.values', grad=t.grad', dependencies=[Dependency(t, x -> x')], device=t.device)

function show(io::IO, t::Tensor)
    println("$(typeof(t)), requires_grad = $(t.requires_grad)")
    show(io, t.values)
end

function show(io::IO, m::MIME"text/plain", t::Tensor)
    println("$(typeof(t)), requires_grad = $(t.requires_grad)")
    show(io, m, t.values)
end

function backward!(t::Tensor, grad::AbstractArray)
    if t.requires_grad
        t.grad .+= grad
        for dep ∈ t.dependencies
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

function *(r::Number, t::Tensor)
    Tensor(
        t.values * r;
        requires_grad=t.requires_grad,
        dependencies=t.requires_grad ? [Dependency(t, x -> r * x)] : nothing,
        device=t.device
    )
end

*(t::Tensor, r::Number) = r * t

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




if abspath(PROGRAM_FILE) == @__FILE__
    a = Tensor(rand(3, 2))
    b = Tensor(rand(2, 3))
    c = Tensor(rand(3, 3))
    d = sum(3 * a' * c + b)

    backward!(d, [1])

    println(a.grad)
    println((3 * ones(size(a.values')) * c.values')')
end
