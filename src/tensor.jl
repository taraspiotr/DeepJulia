using CUDA: CuArray

Tensor = Union{Array,CuArray}

function to(A::Tensor, device::Device)
    if device == cpu
        return Array(A)
    elseif device == gpu
        return CuArray(A)
    else
        throw(ArgumentError("Unrecognized device"))
    end
end

device(A::Tensor) = isa(A, CuArray) ? gpu : cpu
