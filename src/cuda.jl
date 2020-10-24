using CUDA: CuArray

@enum Device begin
    cpu
    gpu
end

ArrayOrCuArray = Union{Array,CuArray}

function to(A::ArrayOrCuArray, device::Device)
    if device == cpu
        return Array(A)
    elseif device == gpu
        return CuArray(A)
    else
        throw(ArgumentError("Unrecognized device"))
    end
end

device(A::ArrayOrCuArray) = isa(A, CuArray) ? gpu : cpu
