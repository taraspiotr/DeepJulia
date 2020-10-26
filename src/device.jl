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
