@enum Device begin
    cpu
end

function to(A::AbstractArray, device::Device)
    if device == cpu
        return Array(A)
    else
        throw(ArgumentError("Unrecognized device"))
    end
end
