using NPZ

abstract type Dataset end

length(datasett::Dataset) = throw("unimplemented")

batchify(x, n) = [x[i:min(i + n - 1, length(x))] for i in 1:n:length(x)]

mutable struct FashionMNIST <: Dataset
    xs::AbstractMatrix{<:Real}
    ys::AbstractVector{<:Integer}
end
    
length(dataset::FashionMNIST) = size(dataset.xs, 1)

function FashionMNIST(dirpath::String, set::String)
    xs = npzread("$(dirpath)/fashion-$(set)-imgs.npz")
    ys = npzread("$(dirpath)/fashion-$(set)-labels.npz")
    FashionMNIST(reshape(xs, :, size(xs, 3))', ys)
end

Base.getindex(dataset::FashionMNIST, i::Integer) = dataset.xs[i, :], dataset.ys[i]
Base.getindex(dataset::FashionMNIST, r::UnitRange{Int64}) = dataset.xs[r, :], dataset.ys[r]

function shuffle!(dataset::FashionMNIST)
    ord = randperm(length(dataset))
    dataset.xs = dataset.xs[ord, :]
    dataset.ys = dataset.ys[ord]
    return dataset
end
