using Statistics

logistic(x) = 1 ./ (1 .+ exp.(-x))

function logistic(t::Tensor)
    values = logistic(t.values)
    if t.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(t, x -> x .* logistic(t.values) .* (1 .- logistic(t.values)))
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=t.device
    )
end

function logloss(y::Tensor, ŷ::Tensor)
    values = [mean(-y.values .* log.(ŷ.values) - (1 .- y.values) .* log.(1 .- ŷ.values))]
    if y.requires_grad || ŷ.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(y, x -> x .* (log.(1 .- ŷ.values) - log.(ŷ.values)) / length(y)),
            Dependency(ŷ, x -> x .* (ŷ.values - y.values) ./ (ŷ.values .* (1 .- ŷ.values) * length(y)))
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=y.device
    )
end


function mse(y::Tensor, ŷ::Tensor)
    values = [mean((y.values - ŷ.values).^2 / 2)]
    if y.requires_grad || ŷ.requires_grad
        requires_grad = true
        dependencies = [
            Dependency(y, x -> x .* (y.values - ŷ.values) / length(y)),
            Dependency(ŷ, x -> x .* (ŷ.values - y.values) / length(y))
        ]
    else
        requires_grad = false
        dependencies = nothing
    end
    Tensor(
        values;
        dependencies=dependencies,
        device=y.device
    )
end
