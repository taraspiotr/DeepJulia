using Statistics

abstract type Loss end

get_loss(l::Loss, y, ŷ) = throw("unimplemented")
get_grad(l::Loss, y, ŷ) = throw("unimplemented")

struct LogLoss <: Loss
end

get_loss(l::LogLoss, y, ŷ) = mean(-y .* log.(ŷ) - (1 .- y) .* log.(1 .- ŷ))
get_grad(l::LogLoss, y, ŷ) = (ŷ - y) ./ (ŷ .* (1 .- ŷ))

struct MSE <: Loss
end

get_loss(l::MSE, y, ŷ) = mean((y - ŷ).^2 / 2)
get_grad(l::MSE, y, ŷ) = (ŷ - y) / length(y)
