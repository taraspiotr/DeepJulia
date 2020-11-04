abstract type Loss end

"""
    get_loss(l, y, ŷ)

Return a loss.
"""
get_loss(l::Loss, y, ŷ) = throw("unimplemented")

struct LogLoss <: Loss
end

get_loss(l::LogLoss, y, ŷ) = logloss(y, ŷ)


struct MSE <: Loss
end

get_loss(l::MSE, y, ŷ) = mse(y, ŷ)
