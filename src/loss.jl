abstract type Loss end

get_loss(l::Loss, y, ŷ) = throw("unimplemented")

struct LogLoss <: Loss
end

get_loss(l::LogLoss, y, ŷ) = logloss(y, ŷ)
