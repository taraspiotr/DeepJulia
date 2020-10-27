function xavier_uniform!(t::Tensor, gain::Real=0)
    k = sqrt(6 / sum(size(t)))
    t.values = (rand(size(t)...) * 2k .- k) * gain
end

function zero!(t::Tensor)
    fill!(t.values, 0)
end
