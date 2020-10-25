mutable struct Variable
    values::Tensor
    grad::Tensor
    
    Variable(values) = new(values, to(zeros(size(values)), device(values)))
    Variable(values, grad) = new(values, grad)
end

function zerograd!(v::Variable)
    fill!(v.grad, 0)
end

function to!(v::Variable, device::Device)
    v.values = to(v.values, device)
    v.grad = to(v.grad, device)
end
