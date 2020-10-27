import Random
using Statistics

using DeepJulia


Random.seed!(123)

run(`curl -fsS http://udon.stacken.kth.se/\~ninjin/comp0090_assignment_1_data.tar.gz -o /tmp/data.tar.gz`)
run(`tar -x -z -f /tmp/data.tar.gz -C /tmp/`)
run(`rm -f /tmp/data.tar.gz`);

trainset = FashionMNIST("/tmp/comp0090_assignment_1_data", "train")
validset = FashionMNIST("/tmp/comp0090_assignment_1_data", "dev")

lr = 1e-2
momentum = 0.9
num_epochs = 10
batch_size = 8
DEVICE = cpu

D = size(trainset[1][1], 1)

model = ModuleList([
        LinearLayer(D, D ÷ 2),
        SigmoidActivation(),
        # LinearLayer(D ÷ 2, D ÷ 2),
        # SigmoidActivation(),
        LinearLayer(D ÷ 2, 1),
        SigmoidActivation(),
        ])

for m ∈ model.modules
    if m isa LinearLayer
        xavier_uniform!(m.W, 4)
        zero!(m.b)
    end
end

loss = LogLoss()
optim = SGD(params(model), lr, momentum)

to!(model, DEVICE)
to!(optim, DEVICE)

for epoch = 1:num_epochs
    shuffle!(trainset)
    for stage ∈ ["train", "valid"]
        total_loss = 0
        accuracy = 0
        dataset = stage == "train" ? trainset : validset
        batches = batchify(dataset, batch_size)
        for (i, (x, y)) ∈ enumerate(batches)
            x, y = Tensor(x; device=DEVICE), Tensor(reshape(y, (:, 1)); device=DEVICE)
            
            if stage == "train"
                zerograd!(optim)
            end
            
            ŷ = forward(model, x)
            l = get_loss(loss, y, ŷ)
            total_loss += l.values[1]

            if stage == "valid"
                accuracy += mean((y.values .> 0.5) .== (ŷ.values .> 0.5))
            end
            
            if stage == "train"
                backward!(l)
                DeepJulia.step(optim)
            end
        end
        println("Epoch: $epoch\t$stage loss: $(total_loss / size(batches, 1))")
        if stage == "valid"
            println("Epoch: $epoch\t$stage accuracy: $(accuracy / size(batches, 1))")
        end

    end
    println()
end
