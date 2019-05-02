using Knet

struct positionwise_feedforward
    w1
    b1
    w2
    b2
end

function positionwise_feedforward(num_inputs::Int, num_inner::Int, num_outputs::Int)
    w1 = param(num_inner, num_inputs);
    b1 = param0(num_inner, 1);
    w2 = param(num_outputs, num_inner);
    b2 = param0(num_outputs, 1);
    return positionwise_feedforward(w1,b1,w2,b2)
end

function (p::positionwise_feedforward)(x)
    p.w2 * relu.(p.w1 * x .+ p.b1) .+ p.b2
end
