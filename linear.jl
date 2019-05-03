using Knet

struct Linear
    w
    b
end

function Linear(num_inputs::Int, num_outputs::Int)
    w = param(num_outputs, num_inputs, init=xavier)
    b = param0(num_outputs, 1);
    return Linear(w,b)
end

function (l::Linear)(x)
    softmax(l.w * x .+ l.b, dims=1)
end
