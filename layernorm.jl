using Knet
using Statistics

struct LayerNorm
    epsilon
end

function (l::LayerNorm)(x)
    means = mean(x, dims=1)
    variances = mean((x .- means).^2, dims=1)
    x = (x .- means)./ ((variances.+l.epsilon).^0.5)
end
