using Knet
using Statistics


include("multihead_attention.jl")
include("positionwise_feedforward.jl")
include("layernorm.jl")


struct encoder
    self_attention
    layernorm
    feedforward
end


function encoder(embedsize::Int, num_inner::Int)
    self_attention = multihead_attention(embedsize)
    layernorm = LayerNorm(0.000000001) #look this figure up
    feedforward = positionwise_feedforward(embedsize, num_inner, embedsize)
    return encoder(self_attention, layernorm, feedforward)
end


function (e::encoder)(x)
    tmp = e.layernorm(x + e.self_attention(x))
    e.layernorm(tmp + e.feedforward(tmp))
end
