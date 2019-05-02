include("layernorm.jl")
include("multihead_attention.jl")
include("positionwise_feedforward.jl")

struct decoder
    self_attention
    encoder_decoder_attention
    layernorm
    feedforward
end


function decoder(embedsize::Int, num_inner::Int; pdrop = 0.1)
    self_attention = multihead_attention(embedsize)
    encoder_attention = multihead_attention(embedsize)
    layernorm = LayerNorm(0.000000001)
    feedforward = positionwise_feedforward(embedsize, num_inner, embedsize)
    return decoder(self_attention, encoder_attention, layernorm, feedforward)
end


function (d::decoder)(top_encoder_output, x)
    tmp1 = d.layernorm(x + d.self_attention(top_encoder_output, x, true)) #top_encoder_output here is redundant and not used
    tmp2 = d.layernorm(tmp1 + d.encoder_decoder_attention(top_encoder_output, tmp1))
    d.layernorm(tmp2 +d.feedforward(tmp2))
end
