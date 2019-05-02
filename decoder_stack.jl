include("decoder.jl")

struct decoder_stack
    decoders
    decoder_stack(decoders...) = new(decoders)
end

function decoder_stack(embedsize::Int, num_inner::Int)
    decoder_stack(decoder(embedsize, num_inner), decoder(embedsize, num_inner), decoder(embedsize, num_inner), decoder(embedsize, num_inner), decoder(embedsize, num_inner), decoder(embedsize, num_inner))
end

function(d::decoder_stack)(top_encoder_output, x)
    (for l in d.decoders; x = l(top_encoder_output, x); end; x) #true for masking
end
