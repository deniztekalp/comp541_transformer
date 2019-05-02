include("encoder.jl")

struct encoder_stack
    encoders
    encoder_stack(encoders...) = new(encoders)
end

function encoder_stack(embedsize::Int, num_inner::Int)
    encoder_stack(encoder(embedsize, num_inner), encoder(embedsize, num_inner), encoder(embedsize, num_inner), encoder(embedsize, num_inner), encoder(embedsize, num_inner), encoder(embedsize, num_inner))
end

function(e::encoder_stack)(x)
    (for l in e.encoders; x = l(x); end; x)
end
