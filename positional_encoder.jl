
struct Positional_Encoder
    w # weight
end

#embedsize needs to be even
function positional_encoder(embedsize::Int, max_sequence_length::Int)
    w = zeros(embedsize, max_sequence_length)
    for pos in 1:max_sequence_length
        for i in 1:div(embedsize, 2)
            w[2*i-1, pos] = sin(pos/(10000^(2*i-1/embedsize)))
            w[2*i, pos] = cos(pos/(10000^(2*i-1/embedsize)))
        end
    end
    return Positional_Encoder(w)
end

#need to add dropout
function(p::Positional_Encoder)(x)
        x_seq_len = size(x, 2)
        p.w[:,1:x_seq_len] + x
end
