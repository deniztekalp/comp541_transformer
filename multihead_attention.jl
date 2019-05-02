using Knet
using LinearAlgebra

struct multihead_attention
    no_of_heads
    projection_dimension
    key_weights
    query_weights
    value_weights
    W
end
#todo: use knet arrays

#no_of_heads * projection_dimension = embedsize
function multihead_attention(embedsize::Int)

    projection_dimension = convert(Int, round(embedsize^(2/3)))
    no_of_heads = convert(Int, sqrt(projection_dimension))
    key_weights = param(projection_dimension, embedsize, no_of_heads);
    query_weights = param(projection_dimension, embedsize, no_of_heads);
    value_weights = param(projection_dimension, embedsize, no_of_heads);
    W = param(embedsize, no_of_heads * projection_dimension)
    return multihead_attention(no_of_heads, projection_dimension, key_weights, query_weights, value_weights, W)
end

function (m::multihead_attention)(key, query, value, mask)
    key_projections = map(x->view(m.key_weights, :,:,x)*key, 1:m.no_of_heads)
    query_projections = map(x->view(m.query_weights, :,:,x)*query, 1:m.no_of_heads)
    value_projections = map(x->view(m.value_weights, :,:,x)*value, 1:m.no_of_heads)
    results = map((q,k,v) -> attention(q,k,v,mask), key_projections, query_projections, value_projections)
    results = vcat(results...)
    m.W * results
end

function (m::multihead_attention)(x)
    m(x, x, x, false)
end

function (m::multihead_attention)(tmp, x, mask)
    m(x, x, x, mask)
end

function (m::multihead_attention)(top_encoder_output, x)
    m(top_encoder_output, x, top_encoder_output, false)
end

#computes scaled dot product attention
function attention(query, key, value, mask)
    key_dimension = size(query, 1)
    scores = (key' * query) ./ sqrt(key_dimension)
    if(mask)
        mask = triu(fill(-Inf, size(scores)), 1)
        scores = scores .+ mask
    end
    scores = value * softmax(scores, dims = 1)
    #if pdropout > 0, do dropout
    return scores
    #sums = reduce(+, scoreMatrix, dims = 2)
end
