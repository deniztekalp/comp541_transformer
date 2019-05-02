struct Embedding
    w #weights
end

function Embedding(embedsize::Int, vocabsize::Int)
    w = param(embedsize, vocabsize);
    return Embedding(w)
end

#x is the token number
function (l::Embedding)(x)
    l.w[:, x]
end
