using Knet

include("decoder_stack.jl")
include("embedding.jl")
include("positional_encoder.jl")
include("encoder_stack.jl")
include("linear.jl")

file_vocab_en = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en")
file_vocab_de = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de")

vocab_de = readlines(file_vocab_de)
vocab_en = readlines(file_vocab_en)

file3 = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en")
file4 = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de")

test1_en = readlines(file3)
test1_de = readlines(file4)


function onehot(size::Int, index::Int)
    tmp = zeros(size, 1)
    tmp[index, 1] = 1
    return tmp
end


struct Transformer
    input_embedding
    output_embedding
    positional_encoding
    encoders
    decoders
    linear
end


function transformer(vocabsize::Int, embedsize::Int, max_sequence_length::Int, )
    input_embedding = Embedding(embedsize,vocabsize)
    output_embedding = Embedding(embedsize,vocabsize)
    positional_encoding = positional_encoder(embedsize, max_sequence_length)
    encoders = encoder_stack(512,2048)
    decoders = decoder_stack(512,2048)
    generator = Linear(vocabsize, embedsize)
    Transformer(input_embedding, output_embedding, positional_encoding, encoders, decoders, generator)
end

function(t::Transformer)(x)
    encoder_output = t.encoders(t.positional_encoding(t.input_embedding(x)))

    start_token = 2
    end_token = 3
    count = 1

    target_embedding = t.output_embedding(start_token)
    target_token = argmax(t.linear(t.decoders(encoder_output,target_embedding)))

    output_string = String(vocab_de[target_token])
    current_output = [target_token]

    while target_token_index != 3 #end_token_index
        output_embedding = t.target_embedding(current_output)
        length = size(current_output)[1]
        target_token = argmax(t.linear(t.decoders(encoder_output,output_embedding)[:,length]))
        push!(current_output, target_token)
        output_string = String(output_string, " ", vocab_de[target_token])
        count = count + 1
        if count > 10
            break
        end
    return output_string
    end
end
