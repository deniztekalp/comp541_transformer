
using Pkg; haskey(Pkg.installed(),"Knet") || Pkg.add("Knet")
using Knet

file1 = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en")
file2 = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de")

train1_en = readlines(file1)
train1_de = readlines(file2)

file3 = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en")
file4 = download("https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de")

test1_en = readlines(file3)
test1_de = readlines(file4)

function minibatch(sentences, batchsize, results)
    table = Dict{Int,Vector{Tuple{String, String}}}()
    data = Any[] 
       
    index = 1
    for sentence in sentences
        n = length(sentence)
        nsentences = get!(table, n, Any[])
        push!(nsentences, (sentence, results[index]))
        if length(nsentences) == batchsize
            push!(data, nsentences)
            empty!(nsentences)
        end
        index += 1
    end
    for pair in table
        push!(data, pair[2])
    end
    return data
end

batchsize = 128
train_data_en = minibatch(train1_en, batchsize, train1_de)
test_data_en = minibatch(test1_en, batchsize, test1_de)
summary(train_data_en)
summary(test_data_en)

function BLEU1gram(test, ref) 
    score1 = 0
    output = split(test, ' ')
    reference = split(ref, ' ')
    counts = Dict{String, Int}()
    for i = 1:length(output)
        if !haskey(counts, output[i])
            counts[output[i]] = 0
        end
        for j = 1:length(reference)
            if output[i] == reference[j]
                counts[output[i]] += 1
                break
            end
        end
    end
    for i = 1:length(reference)
        if !haskey(counts, reference[i])
            counts[reference[i]] = 0
        end
        counts[reference[i]] = min(count(x->x==reference[i], reference), counts[reference[i]] )
    end
    sum = 0
    for c in counts
        sum += c[2]
    end
    return score1 = sum / length(output)
end


function BLEU2gram(test, ref)
    score2 = 0
    output = split(test, ' ')
    reference = split(ref, ' ')
    counts = Dict{String, Int}()
    reference2 = []
    for index = 1:length(reference)-1
        push!(reference2, string(reference[index], " ", reference[index+1]))
    end
    for i = 1:length(output)-1
        str = string(output[i], " ", output[i+1])
        if !haskey(counts, str)
            counts[str] = 0
        end
        for j = 1:length(reference)-1
            if str == string(reference[j], " ", reference[j+1])
                counts[str] += 1
                break
            end
        end
    end
    for i = 1:length(reference)-1
        tmp = string(reference[i], " ", reference[i+1])
        if !haskey(counts, tmp)
            counts[tmp] = 0
        end
        counts[tmp] = min(count(x->x==tmp, reference2), counts[tmp] )
    end
    sum = 0
    for c in counts
        sum += c[2]
    end
    return score2 = sum / (length(output)-1)
end

BLEU1gram("the the the the the the the", "the cat is on the mat")

BLEU2gram("the cat the cat on the mat", "there is a cat on the mat")

function BLEU(test, ref) #need to add brevity, trigram and 4-gram
    return sqrt(BLEU1gram(test, ref)) * sqrt(BLEU2gram(test, ref)) * 100
end

BLEU("the cat the cat on the mat", "there is a cat on the mat")

function randomOutput(minibatch)
    sum = 0
    for sentencePair in minibatch
        sum += BLEU("der das die den dem", sentencePair[2])
    end
    return sum / length(minibatch)
end

randomOutput(train_data_en[20])



