using Printf
using Flux,Flux.Data.MNIST,Statistics
using Flux:onehotbatch,onecold,crossentropy,@epochs
using Base.Iterators:partition
using BSON:@load,@save
using CuArrays
using Tracker
using Random

function prepare_dataset(;train=true)
    train_or_test = ifelse(train,:train,:test)
    imgs = MNIST.images(train_or_test)
    X = hcat(float.(vec.(imgs))...)
    labels = MNIST.labels(train_or_test)
    Y = onehotbatch(labels,0:9)
    return X,Y
end
function define_model(;hidden)
    mlp = Chain(Dense(28^2,hidden,relu),
                Dense(hidden,hidden,relu),
                Dense(hidden,10),
                softmax)
    return mlp
end
function split_dataset_random(X,Y)
    divide_ratio = 0.9
    shuffled_indices = shuffle(1:size(Y)[2])
    divide_idx = round(Int,0.9*length(shuffled_indices))
    train_indices = shuffled_indices[1:divide_idx]
    val_indices = shuffled_indices[divide_idx:end]
    train_X = X[:,train_indices]
    train_Y = Y[:,train_indices]
    val_X = X[:,val_indices]
    val_Y = Y[:,val_indices]
    return train_X,train_Y,val_X,val_Y
end
function train()
    println("Start to train")
    epochs = 10
    X, Y = prepare_dataset(train=true)
    train_X, train_Y, val_X,val_Y = split_dataset_random(X, Y)
    model = define_model(hidden=100) |> gpu #如果没有gpu，只有cpu，此处可以写成 |> cpu。本代码中是一样的。
    loss(x,y)= crossentropy(model(x),y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    batchsize = 64
    train_dataset = gpu.([(train_X[:,batch] ,Float32.(train_Y[:,batch])) for batch in partition(1:size(train_Y)[2],batchsize)])
    val_dataset = gpu.([(val_X[:,batch] ,Float32.(val_Y[:,batch])) for batch in partition(1:size(val_Y)[2],batchsize)])

    callback_count = 0
    #CUDA.allowscalar(false)
    eval_callback = function callback()
        callback_count += 1
        if callback_count == length(train_dataset)
            println("action for each epoch")
            total_loss = 0
            total_acc = 0
            for (vx, vy) in val_dataset
                pred_y = model(vx)
                total_loss += loss(vx, vy)
                total_acc += accuracy(vx,vy)
            end
            total_loss /= length(train_dataset)
            total_acc /= length(train_dataset)
            @printf("total_acc %0.3f,total_loss %0.3f\n",total_acc,total_loss)#跟python中的装饰器类似，代码中他处类似
            callback_count = 0
            pretrained = model |> cpu
            @save "pretrained.bson" pretrained
            callback_count = 0
        end
    end
    optimizer = ADAM()
    @epochs epochs Flux.train!(loss,params(model),train_dataset, optimizer, cb = eval_callback)
    pretrained = model |> cpu
    weights = Tracker.data.(params(pretrained))
    @save "pretrained.bson" pretrained
    @save "weights.bson" pretrained
    println("Finished to train")
end

function predict()
    println("Start to evaluate testset")
    println("loading pretrained model")
    @load "pretrained.bson" pretrained
    model = pretrained |> gpu
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    println("prepare dataset")
    X, Y = prepare_dataset(train=false)
    X = X |> gpu
    Y = Y |> gpu
    acc = accuracy(X, Y)
    @printf("Test dataset accuracy is %0.3f.\n"acc)
    println("Done")
end
function predict2()
    println('Start to evaluate testset')
    println('Loading pretrained model')
    @load 'weights.bson' weights
    model = define_model(hidden=100)
    Flux.loadparams!(model,weights)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    println('Prepare dataset')
    X,Y = prepare_dataset(train=false)
    acc = accuracy(X, Y)
    @printf("Test dataset accuracy is %0.3f.\n"acc)
    println('Done!')
end
function main()
    train()
    predict2()
    predict()
end
main()
