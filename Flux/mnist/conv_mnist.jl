using Flux
using Flux.Data:DataLoader
using Flux.Optimise:Optimiser,WeightDecay
using Flux:onehotbatch,onecold
using Flux:logitcrossentropy
using Statistics,Random
using Logging:with_logger
using TensorBoardLogger:TBLogger,tb_overwrite,set_step!,set_step_increment!
using ProgressMeter:@showprogress
import MLDatasets
using BSON:@load,@save
using CUDA
Base.@kwdef mutable struct Args
    eta = 3e-4  # learning rate
    lambda = 0  # L2 regularize parameter,implemented as weight deacy
    batchsize = 128
    epochs = 10
    seed = 0
    use_cuda = true
    infotime = 1
    checktime = 5
    tblogger = true
    savepath = "logs/"
end
function getdata(args)
    xtrain,ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest,ytest = MLDatasets.MNIST.testdata(Float32)
    xtrain = reshape(xtrain,28,28,1,:)
    xtest = reshape(xtest,28,28,1,:)
    ytrain,ytest = onehotbatch(ytrain,0:9),onehotbatch(ytest,0:9)
    train_loader = DataLoader(xtrain,ytrain,batchsize=args.batchsize,shuffle=true)
    test_loader = DataLoader(xtest,ytest,batchsize=args.batchsize,shuffle= true)
    return train_loader,test_loader
end
function build_lenet5_model(;imgsize=(28,28,1),nclasses=10)
    out_conv_size = (imgsize[1]/4-3,imgsize[2]/4-3,16)
    out_conv_size = convert(Int64,prod(out_conv_size))
    model = Chain(
        Conv((5,5),imgsize[end]=>6,relu),
        MaxPool((2,2)),
        Conv((5,5),6=>16,relu),
        MaxPool((2,2)),
        flatten,
        Dense(prod(out_conv_size),120,relu),
        Dense(120,84,relu),
        Dense(84,nclasses)
    )
    return model
end
function eval_loss_accruacy(loader,model,device)
    loss_value = 0.0f0
    acc = 0.0f0
    ntot = 0
    for (x,y) in loader
        x,y = x|>device,y|>device
        y_pred = model(x)
        loss_value += logitcrossentropy(y_pred,y)
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))
        ntot += size(x)[end]
    end
    return (loss = loss_value/ntot,acc = acc/ntot*100)
end
function train(;kws...)
    args = Args(;kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    if use_cuda
        device = gpu
        @info "Training model on GPU"
    else
        device = cpu
        @info "Training model on CPU"
    end
    # Data prepartion
    train_loader,test_loader = getdata(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"
    # model defination and Optimiser defination
    model = build_lenet5_model() |> device
    ps = Flux.params(model)
    opt = ADAM(args.eta)
    if args.lambda >0
        opt = Optimiser(opt,WeightDecay(args.lambda))
    end
    # logger setting
    if args.tblogger
        tblogger = TBLogger(args.savepath,tb_overwrite)
        set_step_increment!(tblogger,0)
        @info "TensorBoard logging at \"$(args.savepath)\""
    end
    function report(epoch)
        train_info = eval_loss_accruacy(train_loader,model,device)
        test_info = eval_loss_accruacy(test_loader,model,device)
        println("Epoch: $epoch   Train: $(train_info)   Test: $(test_info)")
        if args.tblogger
            set_step!(tblogger,epoch)
            with_logger(tblogger) do
                @info "train" loss=train_info.loss  acc=train_info.acc
                @info "test"  loss=test_info.loss   acc=test_info.acc
            end
        end
    end
    # training model process
    @info "Start training "
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x,y) in train_loader
            x,y = x|>device,y|>device
            gs = gradient(()->logitcrossentropy(model(x),y),ps)
            Flux.Optimise.update!(opt,ps,gs)
        end
        # Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime >0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath,"model.bson")
            let model = cpu(model)
                @save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end
function main()
    train()
end
main()
