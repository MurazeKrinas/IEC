from Settings import *

def predict(batch): # result gets copied into 
    
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output

if __name__ == '__main__':
    start = timeit.default_timer()
    print('Start loading model...')
    PATH = f'./TRTModels/{CFG["model_arch"]}.trt'
    f = open(PATH, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    model = runtime.deserialize_cuda_engine(f.read())
    context = model.create_execution_context()
    print(f'Model: {CFG["model_arch"]}')
    print('Load model successfull!')

    print('\nLoading dataset...')
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.Resize((CFG['img_size'],CFG['img_size'])), transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset = datasets.ImageFolder('./Dataset/Images/', transform=transform_norm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)
    print('Load dataset successfully!')

    print(f'Allocating input and output memory...')
    Iteration = len(dataloader)
    output = np.empty([CFG['batch_size'], 4], dtype = CFG['type']) 
    InputBatch,_ = next(iter(dataloader))
    InputBatch = InputBatch.numpy().astype(CFG['type'])

    d_input = cuda.mem_alloc(InputBatch.nbytes * Iteration * CFG['batch_size'])
    d_output = cuda.mem_alloc(output.nbytes * Iteration * CFG['batch_size'])
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    print('Allocating sucessfully!')

    print('\nWarming up...')
    for i in range(5):
        dummy_input = torch.rand((1, 3, CFG['img_size'], CFG['img_size']))
        dummy_input = dummy_input.numpy().astype(CFG['type'])
        predict(dummy_input)
    print('Warm up done!')
    stop = timeit.default_timer()
    Preprocess = stop - start
    print(f'Preprocess: {Preprocess}')

    Avg = 0.0
    num = 0
    print(f"Batchsize: {CFG['batch_size']}")
    print(f'Number of iteration: {Iteration}')
    for Elm in dataloader:
        image,_ = next(iter(dataloader))
        image = image.numpy().astype(CFG['type'])
        print(f'\nStart validating iteration {num}: ')
        start = timeit.default_timer()
        pred = predict(image)
        stop = timeit.default_timer()
        print(pred)
        Avg += (stop - start)
        num += 1
        print(f'Time for iteration {num}: {stop-start} second')
        print('---------------------------')

    Avg /= Iteration
    print(f'Average validating time per image of {CFG["model_arch"]}.trt: {Avg} second')

    f = open("Benchmark.txt", "a")
    s = f'Average validating time per image of {CFG["model_arch"]}.trt: {Avg} second\nPreprocess: {Preprocess}\n'
    f.write(s)
    f.close()
