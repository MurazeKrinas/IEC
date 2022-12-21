import torch
import fnmatch
from torchvision.transforms import Normalize
import timeit
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from skimage import io
from skimage.transform import resize

Model = {
    'arch': 'Resnet8_V4_Fp16',
    'type': np.float16,
    # 'arch': 'Resnet8_V4_Int8',
    # 'type': np.int8,
    'device': 'cuda:0',
    'batch_size': 1
}

def CreatTestBatch(num):
    ImgPATH=f'./Dataset/Images/test_images/test{num}.jpg'
    img = resize(io.imread(ImgPATH), (112, 112))
    input_batch = np.array(np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), Model['batch_size'], axis=0), dtype=np.float32)
    #print(f'Shape: {input_batch.shape}')
    return input_batch

def preprocess_image(img):
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    return np.array(result, dtype=np.float16)

def predict(batch): # result gets copied into output
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
    print('Start loading model...')
    PATH = f'./TRTModels/{Model["arch"]}.trt'
    f = open(PATH, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    model = runtime.deserialize_cuda_engine(f.read())
    context = model.create_execution_context()
    print(f'Model: {Model["arch"]}')
    print('Load model successfull!')

    dir_path = r'./Dataset/Images/test_images'
    NumImg = len(fnmatch.filter(os.listdir(dir_path), '*.*'))
    Avg = 0.0
    for num in range (NumImg):
        print(f'Allocating input and output memory for image {num}: ')
        output = np.empty([Model['batch_size'], 1000], dtype = Model['type']) 

        InputBatch = CreatTestBatch(num)
        d_input = cuda.mem_alloc(InputBatch.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        print('Allocating sucessfully!')

        print(f'\nStart validating image {num}: ')
        preprocessed_images = np.array([preprocess_image(image) for image in InputBatch])

        start = timeit.default_timer()
        pred = predict(preprocessed_images)
        stop = timeit.default_timer()
        #print(pred)
        Avg += (stop - start) / NumImg 
        print(f'Time for image {num}: {stop-start} second')
        print('---------------------------')

    print(f'Average validating time per image of {Model["arch"]}.trt: {Avg} second')
    

    f = open("Benchmark.txt", "a")
    s = f'Average validating time per image of {Model["arch"]}.trt: {str(Avg)} second\n'
    f.write(s)
    f.close()
