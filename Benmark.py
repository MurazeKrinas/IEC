import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import Normalize
from PIL import Image
import timeit
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from skimage import io
from skimage.transform import resize

CFG = {
    #'model_arch': 'tf_efficientnet_b4', #OK (Just ONNX, Opset = 11)
    #'model_arch': 'convit_tiny', #OK (Just ONNX, Opset = 11)

    #'model_arch': 'cait_s24_224', #OK (Opset = 11)
    #'model_arch': 'coat_tiny', #OK (Opset = 10)
    #'model_arch': 'gmlp_s16_224', #OK (Opset = 11)
    #'model_arch': 'inception_v4', #OK (Opset = 11)
    'model_arch': 'resnet50', #OK (Opset = 11)
    #'model_arch': 'mixer_b16_224_in21k', #OK (Opset = 11)
    #'model_arch': 'deit_base_patch16_224', #OK (Opset = 11)
    #'model_arch': 'vit_base_patch16_224', #OK (Opset = 11)
    
    #'model_arch': 'resmlp_12_224', #ERROR: Operator addcmul
    'device': 'cuda:0',
    'batch_size': 1
}

def CreatTestBatch():
    ImgPATH='./Dataset/test_images/Image/test4.png'
    img = resize(io.imread(ImgPATH), (224, 224))
    input_batch = np.array(np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), CFG['batch_size'], axis=0), dtype=np.float32)
    print(f'Shape: {input_batch.shape}')
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
    PATH = f'./TRTModels/{CFG["model_arch"]}.trt'
    f = open(PATH, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    model = runtime.deserialize_cuda_engine(f.read())
    context = model.create_execution_context()
    print(f'Model: {CFG["model_arch"]}')
    print('Load model successfull!')

    print('\nAllocating input and output memory...')
    output = np.empty([CFG['batch_size'], 1000], dtype = np.float16) 

    InputBatch = CreatTestBatch()
    d_input = cuda.mem_alloc(1 * InputBatch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    print('Allocating sucessfully!')

    print('\nStart validating...')
    Avg = 0
    preprocessed_images = np.array([preprocess_image(image) for image in InputBatch])

    start = timeit.default_timer()
    pred = predict(preprocessed_images)
    print(pred)
    stop = timeit.default_timer()
    Avg += (stop - start)
    print(f'Time: {Avg}')
    print('Validating successfull!')

    '''
    print('Verify our TensorRT output is still accurate...')
    indices = (-pred[0]).argsort()[:5]
    print("Class | Probability (out of 1)")
    print(list(zip(indices, pred[0][indices])))
    '''
    
