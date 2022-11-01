import torch
import torch.onnx
from CassvaImgClassifier import CassvaImgClassifier

CFG = {
    #'model_arch': 'tf_efficientnet_b4_ns',
    #'model_arch': 'vit_base_patch16_224',
    #'model_arch': 'deit_base_patch16_224',
    #'model_arch': 'cait_s24_224',
    #'model_arch': 'convit_tiny',
    #'model_arch': 'inception_v4',
    'model_arch': 'resnet50',
    #'model_arch': 'coat_tiny',
    #'model_arch': 'resmlp_12_224',
    #'model_arch': 'gmlp_s16_224',
    #'model_arch': 'mixer_b16_224_in21k',
    'device': 'cuda:0'
}

print('Start loading model...')
device = torch.device(CFG['device'])
model = CassvaImgClassifier(CFG['model_arch'], 4, pretrained=True).to(device)
model.load_state_dict(torch.load('./trained_model/resnet50_fold_0_0'))
print('Load model successfull!')

BATCH_SIZE = 1
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

Output = './ONNXModels/' + CFG['model_arch'] + '.onnx'

torch.onnx.export(model, dummy_input, Output, verbose=False)
print('Convert model successfully!')