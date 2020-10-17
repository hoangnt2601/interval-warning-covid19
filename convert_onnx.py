import torch
import torch.nn as nn
import onnx
import models
from models.experimental import attempt_load
from utils.activations import Hardswish

weights = 'weights/best.pt'
batch_size = 1
img_size = (640, 640)

# Input
img = torch.zeros((batch_size, 3, *img_size))

model = attempt_load(weights, map_location=torch.device('cpu'))  # load FP32 model
# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
        m.act = Hardswish()  # assign activation

# model.model[-1].export = True  # set Detect() layer export=True
model.eval()

# ONNX export
try:
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = weights.replace('.pt', '.onnx')  # filename
    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['input0'],
                        output_names=['output'])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '-1'
    onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '-1'
    # onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.save(onnx_model, 'weights/dynamic.onnx')
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)