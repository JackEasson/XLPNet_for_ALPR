import io
import torch
import torch.onnx
from XLPDet_pipleline import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trans2onnx():
    # model = OcrNet(cnn_name='vgg', rnn_name='gru', seq_len=27, class_num=15)

    pthfile = "../weights/pplcnet_15_3d.pth"
    # loaded_model = torch.load(pthfile, map_location='cpu')
    model = torch.load(pthfile, map_location='cpu')
    model = model.eval()
    # try:
    #   loaded_model.eval()
    # except AttributeError as error:
    #   print(error)

    # model.load_state_dict(loaded_model['state_dict'])
    # model = model.to(device)

    # data type nchw
    dummy_input1 = torch.rand(1, 3, 416, 416)
    input_names = ["input"]
    # output_names = ["output1", "output2", "output3", "output4"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input1, "../weights/pplcnet_15_3d.onnx", verbose=True, input_names=input_names,
                      output_names=output_names, opset_version=11)


if __name__ == "__main__":
    trans2onnx()

