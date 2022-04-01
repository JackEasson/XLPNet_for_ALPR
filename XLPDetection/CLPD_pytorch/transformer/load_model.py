from XLPDet_pipleline import *


def load():
    weight = "../output/test20220227_shift/model_best.pth"
    target = "../weights/CLPDetNet_es.pth"
    backbone = 'efficientnetv2_s'
    model = CornerLPDet(inp_wh=(416, 416), backbone=backbone, device='cpu', export_onnx=True)
    model.eval()
    # load weights
    print("Load weight from pretrained model ...")
    pretrained_dict = torch.load(weight)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    torch.save(model, target)
    print("Save finish!")


if __name__ == '__main__':
    load()