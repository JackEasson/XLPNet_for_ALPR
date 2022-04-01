import torch
import torch.nn as nn
from XLPDet_pipleline import CornerLPDet
import config as cfg


def multi_gpu2normal_model():
    weight = "./output/pplcnet/model_best.pth"
    global_device = 'cpu'
    model = CornerLPDet(inp_wh=(cfg.INP_SIZE, cfg.INP_SIZE), backbone=cfg.BACKBONE, device=global_device,
                        export_onnx=False)
    gpus = [0, 1]
    model = nn.DataParallel(model.to(global_device), device_ids=gpus, output_device=gpus[0])
    print("Load weight from pretrained model ...")
    pretrained_dict = torch.load(weight)
    model.load_state_dict(pretrained_dict)
    print("=> Load weight successfully.")
    print("Save model's weights ...")
    torch.save(model.module.state_dict(), "./output/pplcnet/model_best_out.pth")
    print("Save finish ...")


def state_dict2whole_model():
    weight = "./output/pplcnet_0319/model_15_out.pth"
    target = "./weights/pplcnet_15_3d.pth"
    global_device = 'cpu'
    model = CornerLPDet(inp_wh=(cfg.INP_SIZE, cfg.INP_SIZE), backbone=cfg.BACKBONE, device=global_device,
                        export_onnx=True)
    print("Load weight from pretrained model ...")
    pretrained_dict = torch.load(weight)
    model.load_state_dict(pretrained_dict)
    print("=> Load weight successfully.")
    print("Save model's weights ...")
    torch.save(model, target)
    print("Save finish!")


if __name__ == '__main__':
    # multi_gpu2normal_model()
    state_dict2whole_model()