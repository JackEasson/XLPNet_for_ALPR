import cv2
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data_loader.base import cv_imread, letterbox, resizebox, base_image2tensor, decode_name2corner, read_filenames, read_filenames_from_txt
from XLPDet_pipleline import CornerLPDet
from data_loader.datasets import EasyDataSet
from utils.ap_tools import *
from strategy.post_process import CLPPostProcessor, nms_out_corners2bboxes
from utils.plot_tools import plot_polygon_bbox
import config as cfg


def main(args):
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    # ============================== Load Model ===========================
    args.device = args.device.lower()
    assert args.device == 'cpu' or isinstance(int(args.device), int)
    global_device = 'cpu'
    if args.device != 'cpu':
        assert torch.cuda.is_available(), 'gpu is not available.'
        gpu_id = int(args.device)
        print('Using gpu: %s' % torch.cuda.get_device_name(gpu_id))
        global_device = 'cuda:%s' % args.device
    print('=> Device setting: %s' % global_device)
    # ============================ model initial ==========================
    model = CornerLPDet(inp_wh=(cfg.INP_SIZE, cfg.INP_SIZE), backbone=cfg.BACKBONE, device=global_device,
                        export_onnx=False)
    model.to(global_device)
    # load weights
    print("Load weight from pretrained model ...")
    pretrained_dict = torch.load(args.weight)
    model.load_state_dict(pretrained_dict)
    print("=> Load weight successfully.")
    model.eval()

    # =============================== Post Process =================================
    post_processor = CLPPostProcessor(conf_thres=cfg.CONF_THRES, gauss_ratio=cfg.GAUSS_RATIO_2, device=global_device,
                                      nms_thres=cfg.NMS_THRES, max_obj=cfg.MAX_OBJ, use_distance=cfg.USE_DISTANCE)

    # =============================== dataset ===============================
    dataset_demo = EasyDataSet(folder_path=args.img_path, inp_size=(416, 416), use_letterbox=cfg.USE_LETTER_BOX)
    if dataset_demo.__len__() == 1 and args.batch_size != 1:
        args.batch_size = 1
    loader_demo = DataLoader(dataset_demo, num_workers=args.workers, batch_size=args.batch_size,
                             drop_last=False, shuffle=False, collate_fn=dataset_demo.base_collate_fn)
    print("=> Demo dataset total images: % d" % dataset_demo.__len__())

    with torch.no_grad():
        net_time, nms_time = 0, 0
        for step, (images_tensor, img_name_list) in enumerate(loader_demo):
            start_time = time.time()
            images_tensor = images_tensor.to(global_device)
            # ==> model feed forward
            out_score, out_corner = model(images_tensor)
            end_time = time.time()
            net_time += end_time - start_time
            for batch_idx in range(args.batch_size):
                start_time = time.time()
                img_name = img_name_list[batch_idx]
                if os.path.isdir(args.img_path):
                    img_path = os.path.join(args.img_path, img_name)
                else:
                    img_path = args.img_path
                img_mat = cv_imread(img_path)  # 解决中文路径问题
                corner_np = np.array([[0] * 8], dtype=np.float32)
                if cfg.USE_LETTER_BOX:
                    new_img, new_corner = letterbox(img=img_mat, corners=corner_np, new_shape=(cfg.INP_SIZE, cfg.INP_SIZE),
                                                    color=(0., 0., 0.))
                else:
                    new_img, new_corner = resizebox(img=img_mat, corners=corner_np, new_shape=(cfg.INP_SIZE, cfg.INP_SIZE))

                out_score_single = out_score[batch_idx]
                out_corner_single = out_corner[batch_idx]
                # NMS
                nms_out = post_processor(out_score_map=out_score_single, out_corner_map=out_corner_single)
                end_time = time.time()
                nms_time += end_time - start_time
                # if empty
                if nms_out is None or nms_out.size(0) == 0:
                    continue
                scores_tensor = nms_out[:, 0]  # size(N)
                corners_tensor = nms_out[:, 1:9]  # size(N, 4)
                corners_np = corners_tensor.cpu().numpy()  # size(N, 4)
                scores_np = scores_tensor.cpu().numpy()  # size(N)
                obj_num = nms_out.shape[0]
                for obj in range(obj_num):
                    plot_polygon_bbox(img_mat=new_img, corners=corners_np, scores=scores_np)
                cv2.imwrite(os.path.join(args.savedir, img_name), new_img)
        img_num = dataset_demo.__len__()
        net_time = net_time / img_num * 1000
        nms_time = nms_time / img_num * 1000
        print("Per image: Net forward time: %.4fms, NMS time: %.4fms" % (net_time, nms_time))
    print("Finish!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0/1/... or cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--weight', type=str, default="./output/pplcnet_0319/model_25_out.pth")
    parser.add_argument('--img_path', type=str, default="./data/example3.jpg")
    parser.add_argument('--savedir', default="./results")

    main(parser.parse_args())