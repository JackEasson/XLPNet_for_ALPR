import os
import shutil
import random
import cv2
import torch
from data_loader.base import cv_imread, letterbox, resizebox, base_image2tensor, decode_name2corner, read_filenames, read_filenames_from_txt
from strategy.post_process import CLPPostProcessor
from utils.plot_tools import plot_polygon_bbox
import config as cfg


class CLPShowDemo2019:
    def __init__(self, oup_folder, inp_size=(704, 704), image_process=base_image2tensor, use_letterbox=True, device='cpu'):
        self.__check_path(oup_folder)
        self.oup_folder = oup_folder
        self.img_path_list = ""
        self.image_process = image_process
        self.inp_size = inp_size  # [h, w]
        self.use_letterbox = use_letterbox
        self.device = device
        self.wh_scale = [self.inp_size[1], self.inp_size[0], self.inp_size[1], self.inp_size[0]]
        self.postprocessor = CLPPostProcessor(conf_thres=cfg.CONF_THRES, gauss_ratio=cfg.GAUSS_RATIO_2, device=device,
                                              nms_thres=cfg.NMS_THRES, use_distance=cfg.USE_DISTANCE)
        self.model = None

    def set_inp_folder(self, image_path):
        self.__check_path(image_path)
        self.img_path_list = self.get_images(image_path)

    @staticmethod
    def __check_path(path):
        assert os.path.exists(path), "Target path: %s doesn't exist." % path

    @staticmethod
    def get_images(img_path):
        if img_path.endswith('.txt'):
            img_name_list = read_filenames_from_txt(img_path)
            img_path_list = [os.path.join(cfg.CCPD2019_ROOT, f) for f in img_name_list]
        elif os.path.isdir(img_path):
            files = read_filenames(img_path)
            img_path_list = [os.path.join(img_path, f) for f in files]
        else:
            img_path_list = [img_path]
        return img_path_list

    def set_model(self, model):
        self.model = model

    def __show_detection(self, img_path_list):
        cnt = 0
        with torch.no_grad():
            for img_path in img_path_list:
                cnt += 1
                img_mat = cv_imread(img_path)  # 解决中文路径问题
                # ==> [2] decode name strs for bbox and number
                img_name = os.path.basename(img_path)
                corner_np = decode_name2corner(img_name, pure_img_name=False)
                if self.use_letterbox:
                    new_img, new_corner = letterbox(img=img_mat, corners=corner_np, new_shape=self.inp_size,
                                                    color=(0., 0., 0.))
                else:
                    new_img, new_corner = resizebox(img=img_mat, corners=corner_np, new_shape=(416, 416))
                # ==> [3] bboxes normalization
                image_tensor = self.image_process(new_img).to(self.device)
                # ==> [4] forward
                out_score, out_corner = self.model(image_tensor.unsqueeze(0))
                # ==> post process
                # nms_out: tensor2d, size(n, 10)
                nms_out = self.postprocessor(out_score_map=out_score, out_corner_map=out_corner)
                # if empty
                if nms_out is None or nms_out.size(0) == 0:
                    cv2.imwrite(os.path.join(self.oup_folder, "%d.jpg" % cnt), new_img)
                    continue
                scores_tensor = nms_out[:, 0]  # size(N)
                corners_tensor = nms_out[:, 1:9]  # size(N, 4)
                corners_np = corners_tensor.cpu().numpy()  # size(N, 4)
                scores_np = scores_tensor.cpu().numpy()  # size(N)
                obj_num = nms_out.shape[0]
                for obj in range(obj_num):
                    plot_polygon_bbox(img_mat=new_img, corners=corners_np, scores=scores_np)
                cv2.imwrite(os.path.join(self.oup_folder, "%d.jpg" % cnt), new_img)

    def ordinary_show(self):
        # if empty
        if len(os.listdir(self.oup_folder)) != 0:
            shutil.rmtree(self.oup_folder)
            os.mkdir(self.oup_folder)
        self.__show_detection(self.img_path_list)

    def random_show(self, n=10):
        # if empty
        if len(os.listdir(self.oup_folder)) != 0:
            shutil.rmtree(self.oup_folder)
            os.mkdir(self.oup_folder)
        part_paths = random.choices(self.img_path_list, k=min(n, len(self.img_path_list)))
        self.__show_detection(part_paths)


if __name__ == '__main__':
    shower = CLPShowDemo2019('./results')
    shower.set_inp_folder(cfg.CCPD2019['test_txt'])
    print(shower.img_path_list[:10])
    img = cv2.imread(shower.img_path_list[1])
    cv2.imshow('1', img)
    cv2.waitKey()
    # shower.random_show(n=2)