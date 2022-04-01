import os
from tqdm import tqdm
import numpy as np
import random
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from utils.general import cv2_get_min_bounding_rect, corner2bbox_np
from data_loader.base import (cv_imread, decode_name2corner, decode_name2bbox, read_filenames_from_txt, read_filenames,
                                           letterbox, resizebox, bbox_letterbox, bbox_resizebox, base_image2tensor)
from utils.plot_tools import plot_polygon_bbox
from data_loader.augmentation import ObjectDetectionAugmentation


# corner version
class CCPDDataSet2019(Dataset):
    def __init__(self, folder, data_txt_path, inp_size=(416, 416), image_process=base_image2tensor,
                 use_letterbox=True, use_augment=False, limit=-1):
        super(CCPDDataSet2019, self).__init__()  # 调用父类初始化
        self.folder = folder
        self.data_txt_path = data_txt_path
        self.image_process = image_process
        self.use_letterbox = use_letterbox
        self.inp_size = inp_size  # [h, w]
        self.wh_scale = [self.inp_size[1], self.inp_size[0], self.inp_size[1], self.inp_size[0]]
        self.gt_labels_global = self.__data_init()  # -> image names list
        self.use_augment = use_augment
        if use_augment:
            self.augmenter = ObjectDetectionAugmentation()
        self.limit = limit if len(self.gt_labels_global) > limit else -1  # -1: get all samples
        self.gt_labels = None
        self.random_subset()

    def __data_init(self):
        gt_labels = read_filenames_from_txt(self.data_txt_path)
        return gt_labels

    def random_subset(self):
        if self.limit == - 1:
            gt_labels = self.gt_labels_global
        else:
            gt_labels = random.sample(self.gt_labels_global, self.limit)
        self.gt_labels = gt_labels

    def __getitem__(self, index):
        """
        NOTICE: corners are all in input scale
        :param index:
        :return: img_tensor size(608, 608); coords_tensor size(4);
                 lp_label_tensor list(obj_num), each size(7 or 8);
                 lp_label_length list(obj_num)
        """
        img_name = self.gt_labels[index]
        # ==> [1] read image -- cv_imread can deal with chinese path
        img_mat = cv_imread(os.path.join(self.folder, img_name))
        # ==> [2] decode name strs for bbox and number
        corners_np = decode_name2corner(img_name, pure_img_name=False)
        if self.use_augment:
            img_mat, corners_np = self.augmenter(img_mat, corners_np)
        if self.use_letterbox:
            # color=(27.5, 127.5, 127.5)
            new_img, corners_np = letterbox(img=img_mat, corners=corners_np, new_shape=self.inp_size, color=(0., 0., 0.))
        else:
            new_img, corners_np = resizebox(img=img_mat, corners=corners_np, new_shape=(416, 416))
        # ==> [3] get minRect
        minRect_list = []
        for i in range(corners_np.shape[0]):
            corners_cur = corners_np[i].reshape(4, 2) + 0.5
            corners_cur = corners_cur.astype(np.int32)
            minRect_list.append(cv2_get_min_bounding_rect(corners_cur))
        minRect_np = np.stack(minRect_list, axis=0)
        # ==> [4] to tensor
        image_tensor = self.image_process(new_img)
        corner_tensor = torch.as_tensor(corners_np, dtype=torch.float32)  # size(n, 8)
        minRect_tensor = torch.as_tensor(minRect_np, dtype=torch.float32)  # size(n, 8)
        # here number_len will be list
        return image_tensor, corner_tensor, minRect_tensor, img_name

    # and we set the collate method
    @staticmethod
    def base_collate_fn(batch):
        image_data = [item[0] for item in batch]
        image_tensor = torch.stack(image_data, dim=0)  # size(batch, c, h, w)
        corners_list = [item[1] for item in batch]
        minRects_list = [item[2] for item in batch]
        img_names_list = [item[3] for item in batch]
        return image_tensor, corners_list, minRects_list, img_names_list

    def __len__(self):
        return len(self.gt_labels)


# corner version
class BBOXCCPDDataSet2019(Dataset):
    def __init__(self, folder, data_txt_path, inp_size=(416, 416), image_process=base_image2tensor,
                 use_letterbox=True, limit=-1):
        super(BBOXCCPDDataSet2019, self).__init__()  # 调用父类初始化
        self.folder = folder
        self.data_txt_path = data_txt_path
        self.image_process = image_process
        self.use_letterbox = use_letterbox
        self.inp_size = inp_size  # [h, w]
        self.wh_scale = [self.inp_size[1], self.inp_size[0], self.inp_size[1], self.inp_size[0]]
        self.gt_labels_global = self.__data_init()  # -> image names list
        self.limit = limit if len(self.gt_labels_global) > limit else -1  # -1: get all samples
        self.gt_labels = None
        self.random_subset()

    def __data_init(self):
        gt_labels = read_filenames_from_txt(self.data_txt_path)
        return gt_labels

    def random_subset(self):
        if self.limit == - 1:
            gt_labels = self.gt_labels_global
        else:
            gt_labels = random.sample(self.gt_labels_global, self.limit)
        self.gt_labels = gt_labels

    def __getitem__(self, index):
        """
        NOTICE: corners are all in input scale
        :param index:
        :return: img_tensor size(608, 608); coords_tensor size(4);
                 lp_label_tensor list(obj_num), each size(7 or 8);
                 lp_label_length list(obj_num)
        """
        img_name = self.gt_labels[index]
        # ==> [1] read image -- cv_imread can deal with chinese path
        img_mat = cv_imread(os.path.join(self.folder, img_name))
        # ==> [2] decode name strs for bbox and number
        corners_np = decode_name2corner(img_name, pure_img_name=False)
        bbox_np = decode_name2bbox(img_name, pure_img_name=False)
        if self.use_letterbox:
            # color=(27.5, 127.5, 127.5)
            new_img1, corners_np = letterbox(img=img_mat, corners=corners_np, new_shape=self.inp_size, color=(0., 0., 0.))
            new_img2, bboxes_np = bbox_letterbox(img=img_mat, bboxes=bbox_np, new_shape=self.inp_size, color=(0., 0., 0.))
        else:
            new_img1, corners_np = resizebox(img=img_mat, corners=corners_np, new_shape=(416, 416))
            new_img2, bboxes_np = bbox_resizebox(img=img_mat, bboxes=bbox_np, new_shape=(416, 416))
        # ==> [3] to tensor
        image_tensor = self.image_process(new_img2)
        corner_tensor = torch.as_tensor(corners_np, dtype=torch.float32)  # size(n, 8)
        bbox_tensor = torch.as_tensor(bboxes_np, dtype=torch.float32)  # size(n, 4)
        # here number_len will be list
        return image_tensor, corner_tensor, bbox_tensor, img_name

    # and we set the collate method
    @staticmethod
    def base_collate_fn(batch):
        image_data = [item[0] for item in batch]
        image_tensor = torch.stack(image_data, dim=0)  # size(batch, c, h, w)
        corners_list = [item[1] for item in batch]
        bboxes_list = [item[2] for item in batch]
        img_names_list = [item[3] for item in batch]
        return image_tensor, corners_list, bboxes_list, img_names_list

    def __len__(self):
        return len(self.gt_labels)


class EasyDataSet(Dataset):
    def __init__(self, folder_path, inp_size=(416, 416), image_process=base_image2tensor,
                 use_letterbox=True):
        super(EasyDataSet, self).__init__()  # 调用父类初始化
        self.folder_path = folder_path
        self.image_process = image_process
        self.use_letterbox = use_letterbox
        self.inp_size = inp_size  # [h, w]
        self.wh_scale = [self.inp_size[1], self.inp_size[0], self.inp_size[1], self.inp_size[0]]
        self.gt_labels = self.__data_init()

    def __data_init(self):
        if os.path.isdir(self.folder_path):
            gt_labels = read_filenames(self.folder_path)
            return gt_labels
        else:
            return [self.folder_path]

    def __getitem__(self, index):
        """
        NOTICE: corners are all in input scale
        :param index:
        :return: img_tensor size(608, 608); coords_tensor size(4);
                 lp_label_tensor list(obj_num), each size(7 or 8);
                 lp_label_length list(obj_num)
        """
        img_path = self.gt_labels[index]
        img_name = os.path.split(img_path)[-1]
        # ==> [1] read image -- cv_imread can deal with chinese path
        img_mat = cv_imread(img_path)
        # ==> [2] decode name strs for bbox and number
        corners_np = np.array([[0] * 8], dtype=np.float32)
        if self.use_letterbox:
            # color=(27.5, 127.5, 127.5)
            new_img, corners_np = letterbox(img=img_mat, corners=corners_np, new_shape=self.inp_size, color=(0., 0., 0.))
        else:
            new_img, corners_np = resizebox(img=img_mat, corners=corners_np, new_shape=(416, 416))
        # ==> [4] to tensor
        image_tensor = self.image_process(new_img)
        # here number_len will be list
        return image_tensor, img_name

    # and we set the collate method
    @staticmethod
    def base_collate_fn(batch):
        image_data = [item[0] for item in batch]
        image_tensor = torch.stack(image_data, dim=0)  # size(batch, c, h, w)
        img_names_list = [item[1] for item in batch]
        return image_tensor, img_names_list

    def __len__(self):
        return len(self.gt_labels)


if __name__ == '__main__':
    # LPDataset = CCPDDataSet2019(folder='E:\\datasets\\ccpd\\CCPD2019\\CCPD2019',
    #                             data_txt_path='E:\\datasets\\ccpd\\CCPD2019\\CCPD2019\\splits\\train.txt',
    #                             inp_size=(416, 416), image_process=base_image2tensor, use_letterbox=True, use_augment=True, limit=-1)
    # print(LPDataset.__len__())
    # trainloader = DataLoader(LPDataset, batch_size=10, num_workers=4, shuffle=False,
    #                          drop_last=False, collate_fn=LPDataset.base_collate_fn)
    # counter = 0
    # for i, (image_tensor, corner_list, minRect_list, img_name) in tqdm(enumerate(trainloader)):
    #     # if first_label_tensor[0] != second_label_tensor[0]:
    #     print('0', image_tensor.requires_grad_())
    #     print('1', corner_list)
    #     print('2', minRect_list)
    #     print('3', img_name)
    #     img_np = image_tensor.numpy()
    #     print(img_np.shape)
    #     for j in range(img_np.shape[0]):
    #         img = img_np[j]
    #         img = img.transpose(1, 2, 0)  # c,h,w -> h,w,c
    #         img = (img + 1.0) / 2.0 * 255
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    #         # img = img.astype(np.int)
    #         corner = corner_list[j].numpy()
    #         plot_polygon_bbox(img_mat=img, corners=corner, scores=[0.978], stage_lvl=[0])
    #         cv2.imshow("cur", img)
    #         cv2.waitKey()
    #         # counter += 1
    #     break
    # ===================================================================================================
    # LPDataset = BBOXCCPDDataSet2019(folder='E:\\datasets\\ccpd\\CCPD2019\\CCPD2019',
    #                                 data_txt_path='E:\\datasets\\ccpd\\CCPD2019\\CCPD2019\\splits\\train.txt',
    #                                 inp_size=(416, 416), image_process=base_image2tensor, use_letterbox=True, limit=-1)
    # print(LPDataset.__len__())
    # trainloader = DataLoader(LPDataset, batch_size=2, num_workers=4, shuffle=False,
    #                          drop_last=False, collate_fn=LPDataset.base_collate_fn)
    # counter = 0
    # for i, (image_tensor, corner_list, bbox_list, img_name) in tqdm(enumerate(trainloader)):
    #     # if first_label_tensor[0] != second_label_tensor[0]:
    #     print('0', image_tensor.shape)
    #     print('1', corner_list)
    #     print('2', bbox_list)
    #     print('3', img_name)
    #     img_np = image_tensor.numpy()
    #     print(img_np.shape)
    #     for j in range(img_np.shape[0]):
    #         img = img_np[j]
    #         img = img.transpose(1, 2, 0)  # c,h,w -> h,w,c
    #         img = (img + 1.0) / 2.0 * 255
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    #         # img = img.astype(np.int)
    #         corner = corner_list[j].numpy()
    #         print(corner)
    #         bbox_trans = corner2bbox_np(corner[0])
    #         print('!!!', bbox_trans, bbox_list[j])
    #         plot_polygon_bbox(img_mat=img, corners=corner, scores=[0.978], stage_lvl=[0])
    #         cv2.imshow("cur", img)
    #         cv2.waitKey()
    #         # counter += 1
    #     break

    LPDataset = EasyDataSet(folder_path='../data', inp_size=(416, 416), image_process=base_image2tensor, use_letterbox=True)
    print(LPDataset.__len__())
    trainloader = DataLoader(LPDataset, batch_size=1, num_workers=4, shuffle=False,
                             drop_last=False, collate_fn=LPDataset.base_collate_fn)
    counter = 0
    for i, (image_tensor, img_name) in tqdm(enumerate(trainloader)):
        print('0', image_tensor.shape)
        print('1', img_name)