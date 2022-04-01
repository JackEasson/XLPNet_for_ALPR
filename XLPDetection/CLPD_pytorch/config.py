# ===================== [1] dataset path =====================

# CCPD2019_ROOT = "/home/r8003/workspace/zw_workspace/ccpd/CCPD2019/"
CCPD2019_ROOT = "E:/datasets/ccpd/CCPD2019/CCPD2019"
CCPD2019 = {
    'root': CCPD2019_ROOT,
    'train_txt': CCPD2019_ROOT + "/splits/train.txt",
    'val_txt': CCPD2019_ROOT + "/splits/val.txt",
    'test_txt': CCPD2019_ROOT + "/splits/test.txt",
}


saveParentFolder = './output'

# ================= global training arguments ================
AUGMENT = {
    'aug_probs': 0.2,
    'select_probs': 0.5,
    'refresh_probs': 0.1,
    'max_optor': 5,
    'contrast': True,
    'blur': True,
    'hsv': True,
    'noise': True,
    'flip': False,
    'translate': True,
    'rotate': True,
}

INP_SIZE = 416  # 320, 512  输入尺寸
BACKBONE = 'efficientnetv2_s'  # 'pplcnet', 'mobilenext', 'efficientnetv2_lite', 'efficientnetv2_s'

LR = 2e-3
LR_SCHEDULER = 'cos'  # (step, exp, cos)  # 学习率调整策略
OPTIMIZER = 'adam'  # 'adam', 'sgd'  # 优化器

# about gauss
GAUSS_RATIO_1 = 1.0  # for label assignment
GAUSS_RATIO_2 = 2.0  # for NMS

# post process
CONF_THRES = 0.4  # 后处理置信度阈值
NMS_THRES = 0.1  # NMS高斯分数阈值
AP_INDEX = 0.7  # AP0.7

DET_LAMBDA = 1.5  # 损失函数定位权重

MAX_OBJ = 50  # 最多保留的检测对象数
USE_AUGMENT = True  # 是否使用数据增广
USE_LETTER_BOX = True  # 是否使用letterbox
USE_DISTANCE = True  # 是否使用距离项

