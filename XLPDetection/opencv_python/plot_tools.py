import cv2
import numpy as np

"""
#define CV_COLOR_RED cv::Scalar(0,0,255)       //纯红
#define CV_COLOR_GREEN cv::Scalar(0,255,0)        //纯绿
#define CV_COLOR_BLUE cv::Scalar(255,0,0)       //纯蓝

#define CV_COLOR_DARKGRAY cv::Scalar(169,169,169) //深灰色
#define CV_COLOR_DARKRED cv::Scalar(0,0,139) //深红色
#define CV_COLOR_ORANGERED cv::Scalar(0,69,255)     //橙红色

#define CV_COLOR_CHOCOLATE cv::Scalar(30,105,210) //巧克力
#define CV_COLOR_GOLD cv::Scalar(10,215,255) //金色
#define CV_COLOR_YELLOW cv::Scalar(0,255,255)     //纯黄色

#define CV_COLOR_OLIVE cv::Scalar(0,128,128) //橄榄色
#define CV_COLOR_LIGHTGREEN cv::Scalar(144,238,144) //浅绿色
#define CV_COLOR_DARKCYAN cv::Scalar(139,139,0)     //深青色


#define CV_COLOR_SKYBLUE cv::Scalar(230,216,173) //天蓝色
#define CV_COLOR_INDIGO cv::Scalar(130,0,75) //藏青色
#define CV_COLOR_PURPLE cv::Scalar(128,0,128)     //紫色

#define CV_COLOR_PINK cv::Scalar(203,192,255) //粉色
#define CV_COLOR_DEEPPINK cv::Scalar(147,20,255) //深粉色
#define CV_COLOR_VIOLET cv::Scalar(238,130,238)     //紫罗兰
"""


def corner2bbox_np(corners):
    """
    :param corners: size(8)
    :return: size(4)
    """
    left = np.min(corners[::2])
    top = np.min(corners[1::2])
    right = np.max(corners[::2])
    bottom = np.max(corners[1::2])
    bbox = np.array([left, top, right, bottom])
    return bbox


# =========================== base plots ===========================
def plot_polygon_bbox(img_mat, corners, scores=None, stage_lvl=None):
    """
    :param img_mat: image mat
    :param corners: numpy, 2d
    :param scores: numpy, 1d
    :param stage_lvl: numpy, 1d
    :return:
    """
    corners = (corners + 0.5).astype(np.int)
    poly_color = (0, 255, 0)  # BGR
    poly_thickness = 2

    box_color = (0, 0, 255)  # BGR
    box_thickness = 1

    # part1 out
    obj_num = corners.shape[0]
    for obj in range(obj_num):
        points = corners[obj].reshape(4, 2)
        # polygon
        for i in range(3):
            cv2.line(img_mat, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]),
                     poly_color, poly_thickness, cv2.LINE_AA)
        cv2.line(img_mat, (points[3][0], points[3][1]), (points[0][0], points[0][1]),
                 poly_color, poly_thickness, cv2.LINE_AA)
        # bbox
        bbox = corner2bbox_np(corners[obj])
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img_mat, (x1, y1), (x2, y2), box_color, box_thickness)
        # score infos
        if scores is not None:
            cv2.rectangle(img_mat, (x1, y1 - 22), (x1 + 46, y1), box_color, thickness=-1)  # 实心框
            cv2.putText(img_mat, "%.2f" % scores[obj], (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if stage_lvl is not None:
            cv2.rectangle(img_mat, (x1, y2), (x1 + 22, y2 + 22), box_color, thickness=-1)  # 实心框
            cv2.putText(img_mat, "%d" % stage_lvl[obj], (x1 + 4, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


if __name__ == '__main__':
    img = cv2.imread("E:\\images\\1.jpg")
    corners = np.array([[100, 100, 200, 80, 200, 180, 100, 150]])
    scores = np.array([0.9241])
    plot_polygon_bbox(img, corners, scores)
    cv2.imshow('1', img)
    cv2.waitKey()

