#pragma once
#ifndef _PLOT_H_
#define _PLOT_H_
#include "base.h"
/*
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
*/

void plot_polygon_bbox(Mat& img, vector<detData> resVec);

#endif // !_PLOT_H_
