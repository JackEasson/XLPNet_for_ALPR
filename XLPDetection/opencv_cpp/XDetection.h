#pragma once
//****yolov4检测接口定义****//
# ifndef _XDETECTION_H_
# define _XDETECTION_H_ 

# include "base.h"
#include "gauss_tools.h"
#include <time.h>

struct DetectData
{
	float score;                    //类别
	cv::Rect rect;              //坐标信息
};

class XLPDetector
{
private:
	float confThreshold; // 置信度阈值（Confidence threshold）
	float nmsThreshold;  // 非极大值抑制阈值（Non-maximum suppression threshold）
	int inpWidth;  // 输入检测网络的图像的宽度320，416，608
	int inpHeight; // 输入检测网络的图像的高度320，416，608
	dnn::Net XLPDetNet;
	GaussianNMS gaussNMS;
	vector<string> OutputsNames;

public:
	vector<string> Dclasses;
	XLPDetector(); // 构造函数1
	XLPDetector(float confThres, float nmsThres, int Width, int Height); // 构造函数2
	~XLPDetector() { cout << "XLPDetector Destructor Called" << endl; }
	int parameter_check();
	void show_member();
	int InitXLPDetNet(string modelWeight);
	void InitNMS(float ratio, int obj_n);
	void PreProcess(Mat& frame, Mat& out_frame, bool use_letterbox);
	vector<detData> PostProcess(Mat& scoreMat, Mat& cornerMat);
	vector<string> getOutputsNames();
	int Inference(Mat& Image, vector<detData>& resDataVec);
	// void ResultSave(Mat Image, vector<ClassData> resultData, string save_name);
};


# endif
