#pragma once
#ifndef _BASE_H_
#define _BASE_H_

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <ostream>
#include <algorithm>
#include<sstream>

using namespace std;
using namespace cv;


struct detData
{
	float score = 0.;
	vector<float> corners = vector<float>(8, 0.);
};

void letterbox_image(Mat& img, Mat& out_img, int tar_w, int tar_h, Scalar pad_scalar);

vector<Point2f> conersFloat2Points(vector<float> cornersVec);
vector<Point2i> conersInt2Points(vector<int> cornersVec);

Rect coners2bbox(vector<int> cornersVec);

string floatConvert2String(float fNum);


bool compareDetData(detData d1, detData d2);

/***************** Mat转vector **********************/
template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat& mat)
{
	return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}

/****************** vector转Mat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
	return dest;
}

/****************** 输出运算符重载1 *********************/
template<typename _Tp>
ostream& operator << (ostream& os, vector<_Tp>& v)
{
	int n = v.size();
	for (int i = 0; i < n; i++) {
		os << v[i] << ", ";
	}
	os << endl;
	return os;
}

/****************** 输出运算符重载2 *********************/
ostream& operator << (ostream& os, vector<detData>& v);

#endif
