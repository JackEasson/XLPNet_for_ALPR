#include "base.h"


void letterbox_image(Mat& img, Mat& out_img, int tar_w, int tar_h, Scalar pad_scalar)
{
	int src_w = img.cols;
	int src_h = img.rows;
	double r = min(double(tar_w) / src_w, double(tar_h) / src_h);
	int new_w = int(src_w * r);
	int new_h = int(src_h * r);
	int dw = tar_w - new_w;
	int dh = tar_h - new_h;
	double half_w = double(dw) / 2.;
	double half_h = double(dh) / 2.;
	Mat img_resized;
	resize(img, img_resized, Size(new_w, new_h));  // Size(w, h)

	int top = int(half_h - 0.1 + 0.5);
	int bottom = int(half_h + 0.1 + 0.5);
	int left = int(half_w - 0.1 + 0.5);
	int right = int(half_w + 0.1 + 0.5);

	copyMakeBorder(img_resized, out_img, top, bottom, left, right, BORDER_CONSTANT, pad_scalar);
}

vector<Point2f> conersFloat2Points(vector<float> cornersVec)
{
	vector<Point2f> pointsVec;
	for (int i = 0; i < 4; i++) {
		Point2f p(cornersVec[i * 2], cornersVec[i * 2 + 1]);
		pointsVec.push_back(p);
	}
	return pointsVec;
}

vector<Point2i> conersInt2Points(vector<int> cornersVec)
{
	vector<Point2i> pointsVec;
	for (int i = 0; i < 4; i++) {
		Point2i p(cornersVec[i * 2], cornersVec[i * 2 + 1]);
		pointsVec.push_back(p);
	}
	return pointsVec;
}

Rect coners2bbox(vector<int> cornersVec)
{
	vector<Point2i> pointsVec = conersInt2Points(cornersVec);
	Rect bRect = boundingRect(pointsVec);
	return bRect;
}

string floatConvert2String(float fNum)
{
	stringstream ss;
	ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << fNum;
	string str = ss.str();
	return str;
}

ostream& operator << (ostream& os, vector<detData>& v)
{
	int n = v.size();
	for (int i = 0; i < n; i++) {
		os << "Score:" << v[i].score << ", Corners:";
		for (int j = 0; j < v[i].corners.size(); j++) {
			os << v[i].corners[j] << ", ";
		}
		os << endl;
	}
	return os;
}

bool compareDetData(detData d1, detData d2)
{
	return d1.score > d2.score;  //Ωµ–Ú≈≈¡–
}