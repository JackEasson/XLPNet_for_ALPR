#include "plot_tools.h"
#include "base.h"


void plot_polygon_bbox(Mat& img, vector<detData> resVec)
{
	Scalar colorPoly = Scalar(0, 255, 0);
	Scalar colorRect = Scalar(0, 0, 255);
	int num = resVec.size();
	for (int i = 0; i < num; i++) {
		float score = resVec[i].score;
		vector<int> corners(resVec[i].corners.begin(), resVec[i].corners.end());

		vector<Point2i> points = conersInt2Points(corners);

		// 绘制内部四边形
		polylines(img, points, true, colorPoly, 2, LINE_AA);
		// 绘制矩形
		Rect bRect = coners2bbox(corners);
		rectangle(img, bRect, colorRect, 1, LINE_8);
		// 显示分数
		Point2i tl = Point(bRect.tl().x, bRect.tl().y - 22);
		Rect labelRect = Rect(tl, Size(46, 22));
		rectangle(img, labelRect, colorRect, -1, LINE_8);
		string text = floatConvert2String(score);
		putText(img, text, Point(bRect.tl().x, bRect.tl().y - 5), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, Scalar(255, 255, 255), 0.6, LINE_AA);
	}
}