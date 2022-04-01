// CornerLPNet_opencv_cpp_project.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "base.h"
#include "XDetection.h"
#include "gauss_tools.h"
#include "plot_tools.h"

void debug_gauss()
{
	vector<float> corners{ 150, 100, 250, 180, 240, 240, 140, 160 };
	RotatedRect rect = get_min_bounding_rect(corners);
	cout << rect.center << endl;
	cout << rect.size << endl;
	cout << rect.angle << endl;
	R2dGaussianDistribution gaussDis(rect);
	vector<float> vx = {195, 150, 250, 196, 215};
	vector<float> vy = {170, 100, 180, 171, 190};
	vector<float> res = gaussDis.getGaussianScores(vx, vy);
	vector<float> res2{ 4, 3, 5, 7 };
	// cout << "res: \n" << res;
}


int main()
{
	int mode = 1;  // 0 for single image and 1 for folder
	
	string onnx_path = "./weights/effS_25_3d.onnx";
	
	// imshow("1", img);
	XLPDetector xlpdetector(0.4, 0.1, 416, 416);
	int check_ret = xlpdetector.parameter_check();
	int initial_ret = xlpdetector.InitXLPDetNet(onnx_path);
	xlpdetector.InitNMS(0.2, 10);
	
	if (mode == 0) {
		string img_path = "./images/0039-1_7-358&448_447&485-447&485_361&482_358&448_444&451-0_0_15_25_24_31_20-85-5.jpg";
		Mat img = imread(img_path);
		Mat img_post;
		xlpdetector.PreProcess(img, img_post, true);
		vector<detData> resDataVec;
		int inf_ret = xlpdetector.Inference(img_post, resDataVec);
		plot_polygon_bbox(img_post, resDataVec);
		imshow("result", img_post);
		waitKey(0);
	}
	else {
		string pattern_jpg;
		vector<cv::String> image_files;
		pattern_jpg = "./images/*.jpg";
		cv::glob(pattern_jpg, image_files);
		for (int i = 0; i < image_files.size(); i++)
		{
			cout << "\nNow process image No" << i + 1 << ": "<< image_files[i] << endl;
			Mat img = imread(image_files[i]);
			Mat img_post;
			xlpdetector.PreProcess(img, img_post, true);
			vector<detData> resDataVec;
			int inf_ret = xlpdetector.Inference(img_post, resDataVec);
			plot_polygon_bbox(img_post, resDataVec);
			imshow("result", img_post);
			waitKey(0);
		}
	}
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
