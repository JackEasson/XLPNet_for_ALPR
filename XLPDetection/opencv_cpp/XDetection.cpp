#include "XDetection.h"
#include "plot_tools.h"

// ȫ�ֲ���
std::map<int, int> label_map{
   {0, 1}, {1, 2} };
std::map<int, cv::Scalar> color_map{
   {1, {0, 0, 255}}, {2, {255, 0, 0}} };

// ------------------------ ��YoloDetector�������� --------------------------
/*
  ��������XLPDetector()
  ���ܣ���ʼ��XLPDetectorȫ�ֿ��Ʋ���
  ��ڲ�������
  ���ڲ�������
  ����ֵ����
*/
XLPDetector::XLPDetector()
{
	confThreshold = 0.5f;
	nmsThreshold = 0.4f;
	inpWidth = 416;
	inpHeight = 416;
}

/*
  ��������XLPDetector(float confThres, float nmsThres, bool use_dist, int Width, int Height)
  ���ܣ���ʼ��yolov3ȫ�ֿ��Ʋ���
  ��ڲ�����confThres: ���Ŷ���ֵ��Confidence threshold��
			nmsThres: �Ǽ���ֵ������ֵ��Non-maximum suppression threshold��
			Width:  �����������ͼ��Ŀ��320��416��608
			Height: �����������ͼ��ĸ߶�320��416��608
  ���ڲ�������
  ����ֵ����
*/
XLPDetector::XLPDetector(float confThres, float nmsThres, int Width, int Height)
{
	confThreshold = confThres;
	nmsThreshold = nmsThres;
	inpWidth = Width;
	inpHeight = Height;
}

/*
  ��������parameter_check()
  ���ܣ��������飬ʹ�ڹ涨��Χ��
  ��ڲ�������
  ���ڲ�������
  ����ֵ��check_flag: 0 ������-1����
*/
int XLPDetector::parameter_check()
{
	int check_flag = 0;
	// ��������
	if (confThreshold <= 0.0f || confThreshold > 1.0f)
	{
		cout << "Usage Error: parameters 'confThreshold' in class: YoloDetector.";
		check_flag = -1;
	}
	if (nmsThreshold <= 0.0f || nmsThreshold > 1.0f)
	{
		cout << "Usage Error: parameters 'nmsThreshold' in class: YoloDetector.";
		check_flag = -1;
	}
	if (inpWidth != 320 && inpWidth != 416 && inpWidth != 608)
	{
		cout << "Usage Error: parameters 'DinpWidth' in class: YoloDetector.";
		check_flag = -1;
	}
	if (inpHeight != 320 && inpHeight != 416 && inpHeight != 608)
	{
		cout << "Usage Error: parameters 'DinpWidth' in class: YoloDetector.";
		check_flag = -1;
	}
	return check_flag;
}

void XLPDetector::show_member()
{
	cout << "\n==> All private members of XLPDetector are shown as following: \n";
	cout << "confThreshold:  " << confThreshold << endl
		<< "nmsThreshold:  " << nmsThreshold << endl
		<< "inpWidth:  " << inpWidth << endl
		<< "inpHeight:  " << inpHeight << endl;
	cout << "==> Show Finish." << endl << endl;
}

/*
  ��������InitXLPDetNet
  ���ܣ���ʼ�����ӿڲ�����ģ���ļ�����������
  ��ڲ�����modelWeight: onnxģ��Ȩ���ļ�·��
  ���ڲ�������
  ����ֵ��----------------------------
		  0----��ʼ���ɹ�
		 -1----ģ�ͼ���ʧ��
 */
int XLPDetector::InitXLPDetNet(string modelWeight)
{
	// ��������ģ��Ȩ���ļ���weight file��
	// ��������
	try
	{
		XLPDetNet = cv::dnn::readNetFromONNX(modelWeight);
	}
	catch (cv::Exception& e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		if (XLPDetNet.empty())
		{
			std::cerr << "Can't load XLPDet network by using the provided weights." << std::endl;
			return (-1);
		}
	}
	XLPDetNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // ʹ��opencv
	XLPDetNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);	// ʹ��CPU
	OutputsNames = getOutputsNames();  // ��ȡ��������������
	return 0;
}

void XLPDetector::InitNMS(float ratio, int obj_n)
{
	gaussNMS = GaussianNMS(ratio, nmsThreshold, obj_n);
}


// �������������
vector<string> XLPDetector::getOutputsNames()
{
	vector<string> names;

	if (names.empty())
	{
		// Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = XLPDetNet.getUnconnectedOutLayers();
		// get the names of all the layers in the network
		std::vector<string> layersNames = XLPDetNet.getLayerNames();
		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}


void XLPDetector::PreProcess(Mat& frame, Mat& out_frame, bool use_letterbox)
{
	if (use_letterbox)
	{
		letterbox_image(frame, out_frame, inpWidth, inpHeight, Scalar(0, 0, 0));
	}
	else
	{
		resize(frame, out_frame, Size(inpWidth, inpHeight));  // Size(w, h)
	}
}

/*
  ��������XLPDet Inference
  ���ܣ�Yoloǰ���������
  ��ڲ�����Image: ����ͼ
  ���ڲ�����bbox: �������
  ����ֵ�� 0----���ɹ�
		  -1----�ӿ�δ��ʼ��
		  -2----����ͼ��Ϊ��
 */
int XLPDetector::Inference(Mat& Image, vector<detData>& resDataVec)
{
	if (Image.empty()) {
		cout << "Inference Error: image empty.\n";
		return -2;
	}

	// ��ͼ���д���һ�� 4D blob
	if (XLPDetNet.empty()) {
		cout << "Inference Error: Net empty.\n";
		return -1;
	}
	Mat blob;
	cv::dnn::blobFromImage(Image, blob, 2.0 / 255.0, cv::Size(inpWidth, inpHeight),
		cv::Scalar(127.5, 127.5, 127.5), true, false); // value -> (-1.0, 1.0), need swapRB

	// ��blob����Ϊ��������
	XLPDetNet.setInput(blob);

	// ����ǰ���䣬������������
	Mat out;
	clock_t start_time = clock();
	XLPDetNet.forward(out, OutputsNames[0]);
	clock_t end_time = clock();
	double t_forward = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;

	start_time = clock();
	Mat detMat(out.size[1], out.size[2], CV_32F, out.ptr());  // size(2704, 12) -> score: 4 + corners: 8
	Mat scoreMat = detMat.colRange(0, 4).clone();
	Mat cornerMat = detMat.colRange(4, 12).clone();
	Mat mm = cornerMat.rowRange(2, 3).clone();
	resDataVec = PostProcess(scoreMat, cornerMat);
	end_time = clock();
	double t_post = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;

	// ���һЩ������Ϣ
	// Put efficiency information. The function getPerfProfile returns the
	// overall time for inference(t) and the timings for each of the layers(in layersTimes)
	// vector<double> layersTimes;
	// double freq = cv::getTickFrequency() / 1000;
	// double t = XLPDetNet.getPerfProfile(layersTimes) / freq;
	string time_info = cv::format("Inference time: %.2f ms", t_forward);
	cout << time_info << endl;

	string time_info2 = cv::format("Post process time: %.2f ms", t_post);
	cout << time_info2 << endl;
	// cv::putText(Image, time_info, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2.0);
	return 0;
}

vector<detData> XLPDetector::PostProcess(Mat& scoreMat, Mat& cornerMat)
{
	vector<detData> detDataVec;
	int N = scoreMat.rows;
	for (int i = 0; i < N; i++) {
		float score = 0;
		for (int j = 0; j < 4; j++) {
			score += scoreMat.at<float>(i, j);
		}
		score /= 4.0;
		if (score < confThreshold) continue;
		detData tmpData;
		tmpData.score = score;
		for (int j = 0; j < 8; j++) {
			tmpData.corners[j] = cornerMat.at<float>(i, j);
		}
		detDataVec.push_back(tmpData);
	}
	// int M = detDataVec.size();
	//for (int i = 0; i < M; i++) {
	//	cout << "Score: " << detDataVec[i].score << " -- " << detDataVec[i].corners[0] << endl;
	//}
	detDataVec = gaussNMS.sortAndSelect(detDataVec);
	vector<detData> keepDataVec = gaussNMS.nmsProcess(detDataVec);
	cout << "Last NMS Result:\n";
	cout << keepDataVec;
	return keepDataVec;
}

