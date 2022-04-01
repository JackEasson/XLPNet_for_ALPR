#pragma once
#ifndef _GAUSS_H_
#define _GAUSS_H_
#include "base.h"

RotatedRect get_min_bounding_rect(vector<float> corners);

// Rotated 2D Gaussian Distribution
class R2dGaussianDistribution
{
private:
	float m_x, m_y, m_w, m_h, m_theta, m_ratio;
	float PI = 3.141593f;
	Mat sigma_i;  // sigma_inverse
	void getSigmaInverse();
public:
	R2dGaussianDistribution(): m_x(0.), m_y(0.), m_w(0.), m_h(0.), m_theta(0.), m_ratio(2.) { };
	R2dGaussianDistribution(float x, float y, float w, float h, float theta, float ratio): m_x(x), m_y(y), m_w(w), m_h(h), m_theta(theta), m_ratio(ratio) { getSigmaInverse(); };
	R2dGaussianDistribution(RotatedRect rect);
	R2dGaussianDistribution(RotatedRect rect, float ratio);
	~R2dGaussianDistribution() { cout << "R2dGaussianDistribution Destructor Called" << endl; }
	void Reset(float x, float y);
	void Reset(RotatedRect rect);
	vector<float> getGaussianScores(vector<float> x, vector<float> y);
};

class GaussianNMS
{
private:
	float g_ratio;
	float nms_thres;
	int max_obj;
	R2dGaussianDistribution gaussDistribution = R2dGaussianDistribution();
public:
	GaussianNMS() : g_ratio(2.0), nms_thres(0.2), max_obj(20) {};
	GaussianNMS(float r, float thres, int n) : g_ratio(r), nms_thres(thres), max_obj(n) {};
	~GaussianNMS() { cout << "GaussianNMS Destructor Called" << endl; }
	void showMembers();
	int maxObjs() { return max_obj; };
	vector<detData> sortAndSelect(vector<detData> primary_res);
	vector<float> gaussianScoreGenerator(float c_x, float c_y, vector<float> x, vector<float> y);
	vector<detData> nmsProcess(vector<detData> medium_res);
};

#endif