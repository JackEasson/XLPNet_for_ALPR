#include "gauss_tools.h"

/*
class CV_EXPORTS RotatedRect
{
public:
    //! various constructors
    RotatedRect();
    RotatedRect(const Point2f& center, const Size2f& size, float angle);
    RotatedRect(const CvBox2D& box);

    //! returns 4 vertices of the rectangle
    void points(Point2f pts[]) const;
    //! returns the minimal up-right rectangle containing the rotated rectangle
    Rect boundingRect() const;
    //! conversion to the old-style CvBox2D structure
    operator CvBox2D() const;

    Point2f center; //< the rectangle mass center
    Size2f size;    //< width and height of the rectangle
    float angle;    //< the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
};
*/
RotatedRect get_min_bounding_rect(vector<float> corners)
{
	vector<Point> ployPoints{ Point(corners[0], corners[1]),Point(corners[2], corners[3]),Point(corners[4], corners[5]),Point(corners[6], corners[7]) };
	RotatedRect rotated_rect = minAreaRect(ployPoints);
	return rotated_rect;
}


R2dGaussianDistribution::R2dGaussianDistribution(RotatedRect rect)
{
    Point2f m_center = rect.center;
    Size2f m_size = rect.size;
    float m_angle = rect.angle;
    m_x = m_center.x;
    m_y = m_center.y;
    m_w = m_size.width;
    m_h = m_size.height;
    m_theta = m_angle / 180 * PI;
    m_ratio = 1.0;
    getSigmaInverse();
}

R2dGaussianDistribution::R2dGaussianDistribution(RotatedRect rect, float ratio)
{
    Point2f m_center = rect.center;
    Size2f m_size = rect.size;
    float m_angle = rect.angle;
    m_x = m_center.x;
    m_y = m_center.y;
    m_w = m_size.width;
    m_h = m_size.height;
    m_theta = m_angle / 180 * PI;
    m_ratio = ratio;
    getSigmaInverse();
}

void R2dGaussianDistribution::getSigmaInverse()
{
    float sigma_11 = m_w * m_ratio, sigma_22 = m_h * m_ratio, sigma_12 = 0.0, sigma_21 = 0.0;
    Mat sigma_mat = (Mat_<float>(2, 2) << sigma_11, sigma_12, sigma_21, sigma_22);
    Mat rotated_mat = (Mat_<float>(2, 2) << cos(m_theta), -sin(m_theta), sin(m_theta), cos(m_theta));
    sigma_mat = rotated_mat * sigma_mat * rotated_mat.t();
    sigma_i = sigma_mat.inv();
}


void R2dGaussianDistribution::Reset(float x, float y)
{
    m_x = x; m_y = y;
}

void R2dGaussianDistribution::Reset(RotatedRect rect)
{
    Point2f m_center = rect.center;
    Size2f m_size = rect.size;
    float m_angle = rect.angle;
    m_x = m_center.x;
    m_y = m_center.y;
    m_w = m_size.width;
    m_h = m_size.height;
    m_theta = m_angle / 180 * PI;
    getSigmaInverse();
}

vector<float> R2dGaussianDistribution::getGaussianScores(vector<float> vx, vector<float> vy)
{
    Mat MatX = Mat(vx).t(), MatY = Mat(vy).t();  // size(1, n)
    Mat errorX = MatX - m_x, errorY = MatY - m_y;
    Mat powX, powY;
    pow(errorX, 2, powX);
    pow(errorY, 2, powY);
    Mat errorIndex = powX * sigma_i.at<float>(0, 0) + errorX.mul(errorY) * (sigma_i.at<float>(0, 1) + sigma_i.at<float>(1, 0)) + powY * sigma_i.at<float>(1, 1);
    Mat scoreMat;
    exp(-0.5 * errorIndex, scoreMat);
    // cout << "scoreMat " << scoreMat << endl;
    vector<float> res = (vector<float>)(scoreMat);
    // vector<float> res = vector<float>();
    return res;
}


// ********************************* GaussianNMS *******************************
void GaussianNMS::showMembers()
{
    cout << "\n==> All private members of GaussianNMS are shown as following: \n";
    cout << "g_ratio:  " << g_ratio << endl
        << "nms_thres:  " << nms_thres << endl
        << "max_obj:  " << max_obj << endl;
    cout << "==> Show Finish." << endl << endl;
}



vector<detData> GaussianNMS::sortAndSelect(vector<detData> primary_res)
{
    sort(primary_res.begin(), primary_res.end(), compareDetData);
    int n = max_obj;
    if (primary_res.size() > n) {
        primary_res = vector<detData>(primary_res.begin(), primary_res.begin() + n);
    }
    return primary_res;
}

vector<float> GaussianNMS::gaussianScoreGenerator(float c_x, float c_y, vector<float> v_x, vector<float> v_y)
{
    gaussDistribution.Reset(c_x, c_y);
    vector<float> scoreVec = gaussDistribution.getGaussianScores(v_x, v_y);
    return scoreVec;
}

vector<detData> GaussianNMS::nmsProcess(vector<detData> medium_res)
{
    int num = medium_res.size();
    vector<int> order;
    vector<detData> keep_res;
    for (int i = 0; i < num; i++) {
        order.push_back(i);
    }
    while (order.size() > 0) {
        if (order.size() == 1) {
            keep_res.push_back(medium_res[order[0]]);
            break;
        }
        // else
        keep_res.push_back(medium_res[order[0]]);
        // 当前最小外接矩形先得到
        RotatedRect r_rect = get_min_bounding_rect(medium_res[order[0]].corners);
        gaussDistribution.Reset(r_rect);
        vector<vector<float>> scoreVec2d;
        // 4组角点分别求高斯得分
        for (int i = 0; i < 4; i++) {
            int cur_x = medium_res[order[0]].corners[i * 2], cur_y = medium_res[order[0]].corners[i * 2 + 1];
            vector<float> vx, vy;
            for (int j = 1; j < order.size(); j++) {
                vx.push_back(medium_res[order[j]].corners[i * 2]);
                vy.push_back(medium_res[order[j]].corners[i * 2 + 1]);
            }
            vector<float> scoreVec = gaussianScoreGenerator(cur_x, cur_y, vx, vy);
            scoreVec2d.push_back(scoreVec);
        }
        vector<float> meanScoreVec(order.size() - 1, 0.0);
        for (int i = 0; i < scoreVec2d.size(); i++) {
            for (int j = 0; j < scoreVec2d[i].size(); j++) {
                meanScoreVec[j] += scoreVec2d[i][j] / 4.0;
            }
        }
        vector<int> new_order;
        for (int i = 0; i < meanScoreVec.size(); i++) {
            if (meanScoreVec[i] < nms_thres) {
                new_order.push_back(order[1 + i]);
            }
        }
        order = new_order;
    }
    return keep_res;
}

