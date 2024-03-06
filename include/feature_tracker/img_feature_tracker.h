#ifndef __IMG_FEATURE_TRACKER_H
#define __IMG_FEATURE_TRACKER_H

#include "common_include.h"

// class Feature
// {
//     public:
//     cv::Point2f 

// }

// class Feature
// {
//     public:
    
//     unsigned long _feature_id;
    
// }

class FeatureTracker
{
private:
public:
    typedef std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>> FeatureInCam; // cam_id  motion
    typedef std::map<int, FeatureInCam> FeatureInImg;                              // feature_id

    double _cur_time;
    double _prev_time;
    cv::Mat _cur_img; // left image
    cv::Mat _prev_img;
    cv::Mat _cur_img1; // right image
    cv::Mat _mask;

    cv::Mat k;
    cv::Mat dist;

    std::vector<cv::Point2f> _cur_pts, _cur_right_pts;
    std::vector<cv::Point2f> _prev_pts;
    std::vector<cv::Point2f> _cur_unpts, _cur_right_unpts;
    std::map<int, cv::Point2f> _cur_id_unpts_map, _prev_id_unpts_map, _cur_id_right_unpts_map, _prev_id_right_unpts_map;
    std::vector<cv::Point2f> _pts_velocity, _right_pts_velocity;

    int rows;
    int cols;

    unsigned long _feature_id_generator;
    std::vector<int> _feature_ids, _feature_right_ids; // 记录特征点的id
    std::vector<int> _track_cnt;                       // 记录特征点被跟踪的次数

    FeatureTracker();
    FeatureInImg trackImage(double cur_time, const cv::Mat &img, const cv::Mat &img1 = cv::Mat());
    bool inBorder(const cv::Point2f &pt);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void setMask();
    std::vector<cv::Point2f> ptsVelocity(std::vector<int> &ids, std::vector<cv::Point2f> &pts,
                                         std::map<int, cv::Point2f> &cur_id_pts, std::map<int, cv::Point2f> &prev_id_pts);

    template <typename T>
    void reduceVector(std::vector<T> &v, std::vector<uchar> &status)
    {
        int j = 0;
        for (size_t i = 0; i < v.size(); i++)
        {
            if (status[i]) // 如果是1就留下
            {
                v[j++] = v[i];
            }
        }
        v.resize(j);
    }
};
#endif // __IMG_FEATURE_TRACKER_H