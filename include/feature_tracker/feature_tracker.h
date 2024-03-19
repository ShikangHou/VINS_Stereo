#ifndef __FEATURE_TRACKER_H
#define __FEATURE_TRACKER_H

#include "common_include.h"

// class Feature
// {
//     public:
//     cv::Point2f

// }f

// class Feature
// {
//     public:

//     unsigned long _feature_id;

// }

class FeatureTracker
{
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
		FeatureInImg; // feature_id  cam_id  pts

	FeatureTracker();
	FeatureInImg trackImage(double cur_time, const cv::Mat& img, const cv::Mat& img1 = cv::Mat());
	bool inBorder(const cv::Point2f& pt) const;
	static double distance(cv::Point2f& pt1, cv::Point2f& pt2);
	void setMask();
	std::vector<cv::Point2f> ptsVelocity(std::vector<unsigned long>& ids, std::vector<cv::Point2f>& pts,
		std::map<unsigned long, cv::Point2f>& cur_id_pts, std::map<unsigned long, cv::Point2f>& prev_id_pts) const;

	template<typename T>
	void reduceVector(std::vector<T>& v, std::vector<uchar>& status)
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

 private:
	double cur_time_;
	double prev_time_;
	cv::Mat cur_img_; // left image
	cv::Mat prev_img_;
	cv::Mat cur_img1_; // right image
	cv::Mat mask_;

	cv::Mat k;
	cv::Mat dist;

	std::vector<cv::Point2f> cur_pts_, cur_right_pts_;     // 当前左帧特征点坐标　　当前右帧特征点坐标
	std::vector<cv::Point2f> prev_pts_;                    //  前一左帧
	std::vector<cv::Point2f> cur_unpts_, cur_right_unpts_; // 当前左帧特征点去畸变归一化坐标　　当前右帧特征点去畸变归一化坐标　　
	std::map<unsigned long, cv::Point2f> cur_id_unpts_map_, prev_id_unpts_map_, cur_id_right_unpts_map_, prev_id_right_unpts_map_;
	std::vector<cv::Point2f> pts_velocity_, right_pts_velocity_;

	unsigned long feature_id_generator_;
	std::vector<unsigned long> feature_ids_, feature_right_ids_; // 记录特征点的id
	std::vector<int> track_cnt_;                       // 记录特征点被跟踪的次数

	int rows_;
	int cols_;
};
#endif // __FEATURE_TRACKER_H