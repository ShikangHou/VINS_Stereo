#include "feature_tracker.h"
#include "parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/fast_math.hpp>

FeatureTracker::FeatureTracker()
{
	feature_id_generator_ = 0;
	k = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	dist = cv::Mat::zeros(5, 1, CV_64F);
};

double FeatureTracker::distance(cv::Point2f& pt1, cv::Point2f& pt2)
{
	double dx = pt1.x - pt2.x;
	double dy = pt1.y - pt2.y;
	return sqrt(dx * dx + dy * dy);
}

// check if the point is in border
bool FeatureTracker::inBorder(const cv::Point2f& pt) const
{
	// round 四舍五入　　ceil 向上取整　　floor　向下取整
	int img_x = std::round(pt.x); // 四舍五入
	int img_y = std::round(pt.y);
	if (img_x >= BORDER_SIZE && img_x <= cols_ - BORDER_SIZE && img_y >= BORDER_SIZE && img_y <= rows_ - BORDER_SIZE)
	{
		return false;
	}
	return true;
}

void FeatureTracker::setMask()
{
	mask_ = cv::Mat(rows_, cols_, CV_8UC1, cv::Scalar(255));
	// 特征点按照跟踪次数从大到小排序,优先保留次数多的特征点
	std::vector<std::pair<int, std::pair<int, cv::Point2f>>> cnt_id_pts;
	for (size_t i = 0; i < cur_pts_.size(); i++)
	{
		cnt_id_pts.emplace_back(track_cnt_[i], make_pair(feature_ids_[i], cur_pts_[i]));
	}

	sort(cnt_id_pts.begin(), cnt_id_pts.end(),
		[](pair<int, pair<int, cv::Point2f>>& a, pair<int, pair<int, cv::Point2f>>& b)
		{ return a.first > b.first; });

	track_cnt_.clear();
	feature_ids_.clear();
	cur_pts_.clear();

	for (auto& it : cnt_id_pts)
	{
		if (mask_.at<uchar>(it.second.second) == 255)
		{
			track_cnt_.push_back(it.first);
			feature_ids_.push_back(it.second.first);
			cur_pts_.push_back(it.second.second);
			// 制作mask，使老特征点的MIN_DIST的范围内不出现新的特征点
			cv::circle(mask_, it.second.second, MIN_DIST, cv::Scalar(0), -1);
		}
	}
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<unsigned long>& ids, vector<cv::Point2f>& pts,
	map<unsigned long, cv::Point2f>& cur_id_pts, map<unsigned long, cv::Point2f>& prev_id_pts) const
{
	vector<cv::Point2f> velocitys;
	cur_id_pts.clear();
	for (size_t i = 0; i < ids.size(); i++)
	{
		cur_id_pts.insert(make_pair(ids[i], pts[i]));
	}

	if (!prev_id_pts.empty())
	{
		double dt = cur_time_ - prev_time_;
		for (size_t i = 0; i < pts.size(); i++)
		{
			std::map<unsigned long, cv::Point2f>::iterator it;
			it = prev_id_pts.find(ids[i]);
			if (it != prev_id_pts.end())
			{
				double vx = (it->second.x - pts[i].x) / dt;
				double vy = (it->second.y - pts[i].y) / dt;
				velocitys.emplace_back(vx, vy);
			}
			else
			{
				velocitys.emplace_back(0.0, 0.0);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < pts.size(); i++)
		{
			velocitys.emplace_back(0.0, 0.0);
		}
	}
	return velocitys;
}

FeatureTracker::FeatureInImg FeatureTracker::trackImage(double cur_time, const cv::Mat& img, const cv::Mat& img1)
{
	FeatureInImg feature_frame;
	cur_img_ = img;
	rows_ = cur_img_.rows;
	cols_ = cur_img_.cols;
	cur_time_ = cur_time;

	//　处理img 左目
	{
		cur_pts_.clear();

		if (!prev_pts_.empty()) // 如果有前一帧
		{
			std::vector<uchar> status;
			std::vector<float> error;
			cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_, status, error, cv::Size(21, 21), 3,
				cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

			// reserve check
			if (FLOW_BACK)
			{
				std::vector<uchar> reverse_status;
				std::vector<cv::Point2f> reverse_pts = prev_pts_; // 提供一个初始参考
				cv::calcOpticalFlowPyrLK(cur_img_, prev_img_, cur_pts_, reverse_pts, reverse_status, error,
					cv::Size(21, 21), 3,
					cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
					cv::OPTFLOW_USE_INITIAL_FLOW);
				for (size_t i = 0; i < status.size(); i++)
				{
					if (status[i] && reverse_status[i] && distance(prev_pts_[i], reverse_pts[i]) <= 0.5)
					{
						status[i] = 1;
					}
					else
					{
						status[i] = 0;
					}
				}
			}

			for (size_t i = 0; i < cur_pts_.size(); i++)
			{
				if (status[i] == 1 && inBorder(cur_pts_[i]))
				{
					status[i] = 0;
				}
			}

			// 剩下的为能够被追踪的特征点
			reduceVector<cv::Point2f>(cur_pts_, status);
			reduceVector<cv::Point2f>(prev_pts_, status);
			reduceVector<unsigned long>(feature_ids_, status);
			reduceVector<int>(track_cnt_, status);
		}

		for (auto& n : track_cnt_)
		{
			n++;
		}

		// 维持特征点数量
		{
			setMask();
			int f_num = MAX_NUM - cur_pts_.size();
			if (f_num > 0)
			{
				if (mask_.empty())
					cout << "mask is empty" << endl;

				vector<cv::Point2f> f_pts;
				cv::goodFeaturesToTrack(cur_img_, f_pts, f_num, 0.01, MIN_DIST, mask_); // MIN_DIST，新特征点之间的最小距离

				for (auto& pt : f_pts)
				{
					cur_pts_.push_back(pt);
					feature_ids_.push_back(feature_id_generator_++);
					track_cnt_.push_back(1);
				}
			}
		}

		cv::undistortPoints(cur_pts_, cur_unpts_, k, dist);
		pts_velocity_ = ptsVelocity(feature_ids_, cur_unpts_, cur_id_unpts_map_, prev_id_unpts_map_);

		for (size_t i = 0; i < feature_ids_.size(); i++)
		{
			int id = feature_ids_[i];
			double x, y, z;
			x = cur_unpts_[i].x;
			y = cur_unpts_[i].y;
			z = 1;

			double u, v;
			u = cur_pts_[i].x;
			v = cur_pts_[i].y;
			int camera_id = 0; // left_camera
			double vx, vy;
			vx = pts_velocity_[i].x;
			vy = pts_velocity_[i].y;

			Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
			xyz_uv_velocity << x, y, z, u, v, vx, vy;
			feature_frame[id].emplace_back(camera_id, xyz_uv_velocity);
		}

		prev_img_ = cur_img_;
		prev_pts_ = cur_pts_;
		prev_id_unpts_map_ = cur_id_unpts_map_;
	}

	// 处理img1
	if (!img1.empty())
	{
		cur_img1_ = img1;
		feature_right_ids_ = feature_ids_; // 如果所有特征点都能跟踪到，左右特征点的id应该一致
		cur_right_pts_.clear();
		cur_right_unpts_.clear();
		cur_id_right_unpts_map_.clear();

		vector<uchar> status;
		vector<float> error;
		cv::calcOpticalFlowPyrLK(cur_img_, cur_img1_, cur_pts_, cur_right_pts_, status, error, cv::Size(21, 21), 3,
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

		if (FLOW_BACK)
		{
			std::vector<uchar> reverse_status;
			std::vector<cv::Point2f> reverse_pts = cur_pts_; // 提供一个初始参考
			cv::calcOpticalFlowPyrLK(cur_img1_, cur_img_, cur_right_pts_, reverse_pts, reverse_status, error,
				cv::Size(21, 21), 3,
				cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
				cv::OPTFLOW_USE_INITIAL_FLOW);
			for (size_t i = 0; i < status.size(); i++)
			{
				if (status[i] && reverse_status[i] && distance(prev_pts_[i], reverse_pts[i]) <= 0.5)
				{
					status[i] = 1;
				}
				else
				{
					status[i] = 0;
				}
			}
		}

		for (size_t i = 0; i < cur_right_pts_.size(); i++)
		{
			if (status[i] == 1 && inBorder(cur_right_pts_[i]))
			{
				status[i] = 0;
			}
		}
		reduceVector<unsigned long>(feature_right_ids_, status);
		reduceVector<cv::Point2f>(cur_right_pts_, status);

		cv::undistortPoints(cur_right_pts_, cur_right_unpts_, k, dist);
		right_pts_velocity_ =
			ptsVelocity(feature_right_ids_, cur_right_unpts_, cur_id_right_unpts_map_, prev_id_right_unpts_map_);

		for (size_t i = 0; i < feature_right_ids_.size(); i++)
		{
			int id = feature_ids_[i];
			double x, y, z;
			x = cur_unpts_[i].x;
			y = cur_unpts_[i].y;
			z = 1;

			double u, v;
			u = cur_pts_[i].x;
			v = cur_pts_[i].y;
			int camera_id = 1; // right_camera
			double vx, vy;
			vx = pts_velocity_[i].x;
			vy = pts_velocity_[i].y;

			Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
			xyz_uv_velocity << x, y, z, u, v, vx, vy;
			feature_frame[id].emplace_back(camera_id, xyz_uv_velocity);
		}

		prev_id_right_unpts_map_ = cur_id_right_unpts_map_;
	}

	prev_time_ = cur_time_;

	return feature_frame;
}
