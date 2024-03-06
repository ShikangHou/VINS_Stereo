#include "img_feature_tracker.h"
#include "parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/fast_math.hpp>

FeatureTracker::FeatureTracker()
{
    _feature_id_generator = 0;
    k = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    dist = cv::Mat::zeros(5, 1, CV_64F);
};

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// check if the point is in border
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    // round 四舍五入　　ceil 向上取整　　floor　向下取整
    int img_x = std::round(pt.x); // 四舍五入
    int img_y = std::round(pt.y);
    if (img_x >= BORDER_SIZE && img_x <= cols - BORDER_SIZE && img_y >= BORDER_SIZE && img_y <= rows - BORDER_SIZE)
    {
        return false;
    }
    return true;
}

void FeatureTracker::setMask()
{
    _mask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    // 特征点按照跟踪次数从大到小排序,优先保留次数多的特征点
    std::vector<std::pair<int, std::pair<int, cv::Point2f>>> cnt_id_pts;
    for (size_t i = 0; i < _cur_pts.size(); i++)
    {
        cnt_id_pts.emplace_back(make_pair(_track_cnt[i], make_pair(_feature_ids[i], _cur_pts[i])));
    }

    sort(cnt_id_pts.begin(), cnt_id_pts.end(), [](pair<int, pair<int, cv::Point2f>> &a, pair<int, pair<int, cv::Point2f>> &b)
         { return a.first > b.first; });

    _track_cnt.clear();
    _feature_ids.clear();
    _cur_pts.clear();

    for (auto &it : cnt_id_pts)
    {
        if (_mask.at<uchar>(it.second.second) == 255)
        {
            _track_cnt.push_back(it.first);
            _feature_ids.push_back(it.second.first);
            _cur_pts.push_back(it.second.second);
            // 制作mask，使老特征点的MIN_DIST的范围内不出现新的特征点
            cv::circle(_mask, it.second.second, MIN_DIST, cv::Scalar(0), -1);
        }
    }
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> velocitys;
    cur_id_pts.clear();
    for (size_t i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    if (!prev_id_pts.empty())
    {
        double dt = _cur_time - _prev_time;
        for (size_t i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double vx = (it->second.x - pts[i].x) / dt;
                double vy = (it->second.y - pts[i].y) / dt;
                velocitys.push_back(cv::Point2f(vx, vy));
            }
            else
            {
                velocitys.push_back(cv::Point2f(0.0, 0.0));
            }
        }
    }
    else
    {
        for (size_t i = 0; i < pts.size(); i++)
        {
            velocitys.push_back(cv::Point2f(0.0, 0.0));
        }
    }
    return velocitys;
}

FeatureTracker::FeatureInImg FeatureTracker::trackImage(double cur_time, const cv::Mat &img, const cv::Mat &img1)
{
    FeatureInImg feature_frame;
    _cur_img = img;
    rows = _cur_img.rows;
    cols = _cur_img.cols;
    _cur_time = cur_time;
    
    //　处理img 左目
    {
        _cur_pts.clear();

        if (_prev_pts.size() > 0) // 如果有前一帧
        {
            std::vector<uchar> status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(_prev_img, _cur_img, _prev_pts, _cur_pts, status, error, cv::Size(21, 21), 3,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

            // reserve check
            if (FLOW_BACK)
            {
                std::vector<uchar> reverse_status;
                std::vector<cv::Point2f> reverse_pts = _prev_pts; // 提供一个初始参考
                cv::calcOpticalFlowPyrLK(_cur_img, _prev_img, _cur_pts, reverse_pts, reverse_status, error, cv::Size(21, 21), 3,
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i] && reverse_status[i] && distance(_prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                    {
                        status[i] = 0;
                    }
                }
            }

            for (size_t i = 0; i < _cur_pts.size(); i++)
            {
                if (status[i] == 1 && inBorder(_cur_pts[i]))
                {
                    status[i] = 0;
                }
            }

            // 剩下的为能够被追踪的特征点
            reduceVector<cv::Point2f>(_cur_pts, status);
            reduceVector<cv::Point2f>(_prev_pts, status);
            reduceVector<int>(_feature_ids, status);
            reduceVector<int>(_track_cnt, status);
        }

        for (auto &n : _track_cnt)
        {
            n++;
        }

        // 维持特征点数量
        {
            setMask();
            int f_num = MAX_NUM - _cur_pts.size();
            if (f_num > 0)
            {
                if (_mask.empty())
                    cout << "mask is empty" << endl;

                vector<cv::Point2f> f_pts;
                cv::goodFeaturesToTrack(_cur_img, f_pts, f_num, 0.01, MIN_DIST, _mask); // MIN_DIST，新特征点之间的最小距离

                for (auto &pt : f_pts)
                {
                    _cur_pts.push_back(pt);
                    _feature_ids.push_back(_feature_id_generator++);
                    _track_cnt.push_back(1);
                }
            }
        }

        cv::undistortPoints(_cur_pts, _cur_unpts, k, dist);
        _pts_velocity = ptsVelocity(_feature_ids, _cur_unpts, _cur_id_unpts_map, _prev_id_unpts_map);

        for (size_t i = 0; i < _feature_ids.size(); i++)
        {
            int id = _feature_ids[i];
            double x, y, z;
            x = _cur_unpts[i].x;
            y = _cur_unpts[i].y;
            z = 1;

            double u, v;
            u = _cur_pts[i].x;
            v = _cur_pts[i].y;
            int camera_id = 0; // left_camera
            double vx, vy;
            vx = _pts_velocity[i].x;
            vy = _pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, u, v, vx, vy;
            feature_frame[id].emplace_back(camera_id, xyz_uv_velocity);
        }

        _prev_img = _cur_img;
        _prev_pts = _cur_pts;
        _prev_id_unpts_map = _cur_id_unpts_map;
    }

    // 处理img1
    if (!img1.empty())
    {
        _cur_img1 = img1;
        _feature_right_ids = _feature_ids; // 如果所有特征点都能跟踪到，左右特征点的id应该一致
        _cur_right_pts.clear();
        _cur_right_unpts.clear();
        _cur_id_right_unpts_map.clear();

        vector<uchar> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(_cur_img, _cur_img1, _cur_pts, _cur_right_pts, status, error, cv::Size(21, 21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

        if (FLOW_BACK)
        {
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_pts = _cur_pts; // 提供一个初始参考
            cv::calcOpticalFlowPyrLK(_cur_img1, _cur_img, _cur_right_pts, reverse_pts, reverse_status, error, cv::Size(21, 21), 3,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i] && reverse_status[i] && distance(_prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                {
                    status[i] = 0;
                }
            }
        }

        for (size_t i = 0; i < _cur_right_pts.size(); i++)
        {
            if (status[i] == 1 && inBorder(_cur_right_pts[i]))
            {
                status[i] = 0;
            }
        }
        reduceVector<int>(_feature_right_ids, status);
        reduceVector<cv::Point2f>(_cur_right_pts, status);

        cv::undistortPoints(_cur_right_pts, _cur_right_unpts, k, dist);
        _right_pts_velocity = ptsVelocity(_feature_right_ids, _cur_right_unpts, _cur_id_right_unpts_map, _prev_id_right_unpts_map);

        for (size_t i = 0; i < _feature_right_ids.size(); i++)
        {
            int id = _feature_ids[i];
            double x, y, z;
            x = _cur_unpts[i].x;
            y = _cur_unpts[i].y;
            z = 1;

            double u, v;
            u = _cur_pts[i].x;
            v = _cur_pts[i].y;
            int camera_id = 1; // right_camera
            double vx, vy;
            vx = _pts_velocity[i].x;
            vy = _pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, u, v, vx, vy;
            feature_frame[id].emplace_back(camera_id, xyz_uv_velocity);
        }

        _prev_id_right_unpts_map = _cur_id_right_unpts_map;
    }

    _prev_time = _cur_time;

    return feature_frame;
}
