#include "feature_tracker/feature_tracker.h"
#include "parameters.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <queue>
#include <thread>

std::queue<sensor_msgs::ImageConstPtr> img0_buf;
std::queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;
ros::Publisher pub1, pub2;

FeatureTracker feature_tracker;

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
	cv_bridge::CvImageConstPtr ptr;
	if (img_msg->encoding == "8UC1")
	{
		sensor_msgs::Image img;
		img.header = img_msg->header;
		img.height = img_msg->height;
		img.width = img_msg->width;
		img.is_bigendian = img_msg->is_bigendian;
		img.step = img_msg->step;
		img.data = img_msg->data;
		img.encoding = "mono8";
		ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
	}
	else
		ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

	cv::Mat img = ptr->image.clone();
	return img;
}

void SyncImageMsgs()
{
	while (ros::ok())
	{
		cv::Mat img0, img1;
		double time;
		if (STEREO)
		{
			std::unique_lock<std::mutex> imgbuf_lock(m_buf);
			if (!img0_buf.empty() && !img1_buf.empty())
			{
				double time0 = img0_buf.front()->header.stamp.toSec();
				double time1 = img1_buf.front()->header.stamp.toSec();

				if (time0 < time1 - 0.003) // img0太早
				{
					img0_buf.pop();
				}
				else if (time0 > time1 + 0.003) // img0太晚
				{
					img1_buf.pop();
				}
				else // 时差在0.003s内，认为同步
				{
					time = time0;
					img0 = getImageFromMsg(img0_buf.front());
					img0_buf.pop();
					img1 = getImageFromMsg(img1_buf.front());
					img1_buf.pop();
				}
			}
			imgbuf_lock.unlock();

			if (!img0.empty())
			{
				FeatureTracker::FeatureInImg fi = feature_tracker.trackImage(time, img0, img1);
//				for (auto &f : fi)
//				{
//					cv::circle(img0, cv::Point2f(f.second[0].second[3], f.second[0].second[4]), 2,
//							   cv::Scalar(0, 255, 0), -1);
//					if (f.second.size() > 1)
//					{
//						cv::circle(img1, cv::Point2f(f.second[1].second[3], f.second[1].second[4]), 2,
//								   cv::Scalar(0, 255, 0), -1);
//					}
//				}
//				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img0).toImageMsg();
//				pub1.publish(msg);
//				msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img1).toImageMsg();
//				pub2.publish(msg);
			}
		}

		else
		{
			std::unique_lock<std::mutex> imgbuf_lock(m_buf);
			if (!img0_buf.empty())
			{
				time = img0_buf.front()->header.stamp.toSec();
				img0 = getImageFromMsg(img0_buf.front());
				img0_buf.pop();
			}
			imgbuf_lock.unlock();
			if (!img0.empty())
			{
				FeatureTracker::FeatureInImg fi = feature_tracker.trackImage(time, img0, img1);
//				for (auto &f : fi)
//				{
//					cv::circle(img0, cv::Point2f(f.second[0].second[3], f.second[0].second[4]), 2,
//							   cv::Scalar(0, 255, 0), -1);
//				}
//				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img0).toImageMsg();
//				pub1.publish(msg);
			}
		}

		std::chrono::milliseconds dura(2);
		std::this_thread::sleep_for(dura);
	}
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
	std::unique_lock<std::mutex> imgbuf_lock(m_buf);
	img0_buf.push(img_msg);
	imgbuf_lock.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
	std::unique_lock<std::mutex> imgbuf_lock(m_buf);
	img1_buf.push(img_msg);
	imgbuf_lock.unlock();
}

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;

	setParameters();

	ros::Subscriber sub_img0 = nh.subscribe("/cam0/image_raw", 100, img0_callback);
	ros::Subscriber sub_img1 = nh.subscribe("/cam1/image_raw", 100, img1_callback);
	pub1 = nh.advertise<sensor_msgs::Image>("/cam0/image_out", 10);
	pub2 = nh.advertise<sensor_msgs::Image>("/cam1/image_out", 10);

	std::thread thread_sync(SyncImageMsgs);
	ros::spin();
	thread_sync.join();

	return 0;
}
