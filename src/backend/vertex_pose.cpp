/*
 * @Author: HouShikang
 * @Date: 2024-03-13 19:19:15
 * @Description:
 */
#include "vertex_pose.h"
#include "sophus/geometry.hpp"
#include <iostream>
namespace backend
{
	// using namespace Sophus;
	// px py pz qw qx qy qz
	void VertexPose::plus(const VecX& delta)
	{
	
		parameters_[0] += delta[0];
		parameters_[1] += delta[1];
		parameters_[2] += delta[2];

		Qd q(parameters_[3], parameters_[4], parameters_[5], parameters_[6]);
		q = q *
			Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();
		q.normalized();

		parameters_[3] = q.w();
		parameters_[4] = q.x();
		parameters_[5] = q.y();
		parameters_[6] = q.z();

	}
} // namespace backend