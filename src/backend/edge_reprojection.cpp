/*
 * @Author: HouShikang
 * @Date: 2024-03-13 20:58:28
 * @Description:
 */
#include <iostream>
#include "edge_reprojection.h"
#include "vertex.h"
#include "sophus/so3.hpp"

namespace backend
{

	void EdgeReprojection::computeResidual()
	{
		double inverse_depth_i = verticies_[0]->parameters()[0];

		Vec7 param_i = verticies_[1]->parameters();
		Qd Qi(param_i[3], param_i[4], param_i[5], param_i[6]);
		Vec3 Pi = param_i.head(3);

		Vec7 param_j = verticies_[2]->parameters();
		Qd Qj(param_j[3], param_j[4], param_j[5], param_j[6]);
		Vec3 Pj = param_j.head(3);

		Vec7 param_ext = verticies_[3]->parameters();;
		Qd qbc(param_ext[3], param_ext[4], param_ext[5], param_ext[6]);
		Vec3 tbc = param_ext.head(3);

		Vec3 pts_cam_i = pts_norm_i_ / inverse_depth_i;
		Vec3 pts_bi = qbc * pts_cam_i + tbc;
		Vec3 pts_w = Qi * pts_bi + Pi;
		Vec3 pts_bj = Qj.inverse() * (pts_w - Pj);
		Vec3 pts_cam_j = qbc.inverse() * (pts_bj - tbc);

		Vec3 pts_norm_j = pts_cam_j / pts_cam_j[2];
		residual_ = (pts_norm_j - pts_norm_j_).head(2);
//		std::cerr << qbc.toRotationMatrix() << std::endl;
	}

	void EdgeReprojection::computeJacobian()
	{
		double inverse_depth_i = verticies_[0]->parameters()[0];

		Vec7 param_i = verticies_[1]->parameters();
		Qd Qi(param_i[3], param_i[4], param_i[5], param_i[6]);
		Vec3 Pi = param_i.head<3>();

		Vec7 param_j = verticies_[2]->parameters();
		Qd Qj(param_j[3], param_j[4], param_j[5], param_j[6]);
		Vec3 Pj = param_j.head<3>();

		Vec7 param_ext = verticies_[3]->parameters();
		Qd qbc(param_ext[3], param_ext[4], param_ext[5], param_ext[6]);
		Vec3 tbc = param_ext.head<3>();

		Vec3 pts_cam_i = pts_norm_i_ / inverse_depth_i;
		Vec3 pts_bi = qbc * pts_cam_i + tbc;
		Vec3 pts_w = Qi * pts_bi + Pi;
		Vec3 pts_bj = Qj.inverse() * (pts_w - Pj);
		Vec3 pts_cam_j = qbc.inverse() * (pts_bj - tbc);

		Mat33 Ri = Qi.toRotationMatrix();
		Mat33 Rj = Qj.toRotationMatrix();
		Mat33 rbc = qbc.toRotationMatrix();

		double dep_j = pts_cam_j[2];
		Mat23 jacobian_res_fj;
		jacobian_res_fj << 1.0 / dep_j, 0, -pts_cam_j[0] / (dep_j * dep_j),
			0, 1.0 / dep_j, -pts_cam_j[1] / (dep_j * dep_j);

		Vec3 jacobian_fj_depth;
		jacobian_fj_depth
			<< rbc.transpose() * Rj.transpose() * Ri * rbc * pts_norm_i_ / -(inverse_depth_i * inverse_depth_i);
		Vec2 jacobian_depth = jacobian_res_fj * jacobian_fj_depth;

		Eigen::Matrix<double, 2, 6> jacobian_pose_i;
		Eigen::Matrix<double, 3, 6> jacobian_fj_posei;
		Mat33 jacobian_i;
		jacobian_i = rbc.transpose() * Rj.transpose();
		jacobian_fj_posei.leftCols<3>() = jacobian_i;
		jacobian_fj_posei.rightCols<3>() = jacobian_i * Ri * -Sophus::SO3d::hat(pts_bi);
		jacobian_pose_i = jacobian_res_fj * jacobian_fj_posei;

		Eigen::Matrix<double, 2, 6> jacobian_pose_j;
		Eigen::Matrix<double, 3, 6> jacobian_fj_posej;
		jacobian_fj_posej.leftCols<3>() = -rbc.transpose() * Rj.transpose();
		jacobian_fj_posej.rightCols<3>() = rbc.transpose() * Sophus::SO3d::hat(pts_bj);
		jacobian_pose_j = jacobian_res_fj * jacobian_fj_posej;

		Eigen::Matrix<double, 2, 6> jacobian_ex;
		Eigen::Matrix<double, 3, 6> jacobian_fj_ex;
		jacobian_fj_ex.leftCols<3>() = rbc.transpose() * (Rj.transpose() * Ri - Mat33::Identity());
		Mat33 temp_r = rbc.transpose() * Rj.transpose() * Ri.transpose() * rbc;
		jacobian_fj_ex.rightCols<3>() = Sophus::SO3d::hat(temp_r * pts_cam_i) - temp_r * Sophus::SO3d::hat(pts_cam_i)
			+ Sophus::SO3d::hat(rbc.transpose() * (Rj.transpose() * (Ri * tbc + Pi - Pj) - tbc));
		jacobian_ex = jacobian_res_fj * jacobian_fj_ex;

		jacobians_[0] = jacobian_depth;
		jacobians_[1] = jacobian_pose_i;
		jacobians_[2] = jacobian_pose_j;
		jacobians_[3] = jacobian_ex;
	}
}