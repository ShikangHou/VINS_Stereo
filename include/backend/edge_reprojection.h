/*
 * @Author: HouShikang
 * @Date: 2024-03-13 20:58:54
 * @Description:
 */
#ifndef BACKEND_REPROJECTION_H
#define BACKEND_REPROJECTION_H

#include "edge.h"

namespace backend
{
  class EdgeReprojection : public Edge
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 　逆深度　　pose_cam_i   pose_cam_j  ex_param
    explicit EdgeReprojection(Vec3 &norm_pts_i, Vec3 &norm_pts_j) : Edge(2, 4, {"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"})
    {
      pts_norm_i_ = norm_pts_i;
      pts_norm_j_ = norm_pts_j;
    }

    std::string typeInfo() override
    {
      return "EdgeReprojection";
    }

    void computeResidual() override;
    void computeJacobian() override;

  private:
    // 　传入的测量值：归一系坐标
    Vec3 pts_norm_i_, pts_norm_j_;
  };

} // namespace backend
#endif // BACKEND_REPROJECTION_H