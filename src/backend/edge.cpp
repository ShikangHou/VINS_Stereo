/*
 * @Author: HouShikang
 * @Date: 2024-03-07 15:39:56
 * @Description:
 */
#include "edge.h"
#include "vertex.h"
#include <iostream>

namespace backend
{
  unsigned long Edge::getID()
  {
    static unsigned global_id = 0;
    return global_id++;
  }

  Edge::Edge(int residual_dimension, int num_verticies,
             const std::vector<std::string> &verticies_types)
  {
    id_ = getID();
    residual_.resize(residual_dimension);
    jacobians_.resize(num_verticies);
    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    information_ = information;
    ordering_id_ = 0;

    if (!verticies_types.empty())
    {
      verticies_types_ = verticies_types;
    }
  }

  Edge::~Edge(){};

  bool Edge::CheckValid()
  {
    if (!verticies_types_.empty())
    {
      for (size_t i = 0; i < verticies_.size(); i++)
      {
        if (verticies_types_[i] != verticies_[i]->typeInfo())
        {
          std::cout << "vertex type does not match, should be: "
                    << verticies_types_[i] << ", but set to "
                    << verticies_[i]->typeInfo() << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  double Edge::chi2() const
  {
    return residual_.transpose() * information_ * residual_;
  }
  double Edge::robustChi2() const
  {
    double e2 = this->chi2();
    if (lossfunction_)
    {
      Eigen::Vector3d rho;
      lossfunction_->compute(e2, rho);
      e2 = rho[0];
    }
    return e2;
  }

  void Edge::robustInfo(double &drho, MatXX &info) const
  {
    if (lossfunction_)
    {
      double e2 = chi2();
      Vec3 rho;
      lossfunction_->compute(e2, rho);

      MatXX robust_info;
      robust_info.setIdentity();
      robust_info *= rho[1] * information_;
      VecX weight_err = information_ * residual_;
      robust_info += 2 * rho[2] * weight_err * weight_err.transpose();

      drho = rho[1];
      info = robust_info;
    }
    else
    {
      drho = 1.0;
      info = information_;
    }
  }

} // namespace backend