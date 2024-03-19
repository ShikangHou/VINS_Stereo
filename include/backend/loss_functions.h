/*
 * @Author: HouShikang
 * @Date: 2024-03-12 21:34:49
 * @Description:
 */
#ifndef BACKEND_LOSS_FUNCTION_H
#define BACKEND_LOSS_FUNCTION_H
#include "eigen_types.h"
namespace backend
{
    class LossFunction
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual ~LossFunction()= default;

        virtual void compute(double chi2, Vec3 &rho) const = 0;
    };

    class HuberLoss : LossFunction
    {
    public:
        explicit HuberLoss(double delta) : delta_(delta){};

        void compute(double chi2,Vec3 &rho) const override;

    private:
        double delta_; // 控制系数
    };

}

#endif //BACKEND_LOSS_FUNCTION_H