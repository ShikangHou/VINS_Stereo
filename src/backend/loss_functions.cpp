/*
 * @Author: HouShikang
 * @Date: 2024-03-13 13:58:16
 * @Description:
 */
#include "loss_functions.h"

namespace backend
{
    void HuberLoss::compute(double chi2, Vec3 &rho) const
    {
        double delta2 = delta_ * delta_;
        if (chi2 < delta2) //  e^2 < delta_^2
        {
            rho[0] = chi2;
            rho[1] = 1;
            rho[2] = 0;
        }
        else
        {                                     // x = chi2
            double e = sqrt(chi2);            // absolut value of the error
            rho[0] = 2 * e * delta_ - delta2; // rho(x)   = 2 * delta * x^(1/2) - delta^2
            rho[1] = delta_ / e;              // rho'(x)  = delta / sqrt(x)
            rho[2] = -0.5 * rho[1] / chi2;    // rho''(x) = -1 / (2*x^(3/2)) = -1/2 * (delta/x) / x
        }
    }

}