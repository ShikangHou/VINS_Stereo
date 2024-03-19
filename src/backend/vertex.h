/*
 * @Author: HouShikang
 * @Date: 2024-03-07 15:39:49
 * @Description:
 */
#ifndef BACKEND_VERTEX_H
#define BACKEND_VERTEX_H
#include "eigen_types.h"

namespace backend
{
    using namespace Eigen;

    class Vertex
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
         * @description: 构造函数
         * @param {int} num_dimension 顶点维度
         * @param {int} local_dimension　本地参数化维度，默认为-1，与顶点维度一致
         * @return {*}
         * explicit　防止隐形转换
         */
        explicit Vertex(int num_dimension, int local_dimension = -1);
        virtual ~Vertex();
        int dimension() const;
        unsigned long id() const;
        unsigned long orderingID() const;
        void setOrderingID(unsigned long ordering_id);
        VecX parameters() const;
        void setParameters(const VecX& param);
        void setFixed(bool fixed = true);
        bool isFixed() const;

        virtual void plus(const VecX &delta); //　可重定义，默认为向量加法

        virtual std::string typeInfo() = 0; // vertex type,子类中实现

        static unsigned long setID();

    protected:
        VecX parameters_;
        int local_dimension_;
        unsigned long id_; // 顶点id

		unsigned long ordering_id_; // 用于在jacobian和hessian中定位
        bool fixed_ = false;
    };
}
#endif //BACKEND_VERTEX_H