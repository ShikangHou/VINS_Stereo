/*
 * @Author: HouShikang
 * @Date: 2024-03-07 15:40:01
 * @Description: backend edge
 */

#ifndef BACKEND_EDGE_H
#define BACKEND_EDGE_H

#include <memory>
#include <utility>
#include "eigen_types.h"
#include "loss_functions.h"

namespace backend
{
	class Vertex;
	class Edge
	{
	 public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		explicit Edge(int residual_dimension, int num_verticies,
			const std::vector<std::string>& verticies_types = std::vector<std::string>());
		virtual ~Edge();
		static unsigned long getID();

		// 添加一个顶点
		void addVertex(const std::shared_ptr<Vertex>& v)
		{
			verticies_.push_back(v);
		}

		// 设置所有顶点
		void setVerticies(const std::vector<std::shared_ptr<Vertex>>& vs)
		{
			verticies_ = vs;
		}

		// 获取第i个顶点
		std::shared_ptr<Vertex> vertex(int i) const
		{
			return verticies_[i];
		}

		// 获取所有顶点
		std::vector<std::shared_ptr<Vertex>> verticies() const
		{
			return verticies_;
		}

		// 获取顶点的数量
		int verticies_num() const
		{
			return verticies_.size();
		}

		// 设置信息矩阵
		void setInformation(const MatXX& information)
		{
			information_ = information;
		}

		// 获取信息矩阵
		MatXX information() const
		{
			return information_;
		}

		std::vector<MatXX> jacobians() const
		{
			return jacobians_;
		}
		VecX residual() const
		{
			return residual_;
		}

		// 设置观测值
		void setMeasurement(const VecX& m)
		{
			measurements_ = m;
		}

		// 获取观测值
		VecX measureMent()
		{
			return measurements_;
		}

		// 计算平方误差
		double chi2() const;

		double robustChi2() const;

		void setLossFunction(const std::shared_ptr<LossFunction>& ptr)
		{
			lossfunction_ = ptr;
		}
		std::shared_ptr<LossFunction> getLossFunction()
		{
			return lossfunction_;
		}

		void robustInfo(double& drho, MatXX& info) const;

		unsigned long id() const
		{
			return id_;
		}

		int OrderingId() const
		{
			return ordering_id_;
		}

		void SetOrderingId(int id)
		{
			ordering_id_ = id;
		};

		/// 检查边的信息是否全部设置
		bool CheckValid();

		virtual std::string typeInfo() = 0;
		virtual void computeResidual() = 0;
		virtual void computeJacobian() = 0;

	 protected:
		unsigned long id_;
		int ordering_id_;
		std::vector<std::string> verticies_types_;
		std::vector<std::shared_ptr<Vertex>> verticies_;
		VecX residual_;                // 残差
		std::vector<MatXX> jacobians_; // 每个jaobian的维度是　residial * verticies[i]
		MatXX information_;            // 信息矩阵
		VecX measurements_;            // 观测

		std::shared_ptr<LossFunction> lossfunction_ = nullptr;
	};

}

#endif // BACKEND_EDGE_H