/*
 * @Author: HouShikang
 * @Date: 2024-03-07 15:40:26
 * @Description:
 */
#ifndef BACKEND_PROBLEM_H
#define BACKEND_PROBLEM_H

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"
#include "tic_toc.h"

#include <iostream>
#include <unordered_map>
#include <map>
#include <memory>

namespace backend
{
	typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
	typedef std::map<unsigned long, std::shared_ptr<Edge>> HashEdge;
	typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIDtoEdge;
	typedef std::map<unsigned long, VecX> HashVertexParam;

	class Problem
	{
	 public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		enum class ProblemType
		{
			GENERIC_PROBLEM,
			SLAM_PROBLEM,
		};

		explicit Problem(ProblemType problem_type);

		virtual ~Problem();

		bool solve(int iterations);

		bool addVertex(const std::shared_ptr<Vertex>& v);

		bool removeVertex(const std::shared_ptr<Vertex>& v);

		bool addEdge(const std::shared_ptr<Edge>& e);

		bool removeEdge(const std::shared_ptr<Edge>& e);

		std::vector<std::shared_ptr<Edge>> getConnectEdges(const std::shared_ptr<Vertex>& vertex);

		bool marginalize(const std::vector<std::shared_ptr<Vertex>>& margVerticies, int pose_dim);

	 private:
		void setOrdering();

		void setOrderingSLAM(const std::shared_ptr<Vertex>& v);

		bool checkOrdering();

		void makeHessian();

		void linearSolver(double lambda = -1.0);

		void updateState();

		void rollbackStates();

		static bool isPoseVertex(const std::shared_ptr<Vertex>& v);

		static bool isLandmarkVertex(const std::shared_ptr<Vertex>& v);

		// LM
		void computeLambdaInitLM();

		bool isGoodStepAndUpdateLambdaLM();
		// void addLambdatoHessianLM();
		// void removeLambdaHessianLM();

		ProblemType problem_type_;         // 问题类型
		HashVertex verticies_;             // all verticies
		HashVertex pose_verticies_;        // pose_verticies
		HashVertex landmark_verticies_;    // landmark_verticies
		HashEdge edges_;                   // all edges
		HashVertexIDtoEdge vertexToEdges_; // vertex id to edges
		HashVertex verticies_marg_;        // marginalization

		MatXX H_;
		VecX b_;
		VecX delta_;

		MatXX H_prior_;
		VecX b_prior_;
		VecX error_prior_;
		MatXX Jt_prior_inv_;
		MatXX Hmm_schur_;
		MatXX Hmm_;
		VecX bp_;
		VecX bm_;

		// use to rollback
		HashVertexParam vertex_param_backup_;
		VecX b_prior_backup_;
		VecX error_prior_backup_;

		unsigned long ordering_poses_;
		unsigned long ordering_landmarks_;
		unsigned long ordering_generic_;

		// LM
		double current_lambda_;
		double ni_;               // lambda缩放系数
		double current_chi_;      // 当前误差
		double stop_thresholdLM_; // 迭代退出条件

		double t_hessian_cost_ = 0.0;
	};

}

#endif //BACKEND_PROBLEM_H
