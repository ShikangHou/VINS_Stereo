/*
 * @Author: HouShikang
 * @Date: 2024-03-07 15:40:19
 * @Description:
 */
#include "problem.h"
#include <Eigen/Dense>

namespace backend
{

	Problem::Problem(const ProblemType problem_type)
	{
		problem_type_ = problem_type;
		verticies_marg_.clear();
	}

	Problem::~Problem() = default;

	bool Problem::isPoseVertex(const std::shared_ptr<Vertex>& v)
	{
		std::string type = v->typeInfo();
		return type == std::string("VertexPose") || type == std::string("VertexSpeedBias");
	}

	bool Problem::isLandmarkVertex(const std::shared_ptr<Vertex>& v)
	{
		std::string type = v->typeInfo();
		return type == std::string("VertexPointXYZ") || type == std::string("VertexInverseDepth");
	}

	bool Problem::addVertex(const std::shared_ptr<Vertex>& v)
	{
		if (verticies_.find(v->id()) == verticies_.end()) // 如果未能找到该顶点
		{
			verticies_.insert(std::make_pair(v->id(), v));
			return true;
		}
		else
			return false;
	}

	bool Problem::removeVertex(const std::shared_ptr<Vertex>& v)
	{
		if (verticies_.find(v->id()) == verticies_.end())
		{
			return false;
		}
		std::vector<std::shared_ptr<Edge>> edges = getConnectEdges(v);
		for (auto& edge : edges)
		{
			removeEdge(edge);
		}
		if (isPoseVertex(v))
		{
			pose_verticies_.erase(v->id());
		}
		else if (isLandmarkVertex(v))
		{
			landmark_verticies_.erase(v->id());
		}
		v->setOrderingID(-1);
		verticies_.erase(v->id());
		vertexToEdges_.erase(v->id());

		return true;
	}

	std::vector<std::shared_ptr<Edge>> Problem::getConnectEdges(const std::shared_ptr<Vertex>& vertex)
	{
		std::vector<std::shared_ptr<Edge>> edges;
		auto range = vertexToEdges_.equal_range(vertex->id());
		for (auto iter = range.first; iter != range.second; iter++)
		{
			if (edges_.find(iter->second->id()) == edges_.end()) // 已经被删除了
			{
				continue;
			}
			edges.push_back(iter->second);
		}
		return edges;
	}

	bool Problem::addEdge(const std::shared_ptr<Edge>& e)
	{
		if (edges_.find(e->id()) == edges_.end())
		{
			edges_.insert(std::make_pair(e->id(), e));
		}
		else
		{
			return false;
		}
		for (auto& vetex : e->verticies())
		{
			vertexToEdges_.insert(std::make_pair(vetex->id(), e));
		}
		return true;
	}

	bool Problem::removeEdge(const std::shared_ptr<Edge>& e)
	{
		if (edges_.find(e->id()) == edges_.end())
		{
			return false;
		}
		edges_.erase(e->id());
		return true;
	}

	void Problem::setOrderingSLAM(const std::shared_ptr<Vertex>& v)
	{
		if (isPoseVertex(v))
		{
			v->setOrderingID(ordering_poses_);
			ordering_poses_ += v->dimension();
			pose_verticies_.insert(std::make_pair(v->id(), v));
		}
		else if (isLandmarkVertex(v))
		{
			v->setOrderingID(ordering_landmarks_);
			ordering_landmarks_ += v->dimension();
			landmark_verticies_.insert(std::make_pair(v->id(), v));
		}
	}

	void Problem::setOrdering()
	{
		ordering_generic_ = 0;
		ordering_poses_ = 0;
		ordering_landmarks_ = 0;

		pose_verticies_.clear();
		landmark_verticies_.clear();

		for (const auto& v : verticies_)
		{
			// 　一般问题，不用设置ordering_id,默认为０，直接构造稠密的Hessian矩阵
			ordering_generic_ += v.second->dimension();
			if (problem_type_ == ProblemType::SLAM_PROBLEM)
			{
				setOrderingSLAM(v.second);
			}
		}

		// 将landmark放到pose后面
		if (problem_type_ == ProblemType::SLAM_PROBLEM)
		{
			for (const auto& v : landmark_verticies_)
			{
				v.second->setOrderingID(v.second->orderingID() + ordering_poses_);
			}
		}
	}

	bool Problem::checkOrdering()
	{
		if (problem_type_ == ProblemType::SLAM_PROBLEM)
		{
			int current_ordering = 0;
			for (const auto& v : pose_verticies_)
			{
				assert(v.second->orderingID() == current_ordering);
				current_ordering += v.second->dimension();
			}

			for (const auto& v : landmark_verticies_)
			{
				assert(v.second->orderingID() == current_ordering);
				current_ordering += v.second->dimension();
			}
		}
		return true;
	}

	void Problem::makeHessian()
	{
		TicToc th;
		MatXX H(MatXX::Zero(ordering_generic_, ordering_generic_));
		VecX b(VecX::Zero(ordering_generic_));

		for (auto& edge : edges_)
		{
			edge.second->computeJacobian();
			edge.second->computeResidual();

			auto jacobian = edge.second->jacobians();
			auto residual = edge.second->residual();
			auto verticies = edge.second->verticies();
			auto information = edge.second->information();
			double drho;
			MatXX robust_info;
			edge.second->robustInfo(drho, robust_info);

			assert(jacobian.size() == verticies.size());

			for (size_t i = 0; i < verticies.size(); i++)
			{
				auto vertex_i = verticies[i];
				auto jacobian_i = jacobian[i];
				auto index_i = vertex_i->orderingID();
				auto dim_i = vertex_i->dimension();

				for (size_t j = i; j < verticies.size(); j++)
				{
					auto vertex_j = verticies[j];

					auto jacobian_j = jacobian[j];
					auto index_j = vertex_j->orderingID();
					auto dim_j = vertex_j->dimension();

					MatXX hessian = jacobian_i.transpose() * robust_info * jacobian_j;

					// 上三角
					H.block(index_i, index_j, dim_i, dim_j) += hessian;
					if (i != j)
					{
						// 下三角与其对称
						H.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
					}
				}

				MatXX temp = drho * jacobian_i.transpose() * information * residual;

				b.segment(index_i, dim_i) -= temp;
			}
		}
		H_ = H;
		b_ = b;

		if (H_prior_.rows() > 0)
		{
			MatXX H_prior_temp = H_prior_;
			VecX b_prior_temp = b_prior_;

			H_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_;
			b_.head(ordering_poses_) += b_prior_temp;
		}

		for (const auto& vertex : verticies_)
		{
			if (isPoseVertex(vertex.second) && vertex.second->isFixed())
			{
				int idx = vertex.second->orderingID();
				int dim = vertex.second->dimension();
				H_.block(idx, 0, dim, H_.cols()).setZero();
				H_.block(0, idx, H_.rows(), dim).setZero();
				b_.segment(idx, dim).setZero();
			}
		}

		t_hessian_cost_ += th.toc();
	}

	void Problem::linearSolver(double lambda)
	{
		delta_.resize(ordering_generic_);
		// 一般问题，Hessian是稠密的，直接求解或使用分解矩阵的方式求解
		if (problem_type_ == ProblemType::GENERIC_PROBLEM)
		{
			if (lambda >= 0)
			{
				MatXX m_lambda = lambda * MatXX::Identity(H_.rows(), H_.cols());
				// delta_ = (H_ + m_lambda).inverse() * b_;
				delta_.noalias() = (H_ + m_lambda).ldlt().solve(b_); // ldlt分解针对正定矩阵
			}
			else
			{
				delta_.noalias() = H_.inverse() * b_;
			}
		}

		// slam问题，Hession是稀疏的，使用schur求解
		if (problem_type_ == ProblemType::SLAM_PROBLEM)
		{
			MatXX m_lambda = lambda * MatXX::Identity(H_.rows(), H_.cols());
			MatXX H = H_ + m_lambda;
			int p_size = ordering_poses_;
			int m_size = ordering_landmarks_;
			MatXX Hpp = H.block(0, 0, p_size, p_size);
			MatXX Hmp = H.block(p_size, 0, m_size, p_size);
			MatXX Hpm = H.block(0, p_size, p_size, m_size);
			MatXX Hmm = H.block(p_size, p_size, m_size, m_size);
			VecX bp = b_.head(p_size);
			VecX bm = b_.tail(m_size);

			// Hmm是对角矩阵，求逆
			MatXX Hmm_inv(MatXX::Zero(m_size, m_size));
			for (const auto& landmark : landmark_verticies_)
			{
				int idx = landmark.second->orderingID() - p_size;
				int dim = landmark.second->dimension();
				Hmm_inv.block(idx, idx, dim, dim) = Hmm.block(idx, idx, dim, dim).inverse();
			}

			MatXX temp = Hpm * Hmm_inv;
			Hmm_schur_ = Hpp - temp * Hmp;
			VecX b_schur = bp - temp * bm;
			VecX x_mp = Hmm_schur_.ldlt().solve(b_schur);
			delta_.head(p_size) = x_mp;
			delta_.tail(m_size) = Hmm.ldlt().solve(bm - Hmp * x_mp);
		}
	}

	void Problem::updateState()
	{
		// 更新状态量
		vertex_param_backup_.clear();
		for (auto& vertex : verticies_)
		{
			int oid = vertex.second->orderingID();
			int dim = vertex.second->dimension();
			vertex_param_backup_.insert(std::make_pair(vertex.first, vertex.second->parameters()));
			VecX delta = delta_.segment(oid, dim);

			vertex.second->plus(delta);
			// std::cout<<"---"<<std::endl;
			// std::cout << vertex.second->parameters() << std::endl;
		}

		// std::cout<<" sss"<<std::endl;

		// 更新先验残差
		if (error_prior_.rows() > 0)
		{
			b_prior_backup_ = b_prior_;
			error_prior_backup_ = error_prior_;

			/// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
			/// \delta x = Computes the linearized deviation from the references (linearization points)

			// todo:question: 是否可以先更新error
			b_prior_ -= H_prior_ * delta_.head(ordering_poses_); // update the error_prior
			error_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);
		}
	}

	void Problem::rollbackStates()
	{
		// rollback vertex
		for (auto& vertex : verticies_)
		{
			VecX params = vertex_param_backup_.find(vertex.first)->second;
			vertex.second->setParameters(params);
		}

		// rollback prior
		if (error_prior_.rows() > 0)
		{
			error_prior_ = error_prior_backup_;
			b_prior_ = b_prior_backup_;
		}
	}

	bool Problem::solve(int iterations)
	{
		TicToc th;
		if (edges_.empty() || verticies_.empty())
		{
			std::cerr << "the numbers of edges or verticies is zero! " << std::endl;
			return false;
		}

		setOrdering();
		checkOrdering();
		makeHessian();
		computeLambdaInitLM();
		bool stop = false;
		int iter = 0;
		while (!stop && iter < iterations)
		{
			std::cout << "iter: " << iter << ", chi: " << current_chi_ << ", lambda: " << current_lambda_ << std::endl;

			bool success = false;
			int false_cnt = 0;
			while (!success)
			{
				linearSolver(current_lambda_);

				// 如果delta很小，或者该次迭代尝试超过10次均不理想，则退出
				if (delta_.squaredNorm() <= 1e-6 || false_cnt > 10) // 二范数
				{
					stop = true;
					break;
				}

				updateState();
				success = isGoodStepAndUpdateLambdaLM();

				if (success)
				{
					makeHessian();
					false_cnt = 0;
				}
				else
				{
					rollbackStates();
					false_cnt++;
				}
			}

			iter++; // 　迭代次数递增

			// 误差特别小，退出
			if (sqrt(current_chi_) <= stop_thresholdLM_)
			{
				stop = true;
			}
		}
		std::cout << "solve problem cost time: " << th.toc() << " ms" << std::endl;
		std::cout << "make hessian matrix cost time: " << t_hessian_cost_ << " ms" << std::endl;
		t_hessian_cost_ = 0;

		return true;
	}

	// marg某一帧的位姿PQ和landmarks
	// 如果不想将该帧下的landmarks加入边缘化，就把对应的edge先去掉
	bool Problem::marginalize(const std::vector<std::shared_ptr<Vertex>>& margVerticies, int pose_dim)
	{
		setOrdering();
		// 　获取pose_vertex相连的edges
		std::vector<std::shared_ptr<Edge>> marg_edges = getConnectEdges(margVerticies[0]);

		std::unordered_map<unsigned long, std::shared_ptr<Vertex>> marg_landmarks;

		int marg_landmarks_dim = 0;
		for (auto& edge : marg_edges)
		{
			auto verticies = edge->verticies();

			for (auto& vertex : verticies)
			{
				if (isLandmarkVertex(vertex) && marg_landmarks.find(vertex->id()) == marg_landmarks.end())
				{
					vertex->setOrderingID(pose_dim + marg_landmarks_dim);
					marg_landmarks.insert(std::make_pair(vertex->id(), vertex));
					marg_landmarks_dim += vertex->dimension();
				}
			}
		}

		int size = pose_dim + marg_landmarks_dim;
		MatXX H_marg(MatXX::Zero(size, size));
		VecX b_marg(VecX::Zero(size));

		// H = H_marg + H_prior, H_marg中包括pose和需要marg掉的landmarks,H_prior中只有pose
		// 构造H_marg b_marg
		for (const auto& edge : marg_edges)
		{
			edge->computeJacobian();
			edge->computeResidual();

			auto jacobians = edge->jacobians();
			auto residual = edge->residual();
			auto verticies = edge->verticies();
			double drho;
			MatXX robust_info;
			edge->robustInfo(drho, robust_info);
			MatXX information = edge->information();
			for (size_t i = 0; i < verticies.size(); i++)
			{
				auto v_i = verticies[i];
				MatXX jacobian_i = jacobians[i];
				int index_i = v_i->orderingID();
				int dim_i = v_i->dimension();

				for (size_t j = i; j < verticies.size(); j++)
				{
					auto v_j = verticies[j];
					int index_j = v_j->orderingID();
					int dim_j = v_j->dimension();
					MatXX jacobian_j = jacobians[j];

					MatXX hessian = jacobian_i.transpose() * robust_info * jacobian_j;
					H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
					if (i != j)
					{
						H_marg.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
					}
				}

				b_marg.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * information * residual;
			}
		}

		// marg landmarks
		int reserve_size = pose_dim;
		if (marg_landmarks_dim > 0)
		{
			int marg_size = marg_landmarks_dim;
			MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
			MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
			MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
			MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size);
			VecX bp = b_marg.head(reserve_size);
			VecX bm = b_marg.tail(marg_size);

			MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));

			for (const auto& marg_landmark : marg_landmarks)
			{
				int index = marg_landmark.second->orderingID() - reserve_size;
				int dim = marg_landmark.second->dimension();
				Hmm_inv.block(index, index, dim, dim) = Hmm.block(index, index, dim, dim).inverse();
			}

			MatXX temp = Hmp * Hmm_inv;
			MatXX Hmm_schur = Hpp - temp * Hmp;
			VecX bm_schur = bp - temp * bm;

			// 此时H_marg也只含有pose
			H_marg = Hmm_schur;
			b_marg = bm_schur;
		}

		// 扩充维度
		size = pose_dim - H_prior_.rows();
		H_prior_.conservativeResize(pose_dim, pose_dim);
		b_prior_.conservativeResize(pose_dim);
		H_prior_.rightCols(size).setZero();
		H_prior_.bottomRows(size).setZero();
		b_prior_.tail(size).setZero();

		// H_marg和H_prior只与pose有关
		if (H_prior_.rows() > 0)
		{
			H_marg += H_prior_;
			b_marg += b_prior_;
		}

		int marg_dim = 0;
		// marg frame and speedbias
		for (const auto& vertex : margVerticies)
		{
			int idx = vertex->orderingID();
			int dim = vertex->dimension();
			marg_dim += dim;

			// H 往下移动
			MatXX temp_rows = H_marg.block(idx, 0, dim, reserve_size);
			MatXX temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
			H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
			H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

			// H 往右移动
			MatXX temp_cols = H_marg.block(0, idx, reserve_size, dim);
			MatXX temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
			H_marg.block(0, idx, reserve_size, reserve_size - idx - dim).noalias() = temp_rightCols;
			H_marg.block(0, reserve_size - dim, reserve_size, dim).noalias() = temp_cols;

			// b 往下移动
			VecX temp_b = b_marg.segment(idx, dim);
			VecX temp_railb = b_marg.segment(idx + dim, reserve_size - idx - dim);
			b_marg.segment(idx, reserve_size - idx - dim) = temp_railb;
			b_marg.segment(reserve_size - idx, dim) = temp_b;
		}

		// todo:question
		// trick
		double eps = 1e-8;
		int m2 = marg_dim;
		int n2 = reserve_size - marg_dim; // marg pose
		Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
		Eigen::MatrixXd Amm_inv = saes.eigenvectors()
			* Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(),
				0)).asDiagonal() *
			saes.eigenvectors().transpose();

		Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
		Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
		Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
		Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
		Eigen::VectorXd brr = b_marg.segment(0, n2);
		Eigen::MatrixXd tempB = Arm * Amm_inv;
		H_prior_ = Arr - tempB * Amr;
		b_prior_ = brr - tempB * bmm2;

		// 对得到的 H_prior_ b_prior_进行处理
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
		Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
		Eigen::VectorXd S_inv = Eigen::VectorXd(
			(saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

		Eigen::VectorXd S_sqrt = S.cwiseSqrt();
		Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
		Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
		error_prior_ = -Jt_prior_inv_ * b_prior_;

		MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
		H_prior_ = J.transpose() * J;
		MatXX tmp_h = MatXX((H_prior_.array().abs() > 1e-9).select(H_prior_.array(), 0));
		H_prior_ = tmp_h;

		// remove vertex and remove edge
		for (const auto& margVerticie : margVerticies)
		{
			removeVertex(margVerticie);
		}

		for (const auto& landmarkVertex : marg_landmarks)
		{
			removeVertex(landmarkVertex.second);
		}

		return true;
	}

	///////////////// LM ///////////////////

	void Problem::computeLambdaInitLM()
	{
		ni_ = 2.;
		current_lambda_ = -1.;
		current_chi_ = 0.0;
		// TODO:: robust cost chi2
		for (const auto& edge : edges_)
		{
//			if (edge.second->chi2() > 100) std::cerr << edge.second->typeInfo() << edge.second->residual() << std::endl;
			current_chi_ += edge.second->chi2();
		}
		if (error_prior_.rows() > 0)
			current_chi_ += error_prior_.norm();

		stop_thresholdLM_ = 1e-6 * current_chi_; // 迭代条件为 误差下降 1e-6 倍

		// lambda 策略
		double max_diagonal = 0; // 对角线上的最大值
		int size = H_.cols();
		assert(H_.rows() == H_.cols() && "Hessian is not square");
		for (int i = 0; i < size; ++i)
		{
			max_diagonal = std::max(fabs(H_(i, i)), max_diagonal);
		}
		double tau = 1e-5;
		current_lambda_ = tau * max_diagonal;
	}

	bool Problem::isGoodStepAndUpdateLambdaLM()
	{
		// nielsen策略
		double scale = 0;
		scale = delta_.transpose() * (current_lambda_ * delta_ + b_);
		scale += 1e-6; // make sure it's non-zero :)

		// recompute residuals after update state
		// 统计所有的残差
		double tempChi = 0.0;
		for (const auto& edge : edges_)
		{
			edge.second->computeResidual();
			tempChi += edge.second->chi2();
		}
		double rho = (current_chi_ - tempChi) / scale;
		if (rho > 0 && std::isfinite(tempChi)) // last step was good, 误差在下降
		{
			double alpha = 1. - pow((2 * rho - 1), 3);
			alpha = std::min(alpha, 2. / 3.);
			double scaleFactor = (std::max)(1. / 3., alpha);
			current_lambda_ *= scaleFactor;
			ni_ = 2;
			current_chi_ = tempChi;
			return true;
		}
		else
		{
			current_lambda_ *= ni_;
			ni_ *= 2;
			return false;
		}
	}
	// void Problem::addLambdatoHessianLM()
	// {
	//     int size = H_.cols();
	//     assert(H_.rows() == H_.cols() && "Hessian is not square");
	//     for (int i = 0; i < size; ++i)
	//     {
	//         H_(i, i) += current_lambda_;
	//     }
	// }

	// void Problem::removeLambdaHessianLM()
	// {
	//     ulong size = H_.cols();
	//     assert(H_.rows() == H_.cols() && "Hessian is not square");
	//     // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
	//     for (ulong i = 0; i < size; ++i)
	//     {
	//         H_(i, i) -= current_lambda_;
	//     }
	// }
}