/*
 * @Author: HouShikang
 * @Date: 2024-03-13 19:19:26
 * @Description:
 */
#ifndef BACKEND_VERTEX_POSE_H
#define BACKEND_VERTEX_POSE_H
#include "vertex.h"

namespace backend
{
	class VertexPose : public Vertex
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		explicit VertexPose() : Vertex(7, 6){};
		~VertexPose() override = default;

		std::string typeInfo() override
		{
			return "VertexPose";
		}

		void plus(const VecX &delta) override;
	};

}

#endif // BACKEND_VERTEX_POSE_H