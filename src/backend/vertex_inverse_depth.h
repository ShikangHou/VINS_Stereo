/*
 * @Author: HouShikang
 * @Date: 2024-03-13 21:15:59
 * @Description:
 */
#ifndef BACKEND_VERTEX_INVERSE_DEPTH_H
#define BACKEND_VERTEX_INVERSE_DEPTH_H

#include "vertex.h"

namespace backend
{
	class VertexInverseDepth : public Vertex
	{
	 public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		VertexInverseDepth() : Vertex(1)
		{
		};

		std::string typeInfo() override
		{
			return "VertexInverseDepth";
		}
	};

} // namespace backend

#endif // BACKEND_VERTEX_INVERSE_DEPTH_H