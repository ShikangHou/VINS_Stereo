/*
 * @Author: HouShikang
 * @Date: 2024-03-07 15:39:43
 * @Description:
 */
#include "vertex.h"
#include <iostream>

namespace backend
{
    Vertex::Vertex(int num_dimension, int local_dimension)
    {

        local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
        id_ = setID();
        ordering_id_ = 0;
        parameters_.resize(num_dimension);
        
  
    }

    Vertex::~Vertex() = default;

    int Vertex::dimension() const
    {
        return local_dimension_;
    }

    unsigned long Vertex::id() const
    {
        // std::cerr<<"id: "<<id_<<std::endl;
        return id_;
    }

    unsigned long Vertex::orderingID() const
    {
        return ordering_id_;
    }

    void Vertex::setOrderingID(const unsigned long ordering_id)
    {
        ordering_id_ = ordering_id;
    }

    VecX Vertex::parameters() const
    {
        return parameters_;
    }

    void Vertex::setParameters(const VecX& param)
    {
        parameters_ = param;

    }

    void Vertex::setFixed(bool fixed)
    {
        fixed_ = fixed;
    }

    bool Vertex::isFixed() const
    {
        return fixed_;
    }

    void Vertex::plus(const VecX &delta)
    {
        parameters_ += delta;
    }

    unsigned long Vertex::setID()
    {
        static unsigned long global_ids = 0;
        return global_ids++;
    }

}