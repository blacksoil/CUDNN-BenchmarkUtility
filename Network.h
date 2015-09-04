//----------------------------------------------------------------------------------
//
// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------


#ifndef NETWORK_H_
#define NETWORK_H_

#include <cudnn.h>
#include <cublas_v2.h>

#include <sstream>
#include <stdint.h>
#include <stdexcept>

#include "Utils.h"
#include "Definition.h"

struct Layer
{
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h;
    value_type *data_d;
    value_type *bias_h;
    value_type *bias_d;

    Layer();
    Layer(int _inputs, int _outputs, int _kernel_dim,
            value_type *_data_h, value_type *_data_d, value_type *_bias_h, value_type *_bias_d);
    ~Layer();

};

struct ConvDimen
{
    int n;
    int c;
    int h;
    int w;

    ConvDimen(){}
    ConvDimen(const ConvDimen &other) {
        n = other.n;
        c = other.c;
        h = other.h;
        w = other.w;
    }
    ConvDimen(int _n, int _c, int _h, int _w) {
        n = _n;
        c = _c;
        h = _h;
        w = _w;
    }
};

struct ConvAlgo
{
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize_b;
};


class Network
{
    cudnnDataType_t mDataType;
    cudnnTensorFormat_t mTensorFormat;
    cudnnHandle_t mCudnnHandle;
    cudnnTensorDescriptor_t mSrcTensorDesc, mDstTensorDesc, mBiasTensorDesc;
    cudnnFilterDescriptor_t mFilterDesc;
    cudnnConvolutionDescriptor_t mConvDesc;
    cudnnPoolingDescriptor_t mPoolingDesc;
    cublasHandle_t mCublasHandle;

    void createHandles();
    void destroyHandles();

  public:
    Network();
    ~Network();
    void resize(int size, value_type **data);
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer& layer, int c, value_type *data);

    ConvDimen getConvDimension(const Layer& conv, const ConvDimen &in);

    ConvAlgo getConvAlgo(const Layer& conv, const ConvDimen &inDimen, const ConvDimen &outDimen);

    void convoluteForward(const Layer& conv,
                          const ConvDimen &inDimen, const ConvDimen &outDimen,
                          value_type* srcData, value_type* dstData,
                          const ConvAlgo &convAlgo, value_type* workspace);
    void activateForward(const ConvDimen inDimen,
                         value_type* srcData, value_type** dstData);

};



#endif
