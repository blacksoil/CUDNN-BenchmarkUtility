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

#include <cudnn.h>
#include <time.h>

#include <iostream>

#include "Network.h"

static uint64_t timediff(struct timespec &start, struct timespec &end);
static uint64_t timediff(struct timespec &start, struct timespec &end)
{
    return ((uint64_t)(end.tv_sec - start.tv_sec)*1000000000LL +
      (end.tv_nsec - start.tv_nsec)) / 1000;
}

static struct timespec start;
static struct timespec end;

Layer::Layer()
        : inputs(0), outputs(0), kernel_dim(0),
          data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL)
{
}

Layer::Layer(int _inputs, int _outputs, int _kernel_dim,
    value_type *_data_h, value_type *_data_d, value_type *_bias_h, value_type *_bias_d)
        : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim), data_h(_data_h),
          data_d(_data_d), bias_h(_bias_h), bias_d(_bias_d)
{

    if (data_d == NULL) {
        throw std::invalid_argument("data_d can't be NULL!");
    } else if (bias_d == NULL) {
        throw std::invalid_argument("bias_d can't be NULL!");
    }
    // TODO: Is it okay for data_h or bias_h to be NULL?

    // Do we need to normalize bias for our purpose?
    /*
    for (int i = 0; i < outputs; i++) {
        bias_h[i] /= value_type(255);
    }
    */

    //checkCudaErrors( cudaMemcpy(bias_d, bias_h, outputs*sizeof(value_type),
    //                            cudaMemcpyHostToDevice) );
}

Layer::~Layer()
{
    // for debugging
    // std::cout << "Destorying Layer with input=" << inputs << " and output=" << outputs << std::endl;

    // TODO: This is buggy! Fix this later when have time
    if (data_h != NULL) {
        delete [] data_h;
    }
    if (data_d != NULL) {
        checkCudaErrors( cudaFree(data_d) );
    }
    if (bias_h != NULL) {
        delete [] bias_h;
    }
    if (bias_d != NULL) {
        checkCudaErrors( cudaFree(bias_d) );
    }
}

Network::Network()
{
    mDataType = CUDNN_DATA_FLOAT;
    mTensorFormat = CUDNN_TENSOR_NCHW;
    createHandles();
};

Network::~Network()
{
    destroyHandles();
}

void Network::createHandles()
{
    checkCUDNN( cudnnCreate(&mCudnnHandle) );
    checkCUDNN( cudnnCreateTensorDescriptor(&mSrcTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&mDstTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&mBiasTensorDesc) );
    checkCUDNN( cudnnCreateFilterDescriptor(&mFilterDesc) );
    checkCUDNN( cudnnCreateConvolutionDescriptor(&mConvDesc) );
    checkCUDNN( cudnnCreatePoolingDescriptor(&mPoolingDesc) );
    checkCublasErrors( cublasCreate(&mCublasHandle) );
}
void Network::destroyHandles()
{
    checkCUDNN( cudnnDestroyPoolingDescriptor(mPoolingDesc));
    checkCUDNN( cudnnDestroyConvolutionDescriptor(mConvDesc) );
    checkCUDNN( cudnnDestroyFilterDescriptor(mFilterDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(mSrcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(mDstTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(mBiasTensorDesc) );
    checkCUDNN( cudnnDestroy(mCudnnHandle) );
    checkCublasErrors( cublasDestroy(mCublasHandle) );
}

void Network::resize(int size, value_type **data)
{
    if (*data != NULL)
    {
        checkCudaErrors( cudaFree(*data) );
    }
    checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
}

void Network::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer& layer, int c, value_type *data)
{
    checkCUDNN( cudnnSetTensor4dDescriptor(mBiasTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            1, c,
                                            1,
                                            1) );
    value_type alpha = value_type(1);
    value_type beta  = value_type(1);
    checkCUDNN( cudnnAddTensor(mCudnnHandle, CUDNN_ADD_SAME_C,
                                  &alpha, mBiasTensorDesc,
                                  layer.bias_d,
                                  &beta,
                                  dstTensorDesc,
                                  data) );
}

ConvDimen Network::getConvDimension(const Layer& conv, const ConvDimen &inDimen) {
    checkCUDNN( cudnnSetTensor4dDescriptor(mSrcTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            inDimen.n, inDimen.c,
                                            inDimen.h, inDimen.w) );

    checkCUDNN( cudnnSetFilter4dDescriptor(mFilterDesc,
                                          mDataType,
                                          conv.outputs,
                                          conv.inputs,
                                          conv.kernel_dim,
                                          conv.kernel_dim) );

    checkCUDNN( cudnnSetConvolution2dDescriptor(mConvDesc,
                                               // mSrcTensorDesc,
                                                //mFilterDesc,
                                                0,0, // padding
                                                1,1, // stride
                                                1,1, // upscale
                                                CUDNN_CROSS_CORRELATION) );
    ConvDimen result;
    // find dimension of convolution output
    checkCUDNN( cudnnGetConvolution2dForwardOutputDim(mConvDesc,
                                            mSrcTensorDesc,
                                            mFilterDesc,
                                            &result.n, &result.c, &result.h, &result.w) );   


    return result;
}

ConvAlgo Network::getConvAlgo(const Layer& conv, const ConvDimen &in, const ConvDimen &out) {
    checkCUDNN( cudnnSetTensor4dDescriptor(mSrcTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            in.n, in.c,
                                            in.h, in.w) );

    checkCUDNN( cudnnSetFilter4dDescriptor(mFilterDesc,
                                          mDataType,
                                          conv.outputs,
                                          conv.inputs,
                                          conv.kernel_dim,
                                          conv.kernel_dim) );

    checkCUDNN( cudnnSetConvolution2dDescriptor(mConvDesc,
                                               // mSrcTensorDesc,
                                                //mFilterDesc,
                                                0,0, // padding
                                                1,1, // stride
                                                1,1, // upscale
                                                CUDNN_CROSS_CORRELATION) );
    
    checkCUDNN( cudnnSetTensor4dDescriptor(mDstTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            out.n, out.c,
                                            out.h, out.w) );

    
    ConvAlgo result;
    /*
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(mCudnnHandle,
                                            mSrcTensorDesc,
                                            mFilterDesc,
                                            mConvDesc,
                                            mDstTensorDesc,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            0,
                                            &result.algo
                                            ) );
    */
    
    /*
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(mCudnnHandle,
                                            mSrcTensorDesc,
                                            mFilterDesc,
                                            mConvDesc,
                                            mDstTensorDesc,
                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                            250000000,
                                            &result.algo
                                            ) );
    */
    result.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(mCudnnHandle,
                                            mSrcTensorDesc,
                                            mFilterDesc,
                                            mConvDesc,
                                            mDstTensorDesc,
                                            result.algo,
                                            &result.workspaceSize_b) );

    return result;
}

void Network::convoluteForward(const Layer& conv,
                      const ConvDimen &inDimen, const ConvDimen &outDimen,
                      value_type* srcData, value_type* dstData,
                      const ConvAlgo &convAlgo, value_type* workspace)
{

    checkCUDNN( cudnnSetTensor4dDescriptor(mSrcTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            inDimen.n, inDimen.c,
                                            inDimen.h, inDimen.w) );

    checkCUDNN( cudnnSetFilter4dDescriptor(mFilterDesc,
                                          mDataType,
                                          conv.outputs,
                                          conv.inputs,
                                          conv.kernel_dim,
                                          conv.kernel_dim) );

    checkCUDNN( cudnnSetConvolution2dDescriptor(mConvDesc,
                                               // mSrcTensorDesc,
                                                //mFilterDesc,
                                                0,0, // padding
                                                1,1, // stride
                                                1,1, // upscale
                                                // TODO? CUDNN_CROSS_CORRELATION or CUDNN_CONVOLUTION?
                                                CUDNN_CROSS_CORRELATION) );

    checkCUDNN( cudnnSetTensor4dDescriptor(mDstTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            outDimen.n, outDimen.c,
                                            outDimen.h, outDimen.w) );

    value_type alpha = value_type(1);
    value_type beta  = value_type(0);
    checkCUDNN( cudnnConvolutionForward(mCudnnHandle,
                                          &alpha,
                                          mSrcTensorDesc,
                                          srcData,
                                          mFilterDesc,
                                          conv.data_d,
                                          mConvDesc,
                                          convAlgo.algo,
                                          workspace,
                                          convAlgo.workspaceSize_b,
                                          &beta,
                                          mDstTensorDesc,
                                          dstData) );
    addBias(mDstTensorDesc, conv, outDimen.c, dstData);
}

void Network::activateForward(const ConvDimen inDimen,
    value_type* srcData, value_type** dstData)
{
    int n = inDimen.n;
    int c = inDimen.c;
    int h = inDimen.h;
    int w = inDimen.w;

    checkCUDNN( cudnnSetTensor4dDescriptor(mSrcTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            n, c,
                                            h,
                                            w) );
    checkCUDNN( cudnnSetTensor4dDescriptor(mDstTensorDesc,
                                            mTensorFormat,
                                            mDataType,
                                            n, c,
                                            h,
                                            w) );
    value_type alpha = value_type(1);
    value_type beta  = value_type(0);
    checkCUDNN( cudnnActivationForward(mCudnnHandle,
                                        CUDNN_ACTIVATION_RELU,
                                        &alpha,
                                        mSrcTensorDesc,
                                        srcData,
                                        &beta,
                                        mDstTensorDesc,
                                        *dstData) );
}
