#include "Network.h"
#include "Definition.h"

// TODO: Have multiple implementation for this, depending on 
//       what OS is compiling this program.
#include "FileUtils.h"

#include <iostream>

#define value_type float

long getCurrentTime();
void executeBenchmark();
BenchTime benchmark(int img_w, int img_h, int img_n, int img_filter_c, int filter_k, int filter_dim);

long getCurrentTime() {
    /* Linux */
    struct timeval tv;

    gettimeofday(&tv, NULL);

    long ret = tv.tv_usec;
    /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */

    /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
    ret += (tv.tv_sec * 1000000);

    return ret;
}

// return the time takes to run the operations
BenchTime benchmark(int img_w, int img_h, int img_n, int img_filter_c, int filter_k, int filter_dim) {
     // --- Initializing Layer ---
    int data_size = img_filter_c * filter_k * filter_dim * filter_dim;
    int data_size_b = data_size * sizeof(value_type);
    value_type *data_h = new value_type[data_size];
    value_type *data_d;
    checkCudaErrors(cudaMalloc(&data_d, data_size_b));
    checkCudaErrors(cudaMemcpy(data_d, data_h, data_size_b, cudaMemcpyHostToDevice));
 
    int bias_size = filter_k;
    int bias_size_b = bias_size * sizeof(value_type);
    value_type *bias_h = new value_type[bias_size];
    value_type *bias_d;
    checkCudaErrors(cudaMalloc(&bias_d, bias_size_b));
    checkCudaErrors(cudaMemcpy(bias_d, bias_h, bias_size_b, cudaMemcpyHostToDevice));
 
    Layer layer = Layer(img_filter_c, filter_k, filter_dim, data_h, data_d, bias_h, bias_d);
 
    // --- Initializing image ---
    int img_size = img_w * img_h * img_n * img_filter_c;
    int img_size_b = img_size * sizeof(value_type);
    value_type *img_data_h = new value_type[img_size_b];
    value_type *img_data_d;
    checkCudaErrors(cudaMalloc(&img_data_d, img_size_b));

    Network network;
    const int NUM_ITER = 30;


    std::stringstream ss;
    ss << "Running #" << NUM_ITER << " convoluteForward:" << 
            "w=" << img_w << " h=" << img_h << " c=" << img_filter_c << " k=" << filter_k <<
            " filter_dim=" << filter_dim << std::endl;
    // std::cout << ss.str();
    // TODO: warm-up?
    long timeTaken;
    long deltaTime;
    long maxTime = 0;
    long totalTime = 0;
    int dstDataSize_b;
    int workspaceSize_b;

    ConvDimen inDimen = { img_n, img_filter_c, img_h, img_w };
    ConvDimen outDimen = network.getConvDimension(layer, inDimen);
    ConvAlgo convAlgo = network.getConvAlgo(layer, inDimen, outDimen);

    dstDataSize_b = outDimen.n * outDimen.c *
            outDimen.h * outDimen.w * sizeof(value_type);
    workspaceSize_b = convAlgo.workspaceSize_b;

    value_type *dstData = NULL;
    value_type *workspace = NULL;
    checkCudaErrors(cudaMalloc(&dstData, dstDataSize_b));
    checkCudaErrors(cudaMalloc(&workspace, convAlgo.workspaceSize_b));

    for (int i = 0 ; i < NUM_ITER ; i++) {
        checkCudaErrors(cudaMemcpy(img_data_d, img_data_h, img_size_b, cudaMemcpyHostToDevice));
        timeTaken = getCurrentTime();
        network.convoluteForward(layer, inDimen, outDimen, img_data_d, dstData, convAlgo, workspace);
        deltaTime = getCurrentTime() - timeTaken;
        totalTime += deltaTime;
        maxTime = std::max(maxTime, deltaTime);
    }
    checkCudaErrors(cudaFree(dstData));
    checkCudaErrors(cudaFree(workspace));
    checkCudaErrors(cudaFree(img_data_d));

    ss.str("");
    ss << "algo=" << ((int) convAlgo.algo) << " dstDataSize_b=" <<
            dstDataSize_b << " workspaceSize_b=" << workspaceSize_b << std::endl;
    //std::cout << ss.str();

    ss.str("");
    ss << "Avg time for each=" << (totalTime / NUM_ITER) << "us." << std::endl << std::endl;
    //std::cout << ss.str();        
    BenchTime result = { (totalTime / NUM_ITER), maxTime };

    return result;
}

void executeBenchmark() {
    int img_ws[] = 
        { 240, 480 };//, 960 };//, 960, 1920, 3840 };
    int img_hs[] =
        { 135, 270 };//, 540 };//, 540, 1080, 2160 };
    int img_dims_size = sizeof(img_ws) / sizeof(int);

    int img_n[] = { 4 };
    int img_n_size = sizeof(img_n) / sizeof(int);

    int img_filter_cs[] =
        { 1, 32, 64 };
    int img_filter_cs_size = sizeof(img_filter_cs) / sizeof(int);

    int filter_ks[] =
        { 1, 32, 64 };
    int filter_ks_size = sizeof(filter_ks) / sizeof(int);

    int filter_dims[] =
        { 5, 9};
    int filter_dims_size = sizeof(filter_dims) / sizeof(int);

    std::stringstream ss;
    ss << "n, w, h, c, k, filter_dim, avg_time(us), max_time(us)" << std::endl;
    std::cout << ss.str();

    long total = 0;
    for (int idx_dim = 0; idx_dim < img_dims_size ; idx_dim++) {
        int w = img_ws[idx_dim];
        int h = img_hs[idx_dim];
        for (int idx_c = 0 ; idx_c < img_filter_cs_size ; idx_c++) {
           int c = img_filter_cs[idx_c];
           for (int idx_k = 0 ; idx_k < filter_ks_size ; idx_k++) {
                int k = filter_ks[idx_k];
                for (int idx_fdims = 0 ; idx_fdims < filter_dims_size ; idx_fdims++) {
                    int fdim = filter_dims[idx_fdims];
                    for (int idx_n = 0 ; idx_n < img_n_size ; idx_n++) {
                        int n = img_n[idx_n];
                        ss.str("");
                        BenchTime benchTime = benchmark(w, h, n, c, k, fdim);
                        total += benchTime.avgTime;
                        ss << n << ", " << w << ", " << h << ", " << c << ", " << k << ", " << 
                                fdim << ", " << benchTime.avgTime << ", " << benchTime.maxTime << std::endl;
                        std::cout << ss.str();
                    }
                }
           }
        }
    }

    ss.str("");
    ss << "Total time taken=" << total << " us." << std::endl;
    std::cout << ss.str() << std::endl;
}

int main(int argc, char** argv) {
    executeBenchmark();
    return 0;
}
