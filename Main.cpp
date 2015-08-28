#include "Network.h"
#include "Definition.h"

// TODO: Have multiple implementation for this, depending on 
//       what OS is compiling this program.
#include "FileUtils.h"


#include <string>
#include <fstream>
#include <iostream>

#define value_type float

using namespace std;

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


void debug();
void debug() {
    std::cout << "debug()" << std::endl;

    // ------ Begin kernel initialization --------------
    // num of output feature map
    int filter_k = 2;
    // num of input feature map
    int filter_c = 2;
    int filter_dim = 3;
    int filter_size = filter_c * filter_k * filter_dim * filter_dim;
    value_type data_h[filter_size];

    std::stringstream ss;
    ss << "Kernel:" << std::endl;
    // Filter tensor ordering is KCHW
    // See: https://devtalk.nvidia.com/default/topic/774597/?comment=4309844
    for (int idx_k = 0;idx_k < filter_k; idx_k++) {
        ss << "k=" << idx_k+1 << std::endl;
        value_type temp[filter_dim*filter_dim] = {0};
        // Change only the center of filter
        if (idx_k == 0) {
            temp[4] = 5;
        } else {
            temp[4] = 1;
        }
        for (int idx_c = 0;idx_c < filter_c; idx_c++) {
            ss << "  c=" << idx_c+1 << std::endl;
            for (int idx_h = 0;idx_h < filter_dim;idx_h++) {
                for (int idx_w = 0 ;idx_w < filter_dim;idx_w++) {
                    int idx = idx_k * filter_c * filter_dim * filter_dim +
                            idx_c * filter_dim * filter_dim +
                            idx_h * filter_dim + 
                            idx_w;
                    data_h[idx] = temp[idx_h*filter_dim +idx_w];
                    ss << data_h[idx] << " ";
                }
                ss << std::endl;
            }
        }
    }

    std::cout << ss.str() << std::endl;
    value_type* data_d;
    checkCudaErrors( cudaMalloc(&data_d, filter_size*sizeof(value_type)) );
    checkCudaErrors( cudaMemcpy(data_d, data_h, filter_size*sizeof(value_type), cudaMemcpyHostToDevice));


    // ------ Begin bias initialization --------------
    float bias_h[filter_k] = { 2, 1 };
    ss.str("");
    ss << "Bias:" << std::endl;
    for (int idx_k = 0 ; idx_k < filter_k ; idx_k++) {
        ss << "k=" << idx_k+1 << std::endl;
        ss << bias_h[idx_k] << std::endl;
    }
    std::cout << ss.str() << std::endl;
    value_type* bias_d;
    checkCudaErrors( cudaMalloc(&bias_d, filter_k*sizeof(value_type)) );
    checkCudaErrors( cudaMemcpy(bias_d, bias_h, filter_k*sizeof(value_type), cudaMemcpyHostToDevice));


    // ------ Begin layer creation --------------
    ss.str("");
    ss << "Creating layer with c=" << filter_c << " k=" << filter_k
            << " filter_dim=" << filter_dim << std::endl;
    std::cout << ss.str() << std::endl;
    Layer l = Layer(filter_c, filter_k, filter_dim, data_h, data_d, bias_h, bias_d);


    Network network;
    // ------ Begin image initialization --------------
    ss.str("");
    ss << "Image:\n";
    int n = 2, c = filter_c;
    int h = 10;
    int w = 10;
    int image_size = n*c*h*w;
    value_type srcData_h[image_size];
    // Image of all 1s
    for (int n_idx = 0 ; n_idx < n ; n_idx++) {
        ss << "n=" << n_idx +1 << std::endl;
        for (int c_idx = 0 ; c_idx < c ; c_idx++) {
            ss << "c=" << c_idx+1 << std::endl;
            for (int h_idx = 0 ; h_idx < h ; h_idx++) {
                for (int w_idx = 0 ; w_idx < w ; w_idx++) {
                    int idx = n_idx*c*h*w + c_idx*h*w + h_idx*w + w_idx;
                    if (n_idx == 0) {
                        if (c_idx == 0) {
                            srcData_h[idx] = idx;
                        } else {
                            srcData_h[idx] = 2;
                        }
                    } else {
                        srcData_h[idx] = 3;
                    }
                    ss << srcData_h[idx] << " ";
                }
                ss << std::endl;
            }
        }
    }
    value_type *srcData;
    checkCudaErrors( cudaMalloc(&srcData, image_size*sizeof(value_type)) );
    checkCudaErrors( cudaMemcpy(srcData, srcData_h, image_size*sizeof(value_type), cudaMemcpyHostToDevice) );
    std::cout << ss.str() << std::endl;

    // -------- Begin convolution ---------------
    value_type *dstData = NULL;
    ss.str("");
    ss << "Before calling convolute n=" << n << " c=" << c << " h=" << h << " w=" << w << std::endl;
    std::cout << ss.str();
    // The first two params are number of image(s) and number of
    // feature maps per image
    ConvDimen inDimen = { n, c, h, w };
    ConvDimen outDimen = network.getConvDimension(l, inDimen);
    ConvAlgo convAlgo = network.getConvAlgo(l, inDimen, outDimen);
    int workspaceSize_b = convAlgo.workspaceSize_b;
    value_type *workspace = NULL;
    checkCudaErrors(cudaMalloc(&workspace, convAlgo.workspaceSize_b));


    int resultSize = sizeof(value_type) * outDimen.n * outDimen.c * outDimen.h * outDimen.w;
    int resultSize_b = resultSize * sizeof(value_type);
    checkCudaErrors(cudaMalloc(&dstData, resultSize_b));
    network.convoluteForward(l, inDimen, outDimen, srcData, dstData, convAlgo, workspace);
    ss.str("");
    n = outDimen.n;
    c = outDimen.c;
    h = outDimen.h;
    w = outDimen.w;
    ss << "After calling convolute n=" << n << " c=" << c << " h=" << h << " w=" << w << std::endl;
    ss << std::endl;
    std::cout << ss.str();

    value_type result[resultSize];
    ss.str("");
    checkCudaErrors( cudaMemcpy(result, dstData, resultSize_b, cudaMemcpyDeviceToHost) );
    for (int n_idx = 0 ; n_idx < n ; n_idx++) {
        ss << "n=" << n_idx + 1 << std::endl;
        for (int c_idx = 0 ; c_idx < c ; c_idx++) {
            ss << "  c=" << c_idx + 1<< std::endl;
            for (int h_idx = 0 ; h_idx < h ; h_idx++) {
                for (int w_idx = 0 ; w_idx < w ; w_idx++) {
                    int idx = n_idx * c * h * w + 
                            c_idx * h * w +
                            h_idx * w +
                            w_idx;
                    ss << result[idx] << " ";
                }
                ss << std::endl;
            }
            ss << std::endl;
        }
    }

    std::cout << "After convolution:\n" << ss.str();
}




value_type *mImg = NULL;
string IMG_PATH = "/data/cudnn/lena_4k.pgm";
//string IMG_PATH = "/data/cudnn/lena_20.pgm";
string IMG_OUT_PATH = "/data/cudnn/lena_4k_out.pgm";
//string IMG_OUT_PATH = "/data/cudnn/lena_20_out.pgm";
const int IMG_WIDTH = 3840;
const int IMG_HEIGHT = 2160;
//const int IMG_WIDTH = 36;
//const int IMG_HEIGHT = 20;


int loadImage(const string &path);
int loadImage(const string &path) {
    if (mImg == NULL) {
        mImg = new value_type[sizeof(value_type) * IMG_WIDTH * IMG_HEIGHT];
    }
         
    return loadPGMImageFile(path.c_str(), IMG_WIDTH, IMG_HEIGHT, mImg);
}

CNNResult loadLayerParams();
CNNResult loadLayerParams() {
    int c = 1;
    int k = 1;
    int dim = 3;
    int fsize = c * k * dim * dim;
    int fsize_b = fsize * sizeof(value_type);
    value_type *data_h = new value_type[fsize];
    value_type *data_d;
    for (int idx_k = 0 ; idx_k < k ; idx_k++) {
        for (int idx_c = 0 ; idx_c < c ; idx_c++) {
            for (int idx_dim_y = 0 ; idx_dim_y < dim ; idx_dim_y++) {
                for (int idx_dim_x = 0 ; idx_dim_x < dim ; idx_dim_x++) {
                    int idx = idx_k * c * dim * dim + 
                            idx_c * dim * dim +
                            idx_dim_y * dim + 
                            idx_dim_x;

                    if (idx_dim_x == 1 && idx_dim_y == 1) {
                        data_h[idx] = 1.5;
                    } else {
                        data_h[idx] = 0;
                    }
                }
            }
        }
    }

    for (int i = 0 ; i < fsize ; i++) {
        cout << "i=" << i << " " << data_h[i] << endl;
    }

    int bsize = k;
    int bsize_b = bsize * sizeof(value_type);
    value_type *bias_h = new value_type[bsize];
    value_type *bias_d;

    checkCudaErrors(cudaMalloc(&data_d, fsize_b));
    checkCudaErrors(cudaMemcpy(data_d, data_h, fsize_b, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&bias_d, bsize_b));
    Layer layer = Layer(c, k, dim, data_h, data_d, bias_h, bias_d);

    Network network;
    int n = 1;
    ConvDimen inDimen = { n, c, IMG_HEIGHT, IMG_WIDTH };
    ConvDimen outDimen = network.getConvDimension(layer, inDimen);

    cout << "After convolution n=" << outDimen.n << " c=" << outDimen.c << " h=" <<
            outDimen.h << " w=" << outDimen.w << endl;

    value_type *img_data_d;
    int isize_b = IMG_WIDTH * IMG_HEIGHT * sizeof(value_type);
    checkCudaErrors(cudaMalloc(&img_data_d, isize_b));
    checkCudaErrors(cudaMemcpy(img_data_d, mImg, isize_b, cudaMemcpyHostToDevice));

    value_type *dstData;
    int dstsize_b = outDimen.w * outDimen.h * sizeof(value_type);
    checkCudaErrors(cudaMalloc(&dstData, dstsize_b));

    ConvAlgo convAlgo = network.getConvAlgo(layer, inDimen, outDimen);
    int workspaceSize_b = convAlgo.workspaceSize_b;
    value_type *workspace = NULL;
    checkCudaErrors(cudaMalloc(&workspace, convAlgo.workspaceSize_b));

    long time = getCurrentTime();
    network.convoluteForward(layer, inDimen, outDimen, img_data_d, dstData, convAlgo, workspace);
    cout << "Convolution takes " << (getCurrentTime() - time) << "us." << endl;

    checkCudaErrors(cudaThreadSynchronize());
    memset(mImg, 0, IMG_WIDTH * IMG_HEIGHT * sizeof(value_type));
    checkCudaErrors(cudaMemcpy(mImg, dstData, dstsize_b, cudaMemcpyDeviceToHost));

    CNNResult result = { outDimen.h, outDimen.w, mImg };

    return result;
}

int saveImage(const value_type *img, const int height, const int width, const string &path);
int saveImage(const value_type *img, const int height, const int width, const string &path) {
    ofstream out;
    out.open(path.c_str(), ios::out | ios::binary | ios::trunc);

    if (out.is_open()) {
        // print out header
        out << "P5" << endl;
        out << "# CREATOR: GIMP PNM Filter Version 1.1" << endl;
        out << width << " " << height << endl;
        out << 255 << endl;

        stringstream ss;
        char img_b[height][width];
        for (int y = 0 ; y < height ; y++) {
            for (int x = 0 ; x < width ; x++) {
                int val = img[y * width + x] * 255.0f;
                //ss << img[y * IMG_WIDTH + x] << endl;
                img_b[y][x] = (val > 255 ? 255 : val);
            }
            out.write(reinterpret_cast<const char*>(img_b[y]), width);
        }

        out.close();
        return 0;
    }
    return -1;
}

int doExecution();
int doExecution() {
    long time = getCurrentTime();
    if (loadImage(IMG_PATH)) {
        std::cout << "Can't load image=" << IMG_PATH << std::endl;
        return -1;
    }

    stringstream ss;
    ss << "Image=" << IMG_PATH << " is loaded in " << (getCurrentTime() - time) / 1e3f << "ms." << endl;
    cout << ss.str();

    CNNResult res = loadLayerParams();

    time = getCurrentTime();
    string outImgPath = IMG_OUT_PATH;
    if (saveImage(res.result, res.h, res.w,  outImgPath)) {
        cout << "Can't save image=" << outImgPath << endl;
        return -1;
    }

    ss.str("");
    ss << "Image=" << outImgPath << " is saved in " << (getCurrentTime() - time) / 1e3f << "ms." << endl;
    cout << ss.str();

    return 0;
}

int main(int argc, char** argv) {
    //executeBenchmark();
    doExecution();
    //debug();
    return 0;
}
