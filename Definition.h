#ifndef _DEFINE_H_
#define _DEFINE_H_

#define checkCUDNN(code) {                                        \
    if (code != CUDNN_STATUS_SUCCESS) {                           \
        std::stringstream codeStream, lineStream;                 \
        codeStream << (unsigned)code;                             \
        lineStream << (unsigned)__LINE__;                         \
        const std::string msg = "ErrCode=" +                      \
                codeStream.str() +                                \
                ", ErrMsg=" + cudnnGetErrorString(code) + "\n" +  \
                __FILE__ +  ":" + lineStream.str();               \
        throw std::runtime_error(msg);                            \
        cudaDeviceReset();                                        \
    }                                                             \
}

#define checkCudaErrors(code) {                                   \
    if (code != 0) {                                              \
        std::stringstream codeStream, lineStream;                 \
        std::string codeStr;                                      \
        codeStream << (unsigned)code;                             \
        lineStream << (unsigned)__LINE__;                         \
        const std::string msg = "ErrCode=" +                      \
                codeStream.str() +                                \
                ", ErrMsg=" + cudaGetErrorString(code) + "\n" +   \
                __FILE__ + ":" + lineStream.str();                \
        throw std::runtime_error(msg);                            \
        cudaDeviceReset();                                        \
    }                                                             \
}

#define checkCublasErrors(code) {                                 \
    if (code != 0) {                                              \
        std::stringstream codeStream;                             \
        std::string codeStr;                                      \
        codeStream << (unsigned)code;                             \
        const std::string msg = "ErrCode=" +                      \
                codeStream.str();                                 \
        throw std::runtime_error(msg);                            \
        cudaDeviceReset();                                        \
    }                                                             \
}

#define value_type float

struct BenchTime {
    long avgTime;
    long maxTime;
};

struct CNNResult {
    int h;
    int w;
    value_type* result;
};
#endif
