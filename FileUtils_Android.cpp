#include "FileUtils.h"
#include <fstream>
#include <iostream>

int loadImageFile(const char* fname, int width, int height, value_type *result) {
    return 0;
}

// This also normalizes the image in a way that 
// each pixel value is 0..1
int loadPGMImageFile(const char* fname, int width, int height, value_type *result) {
    std::ifstream input;
    input.open(fname, std::ifstream::in|std::ifstream::binary);

    // Assume that header size won't be longer than 200
    // TODO: Make this less hacky
    std::streamsize len = 200 + width * height;
    char buffer[len];
    char *img_data = NULL;

    // Read the entire file
    input.read(buffer, len);
    int read_len = input.gcount();
    //std::cout << "Size of file=" << fname << " is " << read_len << std::endl;

    // Indicate that we just encountered a \n
    bool newLine = true;
    // Number of lines of header to be stripped
    int headerToStrip = 3;
    for (int i = 0 ; i < read_len ; i++) {
        //std::cout << buffer[i];
        if (buffer[i] == '\n') {
            newLine = true;
        } else if (newLine) {
            if (headerToStrip == 0) {
                // std::cout << std::endl << "stripped header. index=" << i << std::endl; 
                img_data = &buffer[i+1];
                break;
            } else if (buffer[i] != '#') {
                headerToStrip--;
            }
            newLine = false;
        } 
    }
    //std::cout << std::endl;

    if (img_data == NULL) {
        return -1;
    }

    int cnt = 0;
    // Normalize and store this to a file
    for (int j = 0 ; j < height ; j++) {
        for (int i = 0 ; i < width ; i++) {
            result[j*width + i] = (*img_data++) / 255.0f;
        }
        // std::cout << std::endl;
    }

    return 0;
}

int readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    if (!dataFile.is_open()) {
        return -1;
    }
    int size_b = size*sizeof(value_type);
    *data_h = new value_type[size];
    if (!dataFile.read ((char*) *data_h, size_b)) {
        return -1;
    }

    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
    return 0;
}
