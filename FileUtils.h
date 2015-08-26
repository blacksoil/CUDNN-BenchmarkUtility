#ifndef _FILE_UTILS_H_
#define _FILE_UTILS_H

#include "Network.h"
#include "Definition.h"

#include <fstream>

// data_h and data_d need to be deallocated. data_h is initialized using new
// while data_d is initialized using cudaMalloc
int readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d);

// TODO: The hope is for this function to take any image
int loadImageFile(const char* fname, int width, int height, value_type *result);
// size of result is width * height * value_type
int loadPGMImageFile(const char* fname, int width, int height, value_type *result);

#endif
