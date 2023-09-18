#include "instance.h"

#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>

const char *KernelSource = "\n" \
"__kernel void cclt111(                                                        \n" \
"   __global float* input,                                                  \n" \
"   __global float* output,                                                 \n" \
"   const unsigned int row,                                                 \n" \
"   const unsigned int column)                                              \n" \
"{                                                                          \n" \
"   int i = get_global_id(0);                                               \n" \
"   int b = row * column;                                                   \n" \
"   if(i < row)                                                             \n" \
"   for (int x = 0; x < row; x++) {                                         \n" \
"       float res = 0;                                                      \n" \
"       for (int y = 0; y < column; y++)                                    \n" \
"       res = res + input[i * column + y] * input[b + y * row + x];         \n" \
"       output[i * row + x] = res;                                          \n" \
"   }                                                                       \n" \
"}                                                                          \n" \
"\n";

int main () {
          error e;
          size_t x = OpenCL::platform_number(e);
          printf("Platform number: %ld\n", x);
          OpenCL(1);

          /*printf("\n");
          OpenCL *cl = new OpenCL();
          int ret = cl->load_program_from_source(KernelSource, e);
          if (!ret) cl->release_program("cclt111");
          delete cl;*/
}