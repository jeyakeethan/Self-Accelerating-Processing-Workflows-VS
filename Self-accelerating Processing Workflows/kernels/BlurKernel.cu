#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void blur_image(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    int fsize = 5; // Filter size
    if (offset < width * height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        int hits = 0;
        for (int ox = -fsize; ox < fsize + 1; ++ox) {
            for (int oy = -fsize; oy < fsize + 1; ++oy) {
                if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
                    const int currentoffset = (offset + ox + oy * width) * 3;
                    output_red += input_image[currentoffset];
                    output_green += input_image[currentoffset + 1];
                    output_blue += input_image[currentoffset + 2];
                    hits++;
                }
            }
        }
        output_image[offset * 3] = output_red / hits;
        output_image[offset * 3 + 1] = output_green / hits;
        output_image[offset * 3 + 2] = output_blue / hits;
    }
}