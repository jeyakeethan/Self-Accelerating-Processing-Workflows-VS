``#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <Constants.h>
#include <models/BlurModel.h>

#include <stdio.h>

#include "lodepng.h"

using namespace std;
int main() {
    
    const char* input_file = "../source/Lenna.png";
    const char* output_file = "../output/output.png";

    vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if (error) cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;

    // Prepare the data
    unsigned char *input_image = new unsigned char[(in_image.size() * 3) / 4];
    unsigned char *output_image = new unsigned char[(in_image.size() * 3) / 4];
    int where = 0;
    for (int i = 0; i < in_image.size(); ++i) {
        if ((i + 1) % 4 != 0) {
            input_image[where] = in_image.at(i);
            output_image[where] = 255;
            where++;
        }
    }

    // Run the filter on it
    BlurModel<unsigned char> blurModel(6);
    blurModel.invoke(input_image, output_image, 512, 512);
    blurModel.execute(1);

    // Prepare data for output
    vector<unsigned char> out_image;
    for (int i = 0; i < (in_image.size() * 3) / 4; ++i) {
        out_image.push_back(output_image[i]);
        if ((i + 1) % 3 == 0) {
            out_image.push_back(255);
        }
    }

    // Output the data
    error = lodepng::encode(output_file, out_image, width, height);

    //if there's an error, display it
    if (error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;

    delete[] input_image;
    delete[] output_image;
    return 0;

}