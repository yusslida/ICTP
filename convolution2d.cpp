#include "convolution.hpp"

void convolution(hls::stream<axis_data> &in_data,
                 hls::stream<axis_data> &out_data)
{
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE axis register both port=in_data
#pragma HLS INTERFACE axis register both port=out_data

    data_t image[IMG_ROWS][IMG_COLS];
    acc_t output[IMG_ROWS][IMG_COLS];

    // Fixed 3x3 Sharpen kernel
    const data_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

#pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

    axis_data local_stream;

    // -------------------------------------------------
    // 1. Read image from AXI stream
    // -------------------------------------------------
load_image_rows:
    for (int row = 0; row < IMG_ROWS; row++) {

    load_image_cols:
        for (int col = 0; col < IMG_COLS; col++) {
#pragma HLS PIPELINE II=1

            local_stream = in_data.read();

            image[row][col] =
                (data_t)local_stream.data;
        }
    }

    // -------------------------------------------------
    // 2. Perform 2D convolution
    // -------------------------------------------------
conv_rows:
    for (int row = 0; row < IMG_ROWS; row++) {

    conv_cols:
        for (int col = 0; col < IMG_COLS; col++) {

            acc_t sum = 0;

        kernel_rows:
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {

            kernel_cols:
                for (int kc = 0; kc < KERNEL_SIZE; kc++) {
#pragma HLS UNROLL

                    int image_row =
                        row + kr - (KERNEL_SIZE / 2);

                    int image_col =
                        col + kc - (KERNEL_SIZE / 2);

                    // Zero padding at image boundaries
                    if (image_row >= 0 &&
                        image_row < IMG_ROWS &&
                        image_col >= 0 &&
                        image_col < IMG_COLS) {

                        sum +=
                            image[image_row][image_col]
                            * kernel[kr][kc];
                    }
                }
            }

            output[row][col] = sum;
        }
    }

    // -------------------------------------------------
    // 3. Send filtered image through AXI stream
    // -------------------------------------------------
write_rows:
    for (int row = 0; row < IMG_ROWS; row++) {

    write_cols:
        for (int col = 0; col < IMG_COLS; col++) {
#pragma HLS PIPELINE II=1

            acc_t value = output[row][col];

            // Clamp result to grayscale range
            if (value < 0) {
                value = 0;
            }

            if (value > 255) {
                value = 255;
            }

            local_stream.data = (data_t)value;

            // Set TLAST only for final pixel
            local_stream.last =
                (row == IMG_ROWS - 1 &&
                 col == IMG_COLS - 1);

            out_data.write(local_stream);
        }
    }
}