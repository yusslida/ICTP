#include <iostream>

#include "convolution.hpp"

int main()
{
    hls::stream<axis_data> in_stream;
    hls::stream<axis_data> out_stream;

    axis_data packet;

    data_t image[IMG_ROWS][IMG_COLS];
    data_t reference[IMG_ROWS][IMG_COLS];

    // Same kernel used in convolution.cpp
    const data_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    // -------------------------------------------------
    // 1. Create a simple 128x128 test image
    // -------------------------------------------------
    for (int row = 0; row < IMG_ROWS; row++) {
        for (int col = 0; col < IMG_COLS; col++) {

            // Simple grayscale pattern
            image[row][col] = (row + col) % 256;
        }
    }

    // -------------------------------------------------
    // 2. Create software reference solution
    // -------------------------------------------------
    for (int row = 0; row < IMG_ROWS; row++) {
        for (int col = 0; col < IMG_COLS; col++) {

            acc_t sum = 0;

            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                for (int kc = 0; kc < KERNEL_SIZE; kc++) {

                    int image_row =
                        row + kr - (KERNEL_SIZE / 2);

                    int image_col =
                        col + kc - (KERNEL_SIZE / 2);

                    // Same zero-padding rule as hardware
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

            // Same clipping as hardware
            if (sum < 0)
                sum = 0;

            if (sum > 255)
                sum = 255;

            reference[row][col] = (data_t)sum;
        }
    }

    // -------------------------------------------------
    // 3. Flatten image and send through AXI stream
    // -------------------------------------------------
    for (int row = 0; row < IMG_ROWS; row++) {
        for (int col = 0; col < IMG_COLS; col++) {

            packet.data = image[row][col];

            // Input TLAST on final input pixel
            packet.last =
                (row == IMG_ROWS - 1 &&
                 col == IMG_COLS - 1);

            in_stream.write(packet);
        }
    }

    // -------------------------------------------------
    // 4. Execute DUT
    // -------------------------------------------------
    convolution(in_stream, out_stream);

    // -------------------------------------------------
    // 5. Read output and compare with software result
    // -------------------------------------------------
    int errors = 0;

    for (int row = 0; row < IMG_ROWS; row++) {
        for (int col = 0; col < IMG_COLS; col++) {

            packet = out_stream.read();

            data_t hw_result =
                (data_t)packet.data;

            if (hw_result != reference[row][col]) {

                std::cout
                    << "Error at row "
                    << row
                    << ", col "
                    << col
                    << " expected "
                    << reference[row][col]
                    << " got "
                    << hw_result
                    << std::endl;

                errors++;
            }

            // Check TLAST
            bool final_pixel =
                (row == IMG_ROWS - 1 &&
                 col == IMG_COLS - 1);

            if (final_pixel && packet.last != 1) {

                std::cout
                    << "ERROR: TLAST not active at final pixel"
                    << std::endl;

                errors++;
            }

            if (!final_pixel && packet.last != 0) {

                std::cout
                    << "ERROR: TLAST active too early at row "
                    << row
                    << ", col "
                    << col
                    << std::endl;

                errors++;
            }
        }
    }

    // -------------------------------------------------
    // 6. Final result
    // -------------------------------------------------
    if (errors == 0) {

        std::cout
            << "******** TEST PASSED ********"
            << std::endl;

    } else {

        std::cout
            << "******** TEST FAILED ("
            << errors
            << " errors) ********"
            << std::endl;
    }

    return errors;
}