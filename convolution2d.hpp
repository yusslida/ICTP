#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

#define IMG_ROWS 128
#define IMG_COLS 128

#define KERNEL_SIZE 3

typedef ap_int<16> data_t;
typedef ap_int<32> acc_t;

typedef ap_axis<32, 0, 0, 0> axis_data;

void convolution(hls::stream<axis_data> &in_data,
                 hls::stream<axis_data> &out_data);

#endif