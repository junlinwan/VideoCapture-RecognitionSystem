#include <iostream>
#include "./SDAI/dense.h"
#include "./SDAI/convolution2D.h"
#include "./SDAI/pooling2D.h"
#include "./SDAI/reshape.h"
#include "./SDAI/utils.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "top.h"

	
using namespace std;

#define N 480

/* define the first Convolution2D layer */
//Convolution2D<NB_FILTER1, NB_ROW, NB_COL, ROW/2+1, COL, INPUT_DIM, RELU, SUBSAMPLE_ROW, SUBSAMPLE_ROW, 8, 8, 4, 5, 8, 3> conv1_1(weight1, bias1);
//Convolution2D<NB_FILTER1, NB_ROW, NB_COL, ROW/2+1, COL, INPUT_DIM, RELU, SUBSAMPLE_ROW, SUBSAMPLE_ROW, 8, 8, 4, 5, 8, 3> conv1_2(weight1, bias1);
Convolution2D<NB_FILTER1, NB_ROW, NB_COL, ROW/2+1, COL, INPUT_DIM, RELU, SUBSAMPLE_ROW, SUBSAMPLE_ROW, 8, 8, 4, 5, 8, 3> conv1_1;
Convolution2D<NB_FILTER1, NB_ROW, NB_COL, ROW/2+1, COL, INPUT_DIM, RELU, SUBSAMPLE_ROW, SUBSAMPLE_ROW, 8, 8, 4, 5, 8, 3> conv1_2;
/* define the first MaxPooling2D layer */
MaxPooling2D<POOLING1_ROW, POOLING1_COL, NB_FILTER1, POOLING_ROW, POOLING_COL, 8, 3> pool1;
/* define the second Convolution2D layer */
//Convolution2D<NB_FILTER2, NB_ROW, NB_COL, ROW2, COL2, INPUT_DIM2, RELU, SUBSAMPLE_ROW, SUBSAMPLE_ROW, 8, 5, 4, 3, 8, 3> conv2(weight2, bias2);
Convolution2D<NB_FILTER2, NB_ROW, NB_COL, ROW2, COL2, INPUT_DIM2, RELU, SUBSAMPLE_ROW, SUBSAMPLE_ROW, 8, 5, 4, 3, 8, 3> conv2;
/* define the second MaxPooling2D layer */
MaxPooling2D<POOLING2_ROW, POOLING2_COL, NB_FILTER2, POOLING_ROW, POOLING_COL, 8, 3> pool2;
Reshape3D_1D<POOLING2_ROW / POOLING_ROW, POOLING2_COL / POOLING_COL, NB_FILTER2, 8, 3> reshape;
/* define the Dense layer */
//Dense<DENSE_INPUT, DENSE_OUTPUT, RELU, 1, 0, 8, 4, 4, 3, 8, 4> dense1(weight3);
Dense<DENSE_INPUT, DENSE_OUTPUT, RELU, 1, 0, 8, 4, 4, 3, 8, 4> dense1;
/* define the second Dense layer */
//Dense<DENSE_OUTPUT, DENSE2_OUTPUT, SOFTMAX, 1, 0, 8, 3, 4, 3, 8, 3> dense2(weight4);
Dense<DENSE_OUTPUT, DENSE2_OUTPUT, RELU, 1, 0, 8, 3, 4, 3, 8, 3> dense2;


/*inline float pop_stream(AXI_VAL8 const &e)
{
#pragma HLS INLINE off
	union
	{
		unsigned int oval;
		float ival;
	} converter;

//	converter.oval = e.data;
	converter.oval = e.data;
	return converter.ival;
}*/

inline char pop_stream(AXI_VAL8 const &e)
{
#pragma HLS INLINE off
	union
	{
		unsigned int oval;
		char ival;
	} converter;

//	converter.oval = e.data;
	converter.oval = e.data;
	return converter.ival;
}

inline AXI_VAL32 push_stream(int const &v, bool last)
{
#pragma HLS INLINE off
	AXI_VAL32 e;

	union
	{
		unsigned int oval;
		int ival;
	} converter;

	converter.ival = v;
	e.data = converter.oval;
	// set it to sizeof(T) ones
	e.strb = 15;
	e.keep = 15; //e.strb;
	e.user = 0;
	e.last = last ? 1 : 0;
	e.id = 0;
	e.dest = 0;
	return e;
}

inline AXI_VAL32 push_stream_float(float const &v, bool last)
{
#pragma HLS INLINE off
	AXI_VAL32 e;

	union
	{
		unsigned int oval;
		float ival;
	} converter;

	converter.ival = v;
	e.data = converter.oval;
	// set it to sizeof(T) ones
	e.strb = 15;
	e.keep = 15; //e.strb;
	e.user = 0;
	e.last = last ? 1 : 0;
	e.id = 0;
	e.dest = 0;
	return e;
}

/*
void fork(AXI_VAL8 in_map[1080*1920*3], TYPE_T out_map[1080][1920][3])
{
	Fork: for (int i = 0; i < 1080; i++)
	    	for (int j = 0; j < 1920; j++)
	    	{
	#pragma HLS pipeline
	    		for (int k = 0 ; k < 3; k++)
	    		{
	    			TYPE_T val = pop_stream(in_map[i * 1920 * 3 + j * 3 + k]);
	    			out_map[i][j][k] = val;
	    		}
	    	}
}
*/
/*
void rgb2gary(TYPE_T in_data[1080][1920][3], TYPE_T out_data[1080][1920][INPUT_DIM])
{
		for (int i = 0; i < 1080; i++)
		{
				for (int j = 0; j < 1920; j++)
				{
							out_data[i][j][INPUT_DIM] = 587*in_data[i][j][0] + 114*in_data[i][j][1] + 299*in_data[i][j][2];
				}
		}
}
*/
/*
void resize(TYPE_T src[1080][1920][INPUT_DIM], TYPE_T dst[ROW][COL][INPUT_DIM])
{
				double fx = COL/1920;
				double fy = ROW/1080;

				for (int i = 0; i < ROW; i++)
				{
						//uchar* dstData = dst.ptr<uchar>(i);
						double srcy = i / fy;
						int y = floor(srcy);
						double v = srcy - y;

						//uchar* srcData1 = src.ptr<uchar>(y);
						//uchar* srcData2 = src.ptr<uchar>(y+1);

						for (int j = 0; j < COL; j++)
						{
								double srcx = j / fx;
								int x = floor(srcx);
								double u = srcx - x;

								for(int k = 0; k < INPUT_DIM; k++)
								{
										dst[i][j][k] = (1-u)*(1-v)*src[x][y][k] + (1-u)*v*src[x][y+1][k] + u*(1-v)*src[x+1][y][k] + u*v*src[x+1][y+1][k];
								}
						}
				}
}
*/
/*
void depart(TYPE_T in_map[ROW][COL][INPUT_DIM], TYPE_T out_map1[(ROW/2+1)][COL][INPUT_DIM], TYPE_T out_map2[(ROW/2+1)][COL][INPUT_DIM])
{
	Fork: for (int i = 0; i < ROW; i++)
	    	for (int j = 0; j < COL; j++)
	    	{
	#pragma HLS pipeline
	    		for (int k = 0 ; k < INPUT_DIM; k++)
	    		{
	    			if (i <= ROW/2)
	    				out_map1[i][j][k] = in_map[i][j][k];
	                if (i >= ROW/2-1)
	                	out_map2[i-(ROW/2-1)][j][k] = in_map[i][j][k];
	    		}
	    	}
}
*/

//void trans(float in_data, int BW, int FL, int out_data)
void trans(float in_data, int BW, int FL, f_size out_data)
{
	if(in_data*pow(2,FL) < -pow(2,BW-1))
		out_data = -pow(2,BW-1);
	else if(in_data*pow(2,FL) > (pow(2,BW-1)-1))
		out_data = (pow(2,BW-1)-1);
	else
		out_data = round(in_data*pow(2,FL));
}

//void fork_input(AXI_VAL32 in_map[ROW*COL*INPUT_DIM], TYPE_T out_map1[(ROW/2+1)][COL][INPUT_DIM], TYPE_T out_map2[(ROW/2+1)][COL][INPUT_DIM])
//void fork_input(AXI_VAL32 in_map[ROW*COL*INPUT_DIM], conv1_in out_map1[(ROW/2+1)][COL][INPUT_DIM], conv1_in out_map2[(ROW/2+1)][COL][INPUT_DIM])
void fork_input(AXI_VAL8 in_map[ROW*COL*INPUT_DIM], f_size out_map1[(ROW/2+1)][COL][INPUT_DIM], f_size out_map2[(ROW/2+1)][COL][INPUT_DIM])
{
    Fork: for (int i = 0; i < ROW; i++)
    	for (int j = 0; j < COL; j++)
    	{
/*
#pragma HLS pipeline
*/
    		for (int k = 0 ; k < INPUT_DIM; k++)
    		{
//    			conv1_in val = pop_stream(in_map[i * COL * INPUT_DIM + j * INPUT_DIM + k]);
    			f_size val0 =  pop_stream(in_map[i * COL * INPUT_DIM + j * INPUT_DIM + k]);
//    			int val1;
    			//f_size val1;
    			//trans(val0, 8, 8, val1);
    			if (i <= ROW/2)
    				out_map1[i][j][k] = val0;
                if (i >= ROW/2-1)
                	out_map2[i-(ROW/2-1)][j][k] = val0;
    		}
    	}
}

//void Layer1(TYPE_T in_map1[(ROW/2+1)][COL][INPUT_DIM], TYPE_T in_map2[(ROW/2+1)][COL][INPUT_DIM], TYPE_T out_map1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], TYPE_T out_map2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1])
//void Layer1(conv1_in in_map1[(ROW/2+1)][COL][INPUT_DIM], conv1_in in_map2[(ROW/2+1)][COL][INPUT_DIM], conv1_out out_map1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], conv1_out out_map2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1])
void Layer1(f_size in_map1[(ROW/2+1)][COL][INPUT_DIM], f_size in_map2[(ROW/2+1)][COL][INPUT_DIM], f_size out_map1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], f_size out_map2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1])
{
	conv1_1.feedforward(in_map1, weight1, bias1, out_map1);
	conv1_2.feedforward(in_map2, weight1, bias1, out_map2);

}

//void join1(TYPE_T in_map1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], TYPE_T in_map2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], TYPE_T out_map[POOLING1_ROW][POOLING1_COL][NB_FILTER1])
//void join1(conv1_out in_map1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], conv1_out in_map2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], conv1_out out_map[POOLING1_ROW][POOLING1_COL][NB_FILTER1])
void join1(f_size in_map1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], f_size in_map2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1], f_size out_map[POOLING1_ROW][POOLING1_COL][NB_FILTER1])
{
    JOIN: for (int i = 0; i < POOLING1_ROW/2; i++)
    	for (int j = 0; j < POOLING1_COL; j++)
    		for (int k = 0; k < NB_FILTER1; k++)
    		{
/*
#pragma HLS pipeline
*/
    			out_map[i][j][k] = in_map1[i][j][k];
    			out_map[i+POOLING1_ROW/2][j][k] = in_map2[i][j][k];
    		}
}

//void Layer2(TYPE_T in_map[POOLING1_ROW][POOLING1_COL][NB_FILTER1], TYPE_T out_map[ROW2][COL2][INPUT_DIM2])
//void Layer2(conv2_in in_map[POOLING1_ROW][POOLING1_COL][NB_FILTER1], conv2_in out_map[ROW2][COL2][INPUT_DIM2])
void Layer2(f_size in_map[POOLING1_ROW][POOLING1_COL][NB_FILTER1], f_size out_map[ROW2][COL2][INPUT_DIM2])
{
	pool1.feedforward(in_map, out_map);
}

//void Layer3(TYPE_T in_map[ROW2][COL2][INPUT_DIM2], TYPE_T out_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2])
//void Layer3(conv2_in in_map[ROW2][COL2][INPUT_DIM2], conv2_out out_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2])
void Layer3(f_size in_map[ROW2][COL2][INPUT_DIM2], f_size out_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2])
{
	conv2.feedforward(in_map, weight2, bias2);
	pool2.feedforward(conv2.res, out_map);
}
/*
void Layer4(TYPE_T in_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2], TYPE_T out_map1[DENSE_INPUT],
	 TYPE_T out_map2[DENSE_INPUT], TYPE_T out_map3[DENSE_INPUT], TYPE_T out_map4[DENSE_INPUT])
{
	reshape.feedforward(in_map, out_map1, out_map2, out_map3, out_map4);
}
*/

//void Layer4(TYPE_T in_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2], TYPE_T out_map[DENSE_INPUT])
//void Layer4(conv2_out in_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2], conv2_out out_map[DENSE_INPUT])
void Layer4(f_size in_map[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2], f_size out_map[DENSE_INPUT])
{
	reshape.feedforward(in_map, out_map);
}

//void Layer5(TYPE_T in_map[DENSE_INPUT], TYPE_T out_map[DENSE_OUTPUT])
//void Layer5(fc1_in in_map[DENSE_INPUT], fc1_out out_map[DENSE_OUTPUT])
void Layer5(f_size in_map[DENSE_INPUT], f_size out_map[DENSE_OUTPUT])
{
	dense1.feedforward(in_map, weight3, out_map);
}

//TYPE_PINT Layer6(TYPE_T in_map[DENSE_OUTPUT])
//TYPE_PINT Layer6(fc2_in in_map[DENSE_OUTPUT])
TYPE_PINT Layer6(f_size in_map[DENSE_OUTPUT])
{
	dense2.feedforward(in_map, weight4);
	/* find the classified catagory */
	return utils_find_category<DENSE2_OUTPUT>(dense2.res);
}

void CNN(AXI_VAL8 data[ROW*COL*INPUT_DIM], AXI_VAL32 result[1])
{
#pragma HLS INTERFACE axis port=data
//#pragma HLS INTERFACE m_axi port=result offset=direct depth=1
#pragma HLS INTERFACE axis port=result
#pragma HLS INTERFACE s_axilite port=return
//	conv1_in in_buf1[(ROW/2+1)][COL][INPUT_DIM], in_buf2[(ROW/2+1)][COL][INPUT_DIM];
//	TYPE_T in_buf1[(ROW/2+1)][COL][INPUT_DIM], in_buf2[(ROW/2+1)][COL][INPUT_DIM];
	f_size in_buf1[(ROW/2+1)][COL][INPUT_DIM], in_buf2[(ROW/2+1)][COL][INPUT_DIM];
#pragma HLS ARRAY_PARTITION variable=in_buf1 cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=in_buf2 cyclic factor=2 dim=2
//	conv1_out temp1_1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1],
//	       	  temp1_2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1];
//	conv2_in temp1[POOLING1_ROW][POOLING1_COL][NB_FILTER1];
//	conv2_in temp2[ROW2][COL2][INPUT_DIM2];
//	conv2_out temp3[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2];
//	conv2_out temp4[DENSE_INPUT];
//	fc1_out temp5[DENSE_OUTPUT];
//	TYPE_PINT res_int;
//	TYPE_T temp1_1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1],
//	       	  temp1_2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1];
//	TYPE_T temp1[POOLING1_ROW][POOLING1_COL][NB_FILTER1];
//	TYPE_T temp2[ROW2][COL2][INPUT_DIM2];
//	TYPE_T temp3[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2];
//	TYPE_T temp4[DENSE_INPUT];
//	TYPE_T temp5[DENSE_OUTPUT];
//	TYPE_PINT res_int;
	f_size temp1_1[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1],
	       	  temp1_2[POOLING1_ROW/2][POOLING1_COL][NB_FILTER1];
	f_size temp1[POOLING1_ROW][POOLING1_COL][NB_FILTER1];
	f_size temp2[ROW2][COL2][INPUT_DIM2];
	f_size temp3[POOLING2_ROW/POOLING_ROW][POOLING2_COL/POOLING_COL][NB_FILTER2];
	f_size temp4[DENSE_INPUT];
	f_size temp5[DENSE_OUTPUT];

	TYPE_PINT res_int;

#pragma HLS dataflow
		/* preprocess: resize & rgb2gary*/
//		fork(data,raw_input);
//		rgb2gary(raw_input,mid_input);
//		resize(mid_input,ip_input);
//		depart(ip_input,in_buf1,in_buf2);
		/* conv layer1 */
	fork_input(data, in_buf1, in_buf2);
    Layer1(in_buf1, in_buf2, temp1_1, temp1_2);
    join1(temp1_1, temp1_2, temp1);
    /* pooling layer 1 */
    Layer2(temp1, temp2);
    /* conv and pooling layer 2 */
    Layer3(temp2, temp3);
    /* dense layer 1 */
    Layer4(temp3, temp4);
    Layer5(temp4, temp5);

    /* dense layer 2 */
    res_int = Layer6(temp5);
    result[0] = push_stream(res_int,1);
}

