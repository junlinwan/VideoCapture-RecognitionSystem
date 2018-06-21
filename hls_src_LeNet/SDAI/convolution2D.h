/*
 * @author: jjf, Fudan University
 * @date: 2016/10/13
 */
#ifndef __CONVOLUTION2D_H__
#define __CONVOLUTION2D_H__
#include "activation.h"
#include "configure.h"
#include <assert.h>
#include "mem.h"
#include "reshape.h"
#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

/*
 * @note: define the convolution2D layer
 */

//template<int NB_FILTER, int NB_ROW, int NB_COL, int ROW, int COL, int INPUT_DIM = 1, ACTIVATION AC_FN=LINEAR, int SUBSAMPLE_ROW=1, int SUBSAMPLE_COL=1,
//		int OUT_ROW=(ROW - NB_ROW)/SUBSAMPLE_ROW + 1, int OUT_COL=(COL - NB_COL)/SUBSAMPLE_COL + 1 >
template<int NB_FILTER, int NB_ROW, int NB_COL, int ROW, int COL, int INPUT_DIM = 1, ACTIVATION AC_FN=LINEAR, int SUBSAMPLE_ROW=1, int SUBSAMPLE_COL=1, int BW_IN=0, int FL_IN=0, int BW_PARAM=0, int FL_PARAM=0, int BW_OUT=0, int FL_OUT=0,
		int OUT_ROW=(ROW - NB_ROW)/SUBSAMPLE_ROW + 1, int OUT_COL=(COL - NB_COL)/SUBSAMPLE_COL + 1 >
class Convolution2D
{

//public:
//	Convolution2D(const TYPE_T *WEIGHT, const TYPE_T *BIAS)
//	{
//		assert(ROW > NB_ROW);
//		assert(COL > NB_COL);
//#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
//#if DEBUG
//		cout<<"Convolution2D Layer......"<<endl;
//		cout<<"\tNB_FILTER = " << NB_FILTER << endl;
//		cout<<"\tNB_ROW = " << NB_ROW << endl;
//		cout<<"\tNB_COL = " << NB_COL << endl;
//		cout<<"\tROW = " << ROW << endl;
//		cout<<"\tCOL = " << COL << endl;
//		cout<<"\tINPUT_DIM = "<<INPUT_DIM << endl;
//		cout<<"\tSUBSAMPLE_ROW = " << SUBSAMPLE_ROW << endl;
//		cout<<"\tSUBSAMPLE_COL = " << SUBSAMPLE_COL << endl;
//		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
//		cout<<"\tOUT_COL = " << OUT_COL << endl;
//#endif

		/* initialize the weight and bias */
//		for( int i = 0; i < NB_ROW; i++)
//		{
//			for( int j = 0; j < NB_COL; j++)
//			{
//				for( int m = 0; m < INPUT_DIM; m++)
//				{
//					for(int n = 0; n < NB_FILTER; n++)
//						weight[i][j][m][n] = (p_size)WEIGHT[ i*NB_COL*INPUT_DIM*NB_FILTER + j*INPUT_DIM*NB_FILTER + m*NB_FILTER + n];
//				}
//			}
//		}
//		for( int i = 0; i < NB_FILTER; i++)
//		{
//			bias[i] = (p_size)BIAS[i];
//		}
//	}

public:
	/*the weights is a 4D array with NB_ROW * NB_COL * INPUT_DIM * NB_FILTER */
//	ap_fixed<BW_PARAM, BW_PARAM - FL_PARAM>	    weight[NB_ROW][NB_COL][INPUT_DIM][NB_FILTER];
//	TYPE_T	    weight[NB_ROW][NB_COL][INPUT_DIM][NB_FILTER];
//	p_size	    weight[NB_ROW][NB_COL][INPUT_DIM][NB_FILTER];
	/*the bias is a 1D array with NB_FILTER */
//	ap_fixed<BW_PARAM, BW_PARAM - FL_PARAM> 	bias[NB_FILTER];
//	TYPE_T 	bias[NB_FILTER];
//	p_size	 	bias[NB_FILTER];
//	ap_fixed<BW_OUT, BW_OUT - FL_OUT>     res[OUT_ROW][OUT_COL][NB_FILTER];
//	TYPE_T     res[OUT_ROW][OUT_COL][NB_FILTER];
	f_size      res[OUT_ROW][OUT_COL][NB_FILTER];
//	const TYPE_T *WEIGHT;
//	const TYPE_T *BIAS;

public:
	/*
	 * @note: the feedback function
	 * @params: the input data is a 3D array, ROW * COL * INPUT_DIM
	 */
//	void feedforward(TYPE_T data[ROW][COL][INPUT_DIM])
//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[ROW][COL][INPUT_DIM])
	void feedforward(f_size data[ROW][COL][INPUT_DIM], const TYPE_T *WEIGHT, const TYPE_T *BIAS)
	{
		for( int row = 0; row < OUT_ROW; row++)
		{
			for( int col = 0; col < OUT_COL; col++)
			{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for (int k = 0; k < NB_FILTER; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the weight and bias */
//					ap_fixed<BW_IN + BW_PARAM, BW_IN + BW_PARAM - FL_IN - FL_PARAM> t = bias[k];
//					TYPE_T t = bias[k];
					mid_size t = (mid_size)BIAS[k];

					for (int m = 0; m < NB_ROW; m++)
					{
						for (int n = 0; n < NB_COL; n++)
						{
							for (int v = 0; v < INPUT_DIM; v++)
							{
#if CONVOLUTION2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
//								t += (mid_size) data[row * SUBSAMPLE_ROW + m][col * SUBSAMPLE_COL + n][v] * weight[m][n][v][k];
								t += (mid_size) data[row * SUBSAMPLE_ROW + m][col * SUBSAMPLE_COL + n][v] * WEIGHT[ m*NB_COL*INPUT_DIM*NB_FILTER + n*INPUT_DIM*NB_FILTER + k*NB_FILTER + v];
							}

						}
					}

					/* calculate the activation function */
					res[row][col][k] = (f_size)activation_fn<AC_FN>(t);
				}
			}
		}
	}

//	void feedforward(TYPE_T data[ROW][COL][INPUT_DIM], TYPE_T result[OUT_ROW][OUT_COL][NB_FILTER])
//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[ROW][COL][INPUT_DIM], ap_fixed<BW_OUT, BW_OUT - FL_OUT> result[OUT_ROW][OUT_COL][NB_FILTER])
	void feedforward(f_size data[ROW][COL][INPUT_DIM], const TYPE_T *WEIGHT, const TYPE_T *BIAS, f_size result[OUT_ROW][OUT_COL][NB_FILTER])
	{
		for (int row = 0; row < OUT_ROW; row++)
		{
			for (int col = 0; col < OUT_COL; col++)
			{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for (int k = 0; k < NB_FILTER; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the weight and bias */
//					ap_fixed<BW_IN + BW_PARAM, BW_IN + BW_PARAM - FL_IN - FL_PARAM> t = bias[k];
//					TYPE_T t = bias[k];
//					mid_size t = bias[k];
					mid_size t = (mid_size)BIAS[k];

					for (int m = 0; m < NB_ROW; m++)
					{
						for (int n = 0; n < NB_COL; n++)
						{
							for (int v = 0; v < INPUT_DIM; v++)
							{
#if CONVOLUTION2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
//								t += (mid_size) data[row * SUBSAMPLE_ROW + m][col * SUBSAMPLE_COL + n][v] * weight[m][n][v][k];
								t += (mid_size) data[row * SUBSAMPLE_ROW + m][col * SUBSAMPLE_COL + n][v] * WEIGHT[ m*NB_COL*INPUT_DIM*NB_FILTER + n*INPUT_DIM*NB_FILTER + k*NB_FILTER + v];
							}

						}
					}

					/* calculate the activation function */

//					result[row][col][k] = activation_fn<AC_FN>(t) >> (FL_IN + FL_PARAM - FL_OUT) ;
//					trans(activation_fn<AC_FN>(t) >> (FL_IN + FL_PARAM), BW_OUT, FL_OUT, result[row][col][k]);
					result[row][col][k] = (f_size)activation_fn<AC_FN>(t);
				}
			}
		}
	}
};

}

#endif
