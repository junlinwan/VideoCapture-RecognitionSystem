/*
 * @author: jjf, Fudan University
 * @date: 2016/10/13
 */
#ifndef __POOLING2D_H__
#define __POOLING2D_H__
#include "configure.h"
#include <assert.h>
#include "reshape.h"

#if 1
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

/*
 * @note: 2D Maximum Pooling layer, normally used after Convolution2D layer
 */
template<int ROW, int COL, int NB, int POOL_ROW = 2, int POOL_COL = 2, int BW_IN=0, int FL_IN=0, int OUT_ROW = ROW/POOL_ROW, int OUT_COL = COL/POOL_COL>
class MaxPooling2D
{
public:
	MaxPooling2D()
	{
		assert(OUT_ROW > 0);
		assert(OUT_COL > 0);
#if DEBUG
		cout<<"MaxPooling2D Layer......"<<endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tNB = " << NB << endl;
		cout<<"\tPOOL_ROW = " << POOL_ROW << endl;
		cout<<"\tPOOL_COL = " << POOL_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif
	}
public:
//	ap_fixed<BW_IN, BW_IN - FL_IN> res[OUT_ROW][OUT_COL][NB];
//	TYPE_T res[OUT_ROW][OUT_COL][NB];
	f_size res[OUT_ROW][OUT_COL][NB];

public:
	/*
	 * @note: the input data is ROW x COL x NB 3D array
	 */
//	void feedforward(TYPE_T data[ROW][COL][NB])
//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[ROW][COL][NB])
	void feedforward(f_size data[ROW][COL][NB])
	{
		MAXPOOLING2D: for (int row = 0; row < OUT_ROW; row++)
		{
			for (int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for (int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the maximum value in the local window*/
//					ap_fixed<BW_IN, BW_IN - FL_IN> max = data[row * POOL_ROW][col * POOL_COL][k];
//					TYPE_T max = data[row * POOL_ROW][col * POOL_COL][k];
					f_size max = data[row * POOL_ROW][col * POOL_COL][k];
					for (int i = 0; i < POOL_ROW; i++)
					{
						for (int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
//							ap_fixed<BW_IN, BW_IN - FL_IN> v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
//							TYPE_T v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
							f_size v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
							if (v > max)
								max = v;
						}
					}
					res[row][col][k] = max;
				}
			}
		}
	}

//	void feedforward(TYPE_T data[ROW][COL][NB], TYPE_T result[OUT_ROW][OUT_COL][NB])
//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[ROW][COL][NB], ap_fixed<BW_IN, BW_IN - FL_IN> result[OUT_ROW][OUT_COL][NB])
	void feedforward(f_size data[ROW][COL][NB], f_size result[OUT_ROW][OUT_COL][NB])
	{
		MAXPOOLING2D: for (int row = 0; row < OUT_ROW; row++)
		{
			for (int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for (int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the maximum value in the local window*/
//					TYPE_T max = data[row * POOL_ROW][col * POOL_COL][k];
//					ap_fixed<BW_IN, BW_IN - FL_IN> max = data[row * POOL_ROW][col * POOL_COL][k];
					f_size max = data[row * POOL_ROW][col * POOL_COL][k];
					for (int i = 0; i < POOL_ROW; i++)
					{
						for (int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
//							ap_fixed<BW_IN, BW_IN - FL_IN> v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
//							TYPE_T v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
							f_size v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
							if (v > max)
								max = v;
						}
					}
					result[row][col][k] = max;
				}
			}
		}
	}
};

/*
 * @note: the 2D average pooling layer, normally used after Convolution2D layer
 */
template<int ROW, int COL, int NB, int POOL_ROW, int POOL_COL, int OUT_ROW = ROW/POOL_ROW, int OUT_COL = COL/POOL_COL>
class AveragePooling2D
{
public:
	AveragePooling2D()
	{
#if DEBUG
		cout<<"AveragePooling2D Layer......"<<endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tNB = " << NB << endl;
		cout<<"\tPOOL_ROW = " << POOL_ROW << endl;
		cout<<"\tPOOL_COL = " << POOL_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif
	}
public:
	TYPE_T res[OUT_ROW][OUT_COL][NB];

public:
	/*
	 * @note: the input data is ROW x COL x NB 3D array
	 */
	void feedforward(TYPE_T data[ROW][COL][NB])
	{
		for( int row = 0; row < OUT_ROW; row++)
		{
			for( int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for( int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the mean value in the local window*/
					TYPE_T sum = 0;
					for( int i = 0; i < POOL_ROW; i++)
					{
						for( int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
							sum += data[row*POOL_ROW + i][col*POOL_COL + j][k];
						}
					}
					res[row][col][k] = sum/(POOL_ROW * POOL_COL);
				}
			}
		}
	}
};

}


#endif
