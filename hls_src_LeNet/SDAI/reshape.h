
#ifndef __RESHAPE_H__
#define __RESHAPE_H__
#include "configure.h"
#include "assert.h"


#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

typedef enum{ORDER_X}RESHAPE_MODE;

/*
 * @note: convert 2D array to 1D array
 */
template<int DIM1, int DIM2, RESHAPE_MODE MODE = ORDER_X, int OUTPUT_DIM = DIM1 * DIM2>
class Reshape2D_1D
{
public:
	Reshape2D_1D()
	{
#if DEBUG
		cout <<"Reshape2D_1D Layer......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
		cout <<"\tDIM2 = " << DIM2 << endl;
		cout <<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};

public:
	f_size res[OUTPUT_DIM];

public:
	void feedforward(f_size data[DIM1][DIM2])
	{
		for( int i = 0; i < DIM1; i++)
		{
#if RESHAPE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if RESHAPE_PERF_MODE == PERF_MEDIAN || RESHAPE_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
				switch(MODE)
				{
				case ORDER_X: res[i * DIM2 + j] = data[i][j]; break;
				default: assert(0); break;
				}
			}
		}
	}
};

/*
 * @note: convert 3D array to 1D array
 */
template<int DIM1, int DIM2, int DIM3, int BW_IN=0, int FL_IN=0, RESHAPE_MODE MODE = ORDER_X, int OUTPUT_DIM = DIM1 * DIM2 * DIM3>
class Reshape3D_1D
{
public:
	Reshape3D_1D()
	{
#if DEBUG
		cout <<"Reshape3D_1D Layer......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
		cout <<"\tDIM2 = " << DIM2 << endl;
		cout <<"\tDIM3 = " << DIM3 << endl;
		cout <<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};

public:
//	ap_fixed<BW_IN, BW_IN - FL_IN> res[OUTPUT_DIM];
//	TYPE_T res[OUTPUT_DIM];
	f_size res[OUTPUT_DIM];

public:
//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[DIM1][DIM2][DIM3])
//	void feedforward(TYPE_T data[DIM1][DIM2][DIM3])
	void feedforward(f_size data[DIM1][DIM2][DIM3])
	{
		for( int i = 0; i < DIM1; i++)
		{
#if RESHAPE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if RESHAPE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				for(int k = 0; k < DIM3; k++)
				{
#if RESHAPE_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					switch(MODE)
					{
					case ORDER_X: res[i * DIM2 * DIM3 + j * DIM3 + k] = data[i][j][k]; break;
					default: assert(0); break;
					}
				}
			}
		}
	}

//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[DIM1][DIM2][DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result[DIM1 * DIM2 * DIM3])
//	void feedforward(TYPE_T data[DIM1][DIM2][DIM3], TYPE_T result[DIM1 * DIM2 * DIM3])
	void feedforward(f_size data[DIM1][DIM2][DIM3], f_size result[DIM1 * DIM2 * DIM3])
		{
			for( int i = 0; i < DIM1; i++)
			{
	#if RESHAPE_PERF_MODE == PERF_HIGH
	#pragma HLS pipeline
	#endif
				for( int j = 0; j < DIM2; j++)
				{
	#if RESHAPE_PERF_MODE == PERF_MEDIAN
	#pragma HLS pipeline
	#endif
					for(int k = 0; k < DIM3; k++)
					{
	#if RESHAPE_PERF_MODE == PERF_LOW
	#pragma HLS pipeline
	#endif
						switch(MODE)
						{
						case ORDER_X: result[i * DIM2 * DIM3 + j * DIM3 + k] = data[i][j][k]; break;
						default: assert(0); break;
						}
					}
				}
			}
		}

//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[DIM1][DIM2][DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result1[DIM1 * DIM2 * DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result2[DIM1 * DIM2 * DIM3])
//	void feedforward(TYPE_T data[DIM1][DIM2][DIM3], TYPE_T result1[DIM1 * DIM2 * DIM3], TYPE_T result2[DIM1 * DIM2 * DIM3])
	void feedforward(f_size data[DIM1][DIM2][DIM3], TYPE_T result1[DIM1 * DIM2 * DIM3], f_size result2[DIM1 * DIM2 * DIM3])
	{
		for( int i = 0; i < DIM1; i++)
		{
#if RESHAPE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if RESHAPE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				for(int k = 0; k < DIM3; k++)
				{
#if RESHAPE_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					switch(MODE)
					{
						case ORDER_X:
						{
//							ap_fixed<BW_IN, BW_IN - FL_IN> val = data[i][j][k];
//							TYPE_T val = data[i][j][k];
							f_size val = data[i][j][k];
							result1[i * DIM2 * DIM3 + j * DIM3 + k] = val;
							result2[i * DIM2 * DIM3 + j * DIM3 + k] = val;
							break;
						}
						default: assert(0); break;
					}
				}
			}
		}
	}

//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[DIM1][DIM2][DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result1[DIM1 * DIM2 * DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result2[DIM1 * DIM2 * DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result3[DIM1 * DIM2 * DIM3], ap_fixed<BW_IN, BW_IN - FL_IN> result4[DIM1 * DIM2 * DIM3])
//	void feedforward(TYPE_T data[DIM1][DIM2][DIM3], TYPE_T result1[DIM1 * DIM2 * DIM3], TYPE_T result2[DIM1 * DIM2 * DIM3], TYPE_T result3[DIM1 * DIM2 * DIM3], TYPE_T result4[DIM1 * DIM2 * DIM3])
	void feedforward(f_size data[DIM1][DIM2][DIM3], f_size result1[DIM1 * DIM2 * DIM3], f_size result2[DIM1 * DIM2 * DIM3], f_size result3[DIM1 * DIM2 * DIM3], f_size result4[DIM1 * DIM2 * DIM3])
	{
		for( int i = 0; i < DIM1; i++)
		{
#if RESHAPE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if RESHAPE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				for(int k = 0; k < DIM3; k++)
				{
#if RESHAPE_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					switch(MODE)
					{
						case ORDER_X:
						{
//							ap_fixed<BW_IN, BW_IN - FL_IN> val = data[i][j][k];
//							TYPE_T val = data[i][j][k];
							f_size val = data[i][j][k];
							result1[i * DIM2 * DIM3 + j * DIM3 + k] = val;
							result2[i * DIM2 * DIM3 + j * DIM3 + k] = val;
							result3[i * DIM2 * DIM3 + j * DIM3 + k] = val;
							result4[i * DIM2 * DIM3 + j * DIM3 + k] = val;
							break;
						}
						default: assert(0); break;
					}
				}
			}
		}
	}

};

}

#endif
