/*
 * @author: jjf, Fudan University
 * @date: 2016/10/8
 */
#ifndef __DENSE__H__
#define __DENSE__H__

#include "activation.h"
#include "configure.h"
#include "mem.h"
#include <assert.h>
#include <string.h>

#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{
/*
 * @note: the Fully Connected Layer: Dense Layer
 * 	the input_shape = {INPUT_DIM}
 * 	the output shape = {OUTPUT_DIM}
 */
template<int INPUT_DIM, int OUTPUT_DIM, ACTIVATION AC_FN, int UNROLL_FACTOR = 1, int ORDER = 0, int BW_IN=0, int FL_IN=0, int BW_PARAM=0, int FL_PARAM=0, int BW_OUT=0, int FL_OUT=0>
class Dense
{
//public:
//	Dense(const TYPE_T *WEIGHT)
//	{
//		assert(INPUT_DIM > 0);
//		assert(OUTPUT_DIM > 0);
#if DEBUG
		cout<<"Dense Layer......"<<endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
		/* initialize the weight */
//		for( int i = 0; i < INPUT_DIM + 1; i++)
//		{
//			for( int j = 0; j < OUTPUT_DIM; j++)
//				weight[i][j] = (p_size)WEIGHT[i*OUTPUT_DIM + j];
//		}
//	}
public:
//	ap_fixed<BW_PARAM, BW_PARAM - FL_PARAM>	weight[INPUT_DIM + 1][OUTPUT_DIM];
//	TYPE_T	weight[INPUT_DIM + 1][OUTPUT_DIM];
//	p_size	weight[INPUT_DIM + 1][OUTPUT_DIM];
//	ap_fixed<BW_OUT, BW_OUT - FL_OUT>	res[OUTPUT_DIM];
//	TYPE_T	res[OUTPUT_DIM];
	f_size	res[OUTPUT_DIM];

public:
	/*
	 * @note: the feedforword function
	 * @params: the input data is a 1D array with INPUT_DIM
	 */

//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[INPUT_DIM])
//	void feedforward(TYPE_T data[INPUT_DIM])
	void feedforward(f_size data[INPUT_DIM], const TYPE_T *WEIGHT)
	{
	    DENSE: for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if DENSE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

			/* calculate the weight and bias*/
//	    	ap_fixed<BW_IN + BW_PARAM, BW_IN + BW_PARAM - FL_IN - FL_PARAM> tmp = weight[INPUT_DIM][i];
//	    	TYPE_T tmp = weight[INPUT_DIM][i];
	    	mid_size tmp = (mid_size)WEIGHT[INPUT_DIM*OUTPUT_DIM+i];
			for(int j = 0; j < INPUT_DIM; j++)
			{
#if DENSE_PERF_MODE == PERF_LOW || DENSE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				tmp += (mid_size)data[j] * WEIGHT[j*OUTPUT_DIM+i];
			}

			/* calculate the activation function */
//			res[i] = activation_fn<AC_FN>(tmp) >> (FL_IN + FL_PARAM);
			res[i] = (f_size)activation_fn<AC_FN>(tmp);
		}

		/* for the activation of softmax */
		if( AC_FN == SOFTMAX )
		{
			activation_softmax<OUTPUT_DIM>(res);
		}
	}

//	void feedforward(ap_fixed<BW_IN, BW_IN - FL_IN> data[INPUT_DIM], ap_fixed<BW_OUT, BW_OUT - FL_OUT> result[OUTPUT_DIM/UNROLL_FACTOR])
//	void feedforward(TYPE_T data[INPUT_DIM], TYPE_T result[OUTPUT_DIM/UNROLL_FACTOR])
	void feedforward(f_size data[INPUT_DIM], const TYPE_T *WEIGHT, f_size result[OUTPUT_DIM/UNROLL_FACTOR])
	{
		for( int i = 0; i < OUTPUT_DIM/UNROLL_FACTOR; i++)
		{
#if DENSE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			/* calculate the weight and bias*/
//			ap_fixed<BW_IN + BW_PARAM, BW_IN + BW_PARAM - FL_IN - FL_PARAM> tmp = weight[INPUT_DIM][i+ORDER*(OUTPUT_DIM/UNROLL_FACTOR)];
//			TYPE_T tmp = weight[INPUT_DIM][i+ORDER*(OUTPUT_DIM/UNROLL_FACTOR)];
			mid_size tmp = (mid_size)WEIGHT[INPUT_DIM*OUTPUT_DIM+i+ORDER*(OUTPUT_DIM/UNROLL_FACTOR)];
			for(int j = 0; j < INPUT_DIM; j++)
			{
#if DENSE_PERF_MODE == PERF_LOW || DENSE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				tmp += (mid_size)data[j] * WEIGHT[j*OUTPUT_DIM+i+ORDER*(OUTPUT_DIM/UNROLL_FACTOR)];
			}

			/* calculate the activation function */
//			result[i] = activation_fn<AC_FN>(tmp) >> (FL_IN + FL_PARAM - FL_OUT);
//			trans(activation_fn<AC_FN>(tmp) >> (FL_IN + FL_PARAM), BW_OUT, FL_OUT, result[i]);
			result[i] = (f_size)activation_fn<AC_FN>(tmp);
		}

		/* for the activation of softmax */
		if( AC_FN == SOFTMAX )
		{
			activation_softmax<OUTPUT_DIM>(result);
		}
	}
};

/*
 * @note: the Fully Connected Layer with streamed weight
 * 	the input_shape = {INPUT_DIM}
 * 	the output shape = {OUTPUT_DIM}
 */

}


#endif
