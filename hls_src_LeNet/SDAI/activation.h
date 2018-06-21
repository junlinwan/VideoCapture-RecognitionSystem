/*
 * @author: jjf, Fudan University
 * @date: 2016/10/9
 */
#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
#include <math.h>
#include <assert.h>
#include "configure.h"

namespace SDAI
{

typedef enum{LINEAR, SIGMOID, HARDSIGMOID, TANH, RELU, LEAKYRELU, THRESHOLDEDRELU, SOFTSIGN, SOFTPLUS, SOFTMAX}ACTIVATION;

#define ABS(x)	((x) > 0 ? (x) : -(x))


/*
 * @note: the activation function of LINEAR
 */
inline float activation_linear(f_size x)
{
#pragma HLS INLINE
	return x;
}


/*
 * @note: the activation function of SIGMOID
 */
inline float activation_sigmoid(f_size x)
{
#pragma HLS INLINE
	const float c = -1.0;
	return 1.0/(1.0 + expf(c * x));
}

/*
 * @note: the activation function of HARD SIGMOID
 */
inline float activation_hardsigmoid(f_size x)
{
#pragma HLS INLINE
	const float m = 0.2;
	const float n = 0.5;
	float	v = x * m + n;

	if( v >= 1.0)
		return 1.0;
	else if( v <= 0.0)
		return 0.0;
	else
		return v;
}

/*
 * @note: the activation function of RELU
 */
inline float activation_relu(f_size x)
{
#pragma HLS INLINE
	const float t = 0;
	return x >= t ? x : t;
}


/*
 * @note: the activation function of TANH
 */
inline float activation_tanh(f_size x)
{
#pragma HLS INLINE
	const float c = 2.0;
	return 1.0 - 2.0/(expf(c * x) + 1.0);
}

/*
 * @note: the activation function of softsign
 */
inline float activation_softsign(f_size x)
{
#pragma HLS INLINE
	if( x > 0)
		return x/(1 + x);
	else
		return x/(1 - x);
}

/*
 * @note: the activation function of softplus
 */
inline float activation_softplus(f_size x)
{
#pragma HLS INLINE
	return logf( 1 + expf(x));
}

/*
 * @note: the activation function of softmax
 */
template<int OUTPUT_DIM>
void activation_softmax(f_size in[OUTPUT_DIM])
{
	/* find the maximum */
	float max = in[0];
	for(int i = 1; i < OUTPUT_DIM; i++)
	{
#pragma HLS pipeline
		if( in[i] > max)
			max = in[i];
	}

	/* calculate the new value */
	float val[OUTPUT_DIM];
	float sum = 0;
	for( int i = 0; i < OUTPUT_DIM; i++)
	{
#pragma HLS pipeline
		val[i] = expf(in[i] - max);
		sum += val[i];
	}

	/* calculate the result */
	for( int i = 0; i < OUTPUT_DIM; i++)
	{
#pragma HLS pipeline
		in[i] = val[i]/sum;
	}

}


/*
 * @note: some Advanced Activation Functions
 */
/*
 * @note: the LeakyRelU
 */
inline float activation_leakyrelu(f_size x)
{
	const float alpha = 0.3;
	if( x < 0.0)
		return alpha * x;
	else
		return x;
}

/*
 * @note: the Thresholded ReLU
 */
inline float activation_thresholdedrelu(f_size x)
{
#pragma HLS INLINE
	const float theta = 1.0;
	const float v = 0.0;
	return x > theta ? x : v;
}


/*
 * @note: the  activation function
 */
template<ACTIVATION AC_FN>
inline float activation_fn(f_size x)
{
#pragma HLS INLINE
	float res;
	/* calculate the activation function */
	switch( AC_FN )
	{
	case LINEAR: 		res = activation_linear( x ); 		break;
	case SIGMOID: 		res = activation_sigmoid( x ); 		break;
	case HARDSIGMOID: 	res = activation_hardsigmoid( x ); 	break;
	case TANH: 			res = activation_tanh( x ); 		break;
	case RELU: 			res = activation_relu( x ); 		break;
	case SOFTSIGN: 		res = activation_softsign( x ); 	break;
	case SOFTPLUS: 		res = activation_softplus( x ); 	break;
	case SOFTMAX: 		res = activation_linear( x ); 		break;
	case LEAKYRELU:		res = activation_leakyrelu( x ); 	break;
	case THRESHOLDEDRELU:	res = activation_thresholdedrelu( x ); 	break;
	default: assert( 0 ); break;
	}
	return res;
}

}

#endif
