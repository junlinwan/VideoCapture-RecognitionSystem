/*
 * @author: jjf, Fudan University
 * @date: 2016/10/9
 */
#ifndef __CONFIGURE_H__
#define __CONFIGURE_H__
#include <ap_fixed.h>

namespace SDAI
{
/*
 * @note: configure the performance priority
 */



/*
 * @note: configure the performance for each layer, it can be configured as PERF_HIGH, PERF_MEDIAN and PERF_LOW
 */
#define PERF_HIGH								3
#define PERF_MEDIAN								2
#define PERF_LOW								1

#define DENSE_PERF_MODE							PERF_LOW
#define CONVOLUTION1D_PERF_MODE					PERF_LOW
#define CONVOLUTION2D_PERF_MODE					PERF_LOW
#define EMBEDDING_PERF_MODE						PERF_LOW
#define POOLING1D_PERF_MODE						PERF_LOW
#define POOLING2D_PERF_MODE						PERF_LOW
#define RECURRENT_PERF_MODE						PERF_LOW
#define RESHAPE_PERF_MODE						PERF_LOW
#define UTILS_PERF_MODE							PERF_LOW

/*
 * @note: configure the memory optimization method for AXI master interface
 */
#define	OPT_NONE								0
#define	OPT_MEM									1
#define OPT_BUFFER								2

#define CONVOLUTION1D_OPT_MODE					OPT_MEM
#define POOLING1D_OPT_MODE						OPT_NONE
#define CONVOLUTION2D_OPT_MODE					OPT_MEM
#define POOLING2D_OPT_MODE						OPT_MEM

/*
 * @note: the Debug switch
 */
#define	DEBUG			0

/*
 * @note: user define data type
 */
typedef			unsigned int				TYPE_PINT;
//typedef		double						TYPE_T;
//typedef			float						TYPE_T;
typedef			int							TYPE_T;

int a;

#include "ap_int.h"
typedef char										f_size;
typedef ap_int<4>									p_size;
typedef short int									mid_size;
//typedef ap_fixed<8,8,AP_RND,AP_SAT>									f_size;
//typedef ap_fixed<4,4,AP_RND,AP_SAT>									p_size;
//typedef ap_fixed<11,11,AP_RND,AP_SAT>								mid_size;

}
#endif
