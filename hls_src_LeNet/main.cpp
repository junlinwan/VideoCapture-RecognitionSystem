#include <iostream>
#include "top.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
using namespace std;

inline float pop_stream(AXI_VAL32 const &e)
{
#pragma HLS INLINE off
	union
	{
		unsigned int oval;
		float ival;
	} converter;

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

int main()
{
	const int N = 48;
	/* load the data from the text file */
	unsigned int std_result[N];
	unsigned int result[N];
	float		 sample[N*ROW*COL*INPUT_DIM];

	/* load the standard sample category */
	FILE *fp = fopen("result.txt", "r");
	if( !fp)
	{
		cout << " Failed to open result.txt " << endl;
	}
	int i = 0;
	while( fscanf(fp, "%d", &std_result[i]) && i < N)
	{
		i++;
	}
	fclose(fp);

	/* load the MNIST sample */
	fp = fopen("validation.txt", "r");
	if( !fp)
	{
		cout << " Failed to open result.txt " << endl;
	}
	for( int i = 0; i < N; i++)
	{
		for(int j = 0; j < ROW * COL; j++)
			fscanf(fp, "%f", &sample[i*ROW*COL + j]);
	}
	fclose(fp);



	/* the main process */
	for (int i = 0; i < N; i++)
	{
		AXI_VAL32 temp[ROW*COL*INPUT_DIM];
		for (int j = 0; j < ROW*COL*INPUT_DIM; j++)
			temp[j] = push_stream_float(sample[i*ROW*COL*INPUT_DIM+j], 0);
		CNN(temp, &result[i]);
	}

	/* compare the result */
	int  n_wrong = 0;
	for( int i = 0; i < N; i++)
	{
		if( result[i] != std_result[i])
		{
			n_wrong++;
			cout << i << " th failed," << result[i] << " " << std_result[i] << endl;
		}
	}
	float rate = float(N - n_wrong)/float(N);
	cout << n_wrong << " wrong in " << N << "samples with accuracy of " << rate << endl;
	return 0;
}


