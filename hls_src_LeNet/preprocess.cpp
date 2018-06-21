#include <math.h>
#include <string.h>
#include <stdio.h>

void resize(Mat src, Mat dst, Size dsize, double fx, double fy)
{
				if(src.empty())
						throw exception("Image is empty!");
				int c = src.channels();
				if(dsize.width != 0 && dsize.height != 0)
				{
						fx = (double)dsize.width / (double)src.cols;
						fy = (double)dsize.height / (double)src.rows;
				}
				else if(fx != 0 && fy != 0)
				{
						dsize.width = int(src.cols*fx);
						dsize.height = int(src.rows*fy);
				}
				else
				{
						throw exception("Invalid parameter!");
				}
				for (int i = 0; i < dst.rows; i++)
				{
						uchar* dstData = dst.ptr<uchar>(i);
						double srcy = i / fy;
						int y = cvFloor(srcy);
						double v = srcy - y;
						if(v < 0)
						{
								y = 0;
								v = 0;
						}
						if(y >= src.rows - 1)
						{
								y = src.rows - 2;
								v = 1;
						}
						uchar* srcData1 = src.ptr<uchar>(y);
						uchar* srcData2 = src.ptr<uchar>(y+1);
						for (int j = 0; j < dst.cols*c; j += c)
						{
								double srcx = (j/c) / fx;
								int x = cvFloor(srcx);
								double u = srcx - x;
								if(x < 0)
								{
										x = 0;
										u = 0;
								}
								if(x >= src.cols - 1)
								{
										x = src.clos -2;
										u = 1;
								}
								for£¨int k = 0; k < c; k++£©
								{
										dstData[j+k] = (1-u)*(1-v)*srcData1[x*c+k] + (1-u)*v*srcData2[x*c+k] + u*(1-v)*srcData1[(x+1)*c+k] + u*v*srcData2[(x+1)*c+k];
								}
						}
				}
}

void rgb2gary(Mat src, Mat dst)
{
	int alpha = 0xff << 24;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
		uchar* dstData = dst.ptr<uchar>(i);
		uchar* srcData = src.ptr<uchar>();
		int color = srcData[src.cols*i+j];
		int red = (color & 0x00ff0000) >> 16;
		int green = (color & 0x0000ff00) >> 8;
		int blue = color & 0x000000ff;
		color = (red*299 + green*587 + blue*144) / 1000;
		color = alpha | (color << 16) | (color << 8) | color;
		dstData[src.cols*i+j] = color;
			}
		}
	}