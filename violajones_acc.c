/*
##############################################################################
## THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
## OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, 
## ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, 
## INNOVATIVE OR RELEVANT NATURE, FITNESS FOR A PARTICULAR PURPOSE OR 
## COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.
## In the event of publication, the following notice is applicable:
##
##              (C) COPYRIGHT 2010 THALES RESEARCH & TECHNOLOGY
##                            ALL RIGHTS RESERVED
##
## The entire notice above must be reproduced on all authorized copies.
##
##
## Title:             violajones.c
##
## File:              C file
## Author:            Teodora Petrisor <claudia-teodora.petrisor@thalesgroup.com>
## Description:       C source file
##
## Modification:
## Author:            Paul Brelet  <paul.brelet@thalesgroup.com>
##
###############################################################################
*/

/* ************************************************************************* 
* Pedestrian detection application (adapted from OpenCV)
*        - classification based on Viola&Jones 2001 algorithm 
* 			(Haar-like features, AdaBoost algorithm)
*		 - learning data transcripted from OpenCV generated file
*
* authors:  Teodora Petrisor
* Modifications: Paul Brelet
* 
* ************************************************************************* */

/******INCLUDE********/
/*********DECLARATION*****/
/* Global Library */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <assert.h>

/* Static Library */
#include "violajones.h"

/******MACROS ********/
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )


#define MAX_BUFFERSIZE    256                    /* Maximum name file */
#define MAX_IMAGESIZE     1024                   /* Maximum Image Size */
#define MAX_BRIGHTNESS    255                    /* Maximum gray level */
#define NB_MAX_DETECTION  100                    /* Maximum number of detections */
#define NB_MAX_POINTS     3*NB_MAX_DETECTION     /* Maximum number of detection parameters (3 points/detection) */


#if INFO
#define TRACE_INFO(x) printf x
#else
#define TRACE_INFO(x)
#endif


/* ********************************** FUNCTIONS ********************************** */

/*** Read pgm file, only P2 or P5 type image ***/
void load_image_check(uint32_t *img, char *imgName, int width, int height)
{
	char buffer[MAX_BUFFERSIZE] = {0};   /* Get Image information */
	FILE *fp = NULL;                     /* File pointer */
	int x_size1 = 0, y_size1 = 0;        /* width & height of image1*/
	int max_gray = 0;                    /* Maximum gray level */
	int x = 0, y = 0;                    /* Loop variable */
	int pixel_in = 0;                    /* Get the pixel value */
	int error = 0;                       /* Check if errors */

	/* Input file open */
	
	fp = fopen(imgName, "rb");
	if (NULL == fp)
	{
		TRACE_INFO(("     The file %s doesn't exist!\n\n" , imgName));
		exit(1);
	}
	/* Check of file-type ---P2 or P5 */
	fgets(buffer, MAX_BUFFERSIZE, fp);

	if(buffer[0] == 'P' && buffer[1] == '2')
	{
		/* input of x_size1, y_size1 */
		x_size1 = 0;
		y_size1 = 0;
		while (x_size1 == 0 || y_size1 == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d %d", &x_size1, &y_size1);
			}
		}
		/* input of max_gray */
		max_gray = 0;
		while (max_gray == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d", &max_gray);
			}
		}
		/* Display parameters */
		if (x_size1 > MAX_IMAGESIZE || y_size1 > MAX_IMAGESIZE)
		{
			TRACE_INFO(("     Image size exceeds %d x %d\n\n", MAX_IMAGESIZE, MAX_IMAGESIZE));
			TRACE_INFO(("     Please use smaller images!\n\n"));
			exit(1);
		}
		if (max_gray != MAX_BRIGHTNESS)
		{
			TRACE_INFO(("     Invalid value of maximum gray level!\n\n"));
			exit(1);
		}
		/* Input of image data*/
		for(y=0; y < y_size1; y++)
		{
			for(x=0; x < x_size1; x++)
			{
        // read PGM pixel and check input stream state 
				error = fscanf(fp, "%d", &pixel_in);
				if (error <= 0) 
				{
					if (feof(fp))
					{
						TRACE_INFO(("PGM file, premature EOF !\n"));
					}
					else if (ferror(fp))
					{
						TRACE_INFO(("PGM file format error !\n"));
					}
					else
					{
						TRACE_INFO(("PGM file, fatal error during read !\n"));
						exit(1);
					}
				}
				img[y*x_size1+x] = pixel_in;
			}
		}
	}
	else if(buffer[0] == 'P' && buffer[1] == '5') 
	{
		/* Input of x_size1, y_size1 */
		x_size1 = 0;
		y_size1 = 0;
		while (x_size1 == 0 || y_size1 == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d %d", &x_size1, &y_size1);
			}
		}
		/* Input of max_gray */
		max_gray = 0;
		while (max_gray == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d", &max_gray);
			}
		}
		/* Display parameters */
		TRACE_INFO(("\n   Image width = %d, Image height = %d\n", x_size1, y_size1));
		TRACE_INFO(("     Maximum gray level = %d\n\n", max_gray));
		if (x_size1 > MAX_IMAGESIZE || y_size1 > MAX_IMAGESIZE)
		{
			TRACE_INFO(("     Image size exceeds %d x %d\n\n", MAX_IMAGESIZE, MAX_IMAGESIZE));
			TRACE_INFO(("     Please use smaller images!\n\n"));
			exit(1);
		}
		if (max_gray != MAX_BRIGHTNESS)
		{
			TRACE_INFO(("     Invalid value of maximum gray level!\n\n"));
			exit(1);
		}
		/* Input of image data*/
		for (y = 0; y < y_size1; y++)
		{
			for (x = 0; x < x_size1; x++)
			{
				img[y*x_size1+x] = (uint32_t) fgetc(fp);
			}
		}
	}
	else
	{
		TRACE_INFO(("    Wrong file format, only P2/P5 allowed!\n\n"));
		exit(1);
	}
	fclose(fp);
}
//end function: load_image_check ***********************************************

/*** Get the MAX pixel value from image ****/
int maxImage(uint32_t *img, int height, int width)
{
	int maximg = 0;
	int irow = 0;

	for(irow = 0; irow < height*width; irow++)
	{
		if (img[irow]> maximg )
		{
			maximg = img[irow];
		}
	}
	return maximg;
}
//end function: maxImage *******************************************************

/*** Get image dimensions from pgm file ****/
void getImgDims(char *imgName, int *width, int *height)
{
	FILE *pgmfile = NULL;
	char filename[MAX_BUFFERSIZE]={0};
	char buff1[MAX_BUFFERSIZE]={0};
	// Some PGM files contain a comment in the header so comment this out of the comment is present
	// char buff2[MAX_BUFFERSIZE]={0}; 

	sprintf(filename, imgName);
	pgmfile = fopen(filename,"r");

	if (pgmfile == NULL)
	{
		TRACE_INFO(("\nPGM file \"%s\" cannot be opened !\n",filename));
		exit(1);
	} 
	else
	{        
		fgets(buff1, MAX_BUFFERSIZE, pgmfile);
		// fgets(buff2, MAX_BUFFERSIZE, pgmfile);
		fscanf(pgmfile, "%d %d",width, height);
	}
}
//end function: getImgDims *****************************************************

/*** Write the result image ***/
void imgWrite(uint32_t *imgIn, char img_out_name[MAX_BUFFERSIZE], int height, int width)
{
	FILE *pgmfile_out = NULL;

	int irow = 0;
	int icol = 0;
	int maxval = 0;

	pgmfile_out = fopen(img_out_name, "wt");

	if (pgmfile_out == NULL) 
	{
		TRACE_INFO(("\nPGM file \"%s\" cannot be opened !\n", img_out_name));
		exit(1);
	}
	maxval = maxImage((uint32_t*)imgIn, height, width);
	if (maxval>MAX_BRIGHTNESS)
	{
		fprintf(pgmfile_out, "P2\n# CREATOR: smartcamera.c\n%d %d\n%d\n", width, height, maxval);
	}
	else
	{
		fprintf(pgmfile_out, "P2\n# CREATOR: smartcamera.c\n%d %d\n%d\n", width, height, MAX_BRIGHTNESS);
	}
	for (irow = 0; irow < height; irow++)
	{
		for (icol = 0; icol < width; icol++) 
		{
			fprintf(pgmfile_out, "%d\n", imgIn[irow*width+icol]);
		}
	}
	fclose(pgmfile_out);
}
//end function: imgWrite *******************************************************


/*** Allocation function for Classifier Cascade ***/
CvHaarClassifierCascade* allocCascade_continuous()
{
	int i = 0;
	int j = 0;
	int k = 0;

	CvHaarClassifierCascade *restrict cc;

	cc = (CvHaarClassifierCascade *)malloc(sizeof(CvHaarClassifierCascade)
					+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature));

/*
	cudaMallocHost((CvHaarClassifierCascade **)&cc, sizeof(CvHaarClassifierCascade)
					+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature));
*/

	memset(cc,0,sizeof(CvHaarClassifierCascade)
			+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
			+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
			+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature));

	cc->stageClassifier = (CvHaarStageClassifier*)(((char*)cc) + sizeof(CvHaarClassifierCascade));     


	for (i = 0; i < N_MAX_STAGES; i++)
	{
		cc->stageClassifier[i].classifier = (CvHaarClassifier*)(((char*)cc->stageClassifier) + (N_MAX_STAGES * sizeof(CvHaarStageClassifier)) + (i*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)));	

		
		for(j = 0; j < N_MAX_CLASSIFIERS; j++)
		{
			cc->stageClassifier[i].classifier[j].haarFeature = (CvHaarFeature*)(((char*)&(cc->stageClassifier[N_MAX_STAGES])) + (N_MAX_STAGES*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)) + (((i*N_MAX_CLASSIFIERS)+j)*sizeof(CvHaarFeature)));
			
			for (k = 0; k<2; k++)
			{
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.x0 = 0;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.y0 = 0;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.width = 1;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.height = 1;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].weight = 1.0;
			}
			cc->stageClassifier[i].classifier[j].threshold = 0.0;        
			cc->stageClassifier[i].classifier[j].left = 1.0;
			cc->stageClassifier[i].classifier[j].right = 1.0;
		}
		cc->stageClassifier[i].count = 1;
		cc->stageClassifier[i].threshold = 0.0;
	}
	return cc; 
}

/*** Allocation function for Classifier Cascade ***/
CvHaarClassifierCascade* allocCascade()
{
	int i = 0;
	int j = 0;
	int k = 0;

	CvHaarClassifierCascade *cc;

	cc = malloc(sizeof(CvHaarClassifierCascade));
	cc->stageClassifier = calloc(N_MAX_STAGES,sizeof(CvHaarStageClassifier));
	for (i = 0; i < N_MAX_STAGES; i++)
	{
		cc->stageClassifier[i].classifier = calloc(N_MAX_CLASSIFIERS,sizeof(CvHaarClassifier));
		for(j = 0; j < N_MAX_CLASSIFIERS; j++)
		{
			cc->stageClassifier[i].classifier[j].haarFeature = malloc(sizeof(CvHaarFeature));
			for (k = 0; k<N_RECTANGLES_MAX; k++)
			{
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.x0 = 0;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.y0 = 0;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.width = 1;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.height = 1;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].weight = 1.0;
			}
			cc->stageClassifier[i].classifier[j].threshold = 0.0;        
			cc->stageClassifier[i].classifier[j].left = 1.0;
			cc->stageClassifier[i].classifier[j].right = 1.0;
		}
		cc->stageClassifier[i].count = 1;
		cc->stageClassifier[i].threshold = 0.0;
	}
	return cc;
}
//end function: allocCascade ***************************************************

/*** Deallocation function for the whole Cascade ***/
void releaseCascade(CvHaarClassifierCascade *cc)
{
	int i = 0;
	int j = 0;

	for (i=0; i<N_MAX_STAGES; i++)
	{
		for (j=0; j<N_MAX_CLASSIFIERS; j++)
		{
			free(cc->stageClassifier[i].classifier[j].haarFeature);
		}
		free(cc->stageClassifier[i].classifier);
	}
	free(cc->stageClassifier);
	free(cc);
}
//end function: releaseCascade *************************************************

/*** Read classifier cascade file and build cascade structure***/
void readClassifCascade(char *haarFileName, CvHaarClassifierCascade *cascade, int *nRows, int *nCols, int *nStages)
{
	FILE *haarfile = NULL;

	char line[MAX_BUFFERSIZE] = {0};
	char linetag = 0;

	int x0 = 0, y0 = 0, wR = 0, hR = 0;    // rectangle coordinates
	int iStage = 0;
	int iNode=0;
	int nRectangles = 0;        
	int isRect = 0;

	float thresh = 0.0;
	float featThresh = 0.0;                        
	float weight = 0.0;
	float a = 0.0;
	float b = 0.0;

	haarfile = fopen(haarFileName, "r");
	if (haarfile == NULL) 
	{        
		TRACE_INFO(("\nFile \"%s\" cannot be opened !\n", haarFileName));
		exit(1);
	}
	else
	{
		fscanf(haarfile,"%d %d", nCols, nRows); 
		while (!feof(haarfile))
		{
			fgets(line, MAX_BUFFERSIZE, haarfile);
			linetag = line[0];
			if (isRect) 
			{
				nRectangles++;
			}
			else 
			{
				nRectangles = 1;
			}
			switch (linetag)
			{
				case 'S':
				{
//                Stage number index
					sscanf(line,"%*s %d", &iStage);
					isRect = 0;
					break;
				}
				case 'T':
				{
//                Stage threshold
					sscanf(line,"%*s %*d %f", &thresh);
					isRect = 0;
					assert(iStage<N_MAX_STAGES);
					cascade->stageClassifier[iStage].count = iNode+1;
					cascade->stageClassifier[iStage].threshold = thresh;
					break;
				}
				case 'N':
				{
//                Feature (node) index
					sscanf(line,"%*s %d",&iNode);
					break;
				}
				case 'R':
//                Rectangle feature; encoded as (left corner) column row width height weight
//                weight indicates the type of rectangle (sign(weight)<0 <=> white rectangle, else <=> black rectangle)
				{
					isRect = 1;
					sscanf(line,"%*s %d %d %d %d %f", &x0, &y0, &wR, &hR, &weight);
					assert(iNode<N_MAX_CLASSIFIERS);
				        assert(nRectangles-1<N_RECTANGLES_MAX);
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.x0 = x0;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.y0 = y0;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.width = wR;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.height = hR;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].weight = weight;
					break;
				}
				case 'a':
				{
					sscanf(line,"%*s %f", &a);
					assert(iStage<N_MAX_STAGES);
					assert(iNode<N_MAX_CLASSIFIERS);
					cascade->stageClassifier[iStage].classifier[iNode].left = a;
					break;
				}                                
				case 'b':
				{
					sscanf(line,"%*s %f", &b);
					assert(iStage<N_MAX_STAGES);
                                        assert(iNode<N_MAX_CLASSIFIERS);
					cascade->stageClassifier[iStage].classifier[iNode].right = b;
					break;
				}
				default:
				{
					isRect = 0;
					sscanf(line,"%f",&featThresh);
					assert(iStage<N_MAX_STAGES);
                                        assert(iNode<N_MAX_CLASSIFIERS);
					cascade->stageClassifier[iStage].classifier[iNode].threshold = featThresh;
				}
			}
		}        
		*nStages = iStage+1;
		assert(*nStages<N_MAX_STAGES);
		cascade->count = *nStages;
		cascade->orig_window_sizeR = *nRows;
		cascade->orig_window_sizeC = *nCols;
	}
	fclose(haarfile);
}
//end function: readClassifCascade *********************************************

/*** Pixel-wise square image ****/
void imgDotSquare(uint32_t *imgIn, uint32_t *imgOut, int height, int width)
{
	int irow = 0, icol = 0;

	#pragma acc parallel loop collapse(2) independent present(imgOut[0:width*height], imgIn[0:width*height])
	for (irow = 0; irow < height; irow++){
		for (icol = 0; icol < width; icol++){
			imgOut[irow*width+icol] = imgIn[irow*width+icol] * imgIn[irow*width+icol];
		}
	}
}
//end function: imgDotSquare ***************************************************

/*** Compute variance-normalized image ****/
void imgNormalize(uint32_t *imgIn, double *imgOut, double normFact, int height, int width)
{
	int irow = 0, icol = 0;
	int dim = 0;
	double meanImg = 0.0;

	dim = width*height;

	if(dim != 0)
	{
		meanImg = (imgIn[(height-1)*width+(width-1)])/dim;
	}

	if(normFact != 0)
	{
		for (irow = 0; irow < height; irow++)
		{
			for (icol = 0; icol < width; icol++)
			{
				imgOut[irow*width+icol] = (imgIn[irow*width+icol]- meanImg)/sqrt(normFact);
			}
		}
	}
}
//end function: imgNormalize ***************************************************
double aux_test = -1; 

/*** Cast int image as double ****/
// #pragma acc routine 
void imgCopy(uint32_t *imgIn, float *imgOut, int height, int width)
{
	int irow = 0, icol = 0;

	#pragma acc parallel loop collapse(2) independent present(imgIn[0:width*height], imgOut[0:width*height])
	for (irow = 0; irow < height; irow++){
		for (icol = 0; icol < width; icol++){
			// aux_test = (double)imgIn[irow*width+icol];
			imgOut[irow*width+icol]  = (float)imgIn[irow*width+icol]; 
		}
	}
}
//end function: imgCopy *******************************************************

/*** Copy one haarFeature into another ****/
void featCopy(CvHaarFeature *featSource, CvHaarFeature *featDest)
{
	int i = 0;

	for (i = 0; i < 3; i++)
	{
		featDest->rect[i].r.x0 = featSource->rect[i].r.x0;
		featDest->rect[i].r.y0 = featSource->rect[i].r.y0;
		featDest->rect[i].r.width = featSource->rect[i].r.width;
		featDest->rect[i].r.height = featSource->rect[i].r.height;        
		featDest->rect[i].weight = featSource->rect[i].weight;        
	}
}
//end function: featCopy *******************************************************

/*** Compute integral image ****/
void computeIntegralImg(uint32_t *imgIn, uint32_t *imgOut, int height, int width)
{
	int irow = 0, icol = 0;
	uint32_t row_sum = 0;

	for (irow = 0 ; irow < height; irow++)
	{
		row_sum = 0;
		for (icol = 0 ; icol < width; icol++)
		{
			row_sum += imgIn[irow*width+icol];
			if (irow > 0)
			{
				imgOut[irow*width+icol] = imgOut[(irow-1)*width+icol] + row_sum;
			}
			else
			{
				imgOut[irow*width+icol] = row_sum;
			}
		}
	}
}

void computeIntegralImgRowACC(uint32_t *imgIn, uint32_t *imgOut, int width, int height)
{	
	int irow, icol;

	#pragma acc parallel loop independent present(imgIn[0:width*height], imgOut[0:width*height])
	for (irow = 0 ; irow < height; irow++)
	{
		uint32_t row_sum=0;	
		#pragma acc loop seq 
		for(icol=0; icol<width; icol++)
		{
			row_sum += imgIn[irow*width+icol];
			imgOut[irow*width+icol] = row_sum; 
		}
	}
	
}


void computeIntegralImgColACC(uint32_t *imgOut, int width, int height)
{
	int icol = 0, irow;

	#pragma acc parallel loop independent present(imgOut[0:width*height])
	for (icol = 0 ; icol < width; icol++)
	{
		uint32_t col_sum=0;
		#pragma acc loop seq 
		for(irow=0; irow<height; irow++)
		{
				col_sum += imgOut[icol+irow*width];
				imgOut[icol+irow*width] = col_sum;
		}
	}
}

//end function: computeIntegralImg *********************************************

/*** Recover any pixel in the image by using the integral image ****/
#pragma acc routine seq 
 float getImgIntPixel(float *img, int row, int col, int real_height, int real_width)
{
	float pval = 0.0;

	if ((row == 0) && (col == 0))
	{
		pval = img[row*real_width+col];
		return pval;
	}
	if ((row > 0) && (col > 0))
	{
		pval = img[(row-1)*real_width+(col-1)] - img[(row-1)*real_width+col] - img[row*real_width+(col-1)] + img[row*real_width+col];
	}
	else
	{
		if (row == 0)
		{
			pval = img[col] - img[col-1];
		}
		else
		{
			if (col == 0)
			{
				pval = img[row*real_width] - img[(row-1)*real_width];
			}
		}
	}
	return pval;
}
//end function: getImgIntPixel *************************************************

/*** Compute any rectangle sum from integral image ****/
// #pragma acc routine seq 
float computeArea(float *img, int row, int col, int height, int width, int real_height, int real_width)
{
	float sum = 0.0;
	int cornerComb = 0;

  // rectangle = upper-left corner pixel of the image
	if ((row == 0) && (col == 0) && (width == 1) && (height == 1))
	{
		sum = img[0];
		return sum;
	}
  // rectangle = pixel anywhere in the image
	else
	{
		if ((width == 1) && (height == 1))
		{
			sum = getImgIntPixel((float *)img, row, col, real_height, real_width);
			return sum;        
		}
    // map upper-left corner of rectangle possible combinations        
		if ((row == 0) && (col == 0))
		{
			cornerComb = 1;
		}
		if ((row == 0) && (col > 0))
		{
			cornerComb = 2;
		}
		if ((row > 0) && (col == 0))
		{
			cornerComb = 3;
		}
		if ((row > 0) && (col > 0))
		{
			cornerComb = 4;
		}

		switch (cornerComb)
		{
			case 1:
			{
        // row = 0, col = 0
				sum = img[(row+height-1)*real_width+(col+width-1)];
				break;
			}
			case 2:
			{
        // row = 0, col > 0
				sum = (img[(row+height-1)*real_width+(col+width-1)] - img[(row+height-1)*real_width+(col-1)]);
				break;
			}
			case 3:
			{
        // row > 0, col = 0
				sum = (img[(row+height-1)*real_width+(col+width-1)] - img[(row-1)*real_width+(col+width-1)]);
				break;
			}
			case 4:
			{
        // row > 0, col > 0
				sum = (img[(row+height-1)*real_width+(col+width-1)] - img[(row-1)*real_width+(col+width-1)] - img[(row+height-1)*real_width+(col-1)] + img[(row-1)*real_width+(col-1)]);
				break;
			}
			default:
			{
				TRACE_INFO(("Error: \" This case is impossible!!!\"\n"));
				break;
			}
		}

		if(sum >= DBL_MAX-1)
		{
			sum = DBL_MAX;
		}
	}
	return sum;
}
//end function: computeArea ****************************************************

/*** Compute parameters for each rectangle in a feature: 
****        upper-left corner, width, height, sign       ****/
#pragma acc routine seq 	
void getRectangleParameters(CvHaarFeature *f, int iRectangle, int nRectangles, double scale, int rOffset, int cOffset, int *row, int *col, int *height, int *width)
{
	int r = 0, c = 0, h = 0, w = 0;

	w = f->rect[1].r.width;
	h = f->rect[1].r.height;

	if ((iRectangle > nRectangles) || (nRectangles < 2))
	{
		TRACE_INFO(("Problem with rectangle index %d/%d or number of rectangles.\n", iRectangle, nRectangles));
		return;
	}

  // get upper-left corner according to rectangle index in the feature (max 4-rectangle features)
	switch (iRectangle)
	{
		case 0: 
		{
			r = f->rect[0].r.y0;
			c = f->rect[0].r.x0;
			break;
		}
		case 1: 
		{
			switch (nRectangles)
			{
				case 2:
				{
					if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
					{
						if (f->rect[0].r.width == f->rect[1].r.width) 
						{
							r = f->rect[0].r.y0 + h;
							c = f->rect[0].r.x0;
						}
						else
						{
							r = f->rect[0].r.y0;
							c = f->rect[0].r.x0 + w;
						}
					}
					else
					{
						r = f->rect[1].r.y0;
						c = f->rect[1].r.x0;
					}
					break;
				}
				case 3:
				{
					r = f->rect[1].r.y0;
					c = f->rect[1].r.x0;
					break;
				}
				case 4:
				{
					if ((f->rect[0].r.x0 == f->rect[1].r.x0) &&  (f->rect[0].r.y0 == f->rect[1].r.y0))
					{
						r = f->rect[0].r.y0;
						c = f->rect[0].r.x0 + w;
					}
					else
					{
						r = f->rect[1].r.y0;
						c = f->rect[1].r.x0;
					}
					break;
				}
			}
			break;
		}
		case 2: 
		{
			if (nRectangles == 3) 
			{
				if (f->rect[0].r.x0 == f->rect[1].r.x0)
				{
					r = f->rect[1].r.y0 + h;
					c = f->rect[0].r.x0;
				}
				else
				{
					r = f->rect[0].r.y0;
					c = f->rect[1].r.x0 + w;
				}
			}
			else
			{
				if ((f->rect[0].r.x0 == f->rect[1].r.x0) &&  (f->rect[0].r.y0 == f->rect[1].r.y0))
				{
					r = f->rect[0].r.y0 + h;
					c = f->rect[0].r.x0;
				}
				else
				{
					r = f->rect[2].r.y0;
					c = f->rect[2].r.x0;
				}
			}
			break;
		}
		case 3: 
		{
			if ((f->rect[0].r.x0 == f->rect[1].r.x0) &&  (f->rect[0].r.y0 == f->rect[1].r.y0))
			{
				r = f->rect[2].r.y0;
				c = f->rect[2].r.x0;
			}
			else
			{
				r = f->rect[2].r.y0;
				c = f->rect[2].r.x0 + w;
			}
			break;
		}
	}

	*row = (int)(floor(r*scale)) + rOffset;
	*col = (int)(floor(c*scale)) + cOffset;
	*width = (int)(floor(w*scale));
	*height = (int)(floor(h*scale));
}
//end function: getRectangleParameters *****************************************

/*** Re-create feature structure from rectangle coordinates and feature type (test function!) ****/
void writeInFeature(int rowVect[4], int colVect[4], int hVect[4], int wVect[4], float weightVect[4], int nRects, CvHaarFeature *f_scaled)
{
	f_scaled->rect[1].r.width = wVect[1];
	f_scaled->rect[1].r.height = hVect[1];

	f_scaled->rect[0].r.x0 = colVect[0];
	f_scaled->rect[0].r.y0 = rowVect[0];

	switch (nRects)
	{
		case 2:
		{
			f_scaled->rect[0].weight = -1.0;
			f_scaled->rect[1].weight = 2.0;
			f_scaled->rect[2].weight = 0.0;

			if ((weightVect[0] == 2.0) && (weightVect[2] == 0.0))
			{
				f_scaled->rect[1].r.x0 = colVect[0];
				f_scaled->rect[1].r.y0 = rowVect[0];
			}
			else
			{
				f_scaled->rect[1].r.x0 = colVect[1];
				f_scaled->rect[1].r.y0 = rowVect[1];
			}
			if (rowVect[0] == rowVect[1])
			{
				f_scaled->rect[0].r.width = wVect[1] * 2;
				f_scaled->rect[0].r.height = hVect[1];
			}
			else
			{
				f_scaled->rect[0].r.width = wVect[1];
				f_scaled->rect[0].r.height = hVect[1] * 2;
			} 
			break;
		}
		case 3:
		{
			f_scaled->rect[0].weight = -1.0;
			f_scaled->rect[1].weight = 3.0;
			f_scaled->rect[2].weight = 0.0;

			if (rowVect[0] == rowVect[1])
			{
				f_scaled->rect[0].r.width = wVect[1] * 3;
				f_scaled->rect[0].r.height = hVect[1];
			}
			else
			{
				f_scaled->rect[0].r.width = wVect[1];
				f_scaled->rect[0].r.height = hVect[1] * 3;
			} 
			f_scaled->rect[1].r.x0 = colVect[1];
			f_scaled->rect[1].r.y0 = rowVect[1];
			break;
		}
		case 4:
		{
			f_scaled->rect[0].weight = -1.0;
			f_scaled->rect[1].weight = 2.0;
			f_scaled->rect[2].weight = 2.0;

			f_scaled->rect[0].r.width = wVect[1]*2;
			f_scaled->rect[0].r.height = hVect[1]*2;

			if (weightVect[0] == 2.0)
			{
				f_scaled->rect[1].r.x0 = colVect[0];
				f_scaled->rect[1].r.y0 = rowVect[0];
				f_scaled->rect[2].r.x0 = colVect[3];
				f_scaled->rect[2].r.y0 = rowVect[3];
			}
			else
			{
				f_scaled->rect[1].r.x0 = colVect[1];
				f_scaled->rect[1].r.y0 = rowVect[1];
				f_scaled->rect[2].r.x0 = colVect[2];
				f_scaled->rect[2].r.y0 = rowVect[2];
			}

			f_scaled->rect[2].r.width = wVect[1];
			f_scaled->rect[2].r.height = hVect[1];
			break;
		}
	}
}
//end function: writeInFeature *************************************************

/*** Compute feature value (this is the core function!) ****/

#pragma acc routine seq 
// void computeFeature(double *img, double *imgSq, CvHaarFeature *f, double *featVal, int irow, int icol, int height, int width, double scale, float scale_correction_factor, CvHaarFeature *f_scaled, int real_height, int real_width)
void computeFeature(float *img, float *imgSq, CvHaarFeature *f, float *featVal, int irow, int icol, int height, int width, double scale, float scale_correction_factor, int real_height, int real_width)
{
	int nRects = 0;
	int col = 0;
	int row = 0;
	int wRect = 0;
	int hRect = 0;
	int i = 0;
	int colVect[4] = {0};
	int rowVect[4] = {0};
	int wVect[4] = {0};
	int hVect[4] = {0};

	float w1 = 0.0;
	float rectWeight[4] = {0};

	float val = 0.0;
	float s[N_RECTANGLES_MAX] = {0};

	*featVal = 0.0;

	w1 = f->rect[0].weight * scale_correction_factor;

  // Determine feature type (number of rectangles) according to weight
	if (f->rect[2].weight == 2.0)
	{
		nRects = 4;
		if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
		{
			rectWeight[0] = -w1; 
			rectWeight[1] = w1;
			rectWeight[2] = w1;
			rectWeight[3] = -w1; 
		}
		else
		{
			rectWeight[0] = w1;
			rectWeight[1] = -w1; 
			rectWeight[2] = -w1; 
			rectWeight[3] = w1; 
		}
	}
	else 
	{ 
		if (f->rect[1].weight == 2.0)
		{
			nRects = 2;
			if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
			{
				rectWeight[0] = -w1;
				rectWeight[1] = w1;
			}
			else
			{
				rectWeight[0] = w1;
				rectWeight[1] = -w1;
			}
			rectWeight[2] = 0.0;
			rectWeight[3] = 0.0;
		}
		else
		{
			nRects = 3;
			rectWeight[0] = w1;
			rectWeight[1] = -2.0*w1;
			rectWeight[2] = w1;
			rectWeight[3] = 0.0;
		}
	}
	#pragma acc loop seq private(row, col, hRect, wRect) 
	for (i = 0; i<nRects; i++)
	{
		s[i] = 0.0; 
		getRectangleParameters(f, i, nRects, scale, irow, icol, &row, &col, &hRect, &wRect);
		s[i] = computeArea((float *)img, row, col, hRect, wRect, real_height, real_width);

		if (fabs(rectWeight[i]) > 0.0)
		{
			val += rectWeight[i]*s[i];
		}
    // test values for each rectangle
		rowVect[i] = row; colVect[i] = col; hVect[i] = hRect; wVect[i] = wRect;
	}
	*featVal = val;
	// writeInFeature(rowVect,colVect,hVect,wVect,rectWeight,nRects,f_scaled);
}
//end function: computeFeature *************************************************

/*** Calculate the Variance ****/
#pragma acc routine seq 
float computeVariance(float *img, float *imgSq, int irow, int icol, int height, int width, int real_height, int real_width)
{
	int nPoints = 0;

	float s1 = 0.0;
	float s2 = 0.0;
	float f1 = 0.0;
	float f2 = 0.0;
	float varFact = 0.0;

	nPoints = height*width;

	s1 = (float)computeArea((float *)img, irow, icol, height, width, real_height, real_width);
	s2 = (float)computeArea((float *)imgSq, irow, icol, height, width, real_height, real_width);

	if(nPoints != 0)
	{
		f1 = (float)(s1/nPoints);
		f2 = (float)(s2/nPoints);
	}

	if(f1*f1 > f2)
	{
		varFact = 0.0;
	}
	else
	{
		varFact = f2 - f1*f1;
	}

	return varFact;
}
//end function: computeVariance ************************************************

/*** Allocate one dimension integer pointer ****/
uint32_t *alloc_1d_uint32_t(int n)
{
	uint32_t *new;

	new = (uint32_t *) malloc ((unsigned) (n * sizeof (uint32_t)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_1D_UINT_32T: Couldn't allocate array of integer\n"));
		return (NULL);
	}
	return (new);
}
//end function: alloc_1d_uint32_t **********************************************

/*** Allocate one dimension double pointer ****/
double *alloc_1d_double(int n)
{
	double *new;

	new = (double *) malloc ((unsigned) (n * sizeof (double)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_1D_DOUBLE: Couldn't allocate array of integer\n"));
		return (NULL);
	}
	return (new);
}
//end function: alloc_1d_double ************************************************

/*** Allocate one dimension float pointer ****/
float *alloc_1d_float(int n)
{
	float *new;

	new = (float *) malloc ((unsigned) (n * sizeof (float)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_1D_FLOAT: Couldn't allocate array of integer\n"));
		return (NULL);
	}
	return (new);
}
//end function: alloc_1d_float ************************************************

/*** Allocate 2d array of integers ***/
uint32_t **alloc_2d_uint32_t(int m, int n)
{
	int i;
	uint32_t **new;

	new = (uint32_t **) malloc ((unsigned) (m * sizeof (uint32_t *)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_2D_UINT_32T: Couldn't allocate array of integer ptrs\n"));
		return (NULL);
	}

	for (i = 0; i < m; i++) {
		new[i] = alloc_1d_uint32_t(n);
	}

	return (new);
}
//end function: alloc_2d_uint32_t **********************************************

/* Draws simple or filled square */
void raster_rectangle(uint32_t* img, int x0, int y0, int radius, int real_width)
{
	int i=0;

	#pragma acc loop independent
	for(i=-radius/2; i<radius/2; i++)
	{
		assert((i + x0 + (y0 + (int)(radius)) * real_width)>0);
		assert((i + x0 + (y0 + (int)(radius)) * real_width)<(480*640));
		assert((i + x0 + (y0 - (int)(radius)) * real_width)>0);
		assert((i + x0 + (y0 - (int)(radius)) * real_width)<(480*640));
		img[i + x0 + (y0 + (int)(radius)) * real_width]=255;
		img[i + x0 + (y0 - (int)(radius)) * real_width]=255;
	}
	#pragma acc loop independent
	for(i=-(int)(radius); i<(int)(radius); i++)
	{
		assert(((x0 + (int)(radius/2)) + (y0+i) * real_width)>0);
		assert(((x0 + (int)(radius/2)) + (y0+i) * real_width)<(480*640));
		assert(((x0 - (int)(radius/2)) + (y0+i) * real_width)>0);
		assert(((x0 - (int)(radius/2)) + (y0+i) * real_width)<(480*640));
		img[(x0 + (int)(radius/2)) + (y0+i) * real_width]=255;
		img[(x0 - (int)(radius/2)) + (y0+i) * real_width]=255;
	}	
}
//end function: raster_rectangle **************************************************


// Loop ending condition
char InProcessingLoop() 
{
	#if CONTINUOUS_PROC
		const char bExitCond = 1;
		return bExitCond;
	#else
		return 0;
	#endif
}


/* ********************************** MAIN ********************************** */
int main( int argc, char** argv )
{
	// Timer declaration 
	time_t start, end, frame_start, frame_end;

	// Pointer declaration
	CvHaarClassifierCascade *restrict cascade = NULL;
	CvHaarClassifierCascade *restrict cascade_scaled = NULL;
	CvHaarFeature *restrict feature_scaled = NULL;

	char *imgName = NULL;
	char *haarFileName = NULL;
	char result_name[MAX_BUFFERSIZE]={0};

	uint32_t *img = NULL;
	uint32_t *imgInt = NULL;
	uint32_t *imgSq = NULL;
	uint32_t *imgSqInt = NULL;
	uint32_t **result2 = NULL;


	uint32_t **goodcenterX=NULL;
	uint32_t **goodcenterY=NULL;
	uint32_t **goodRadius=NULL;
	uint32_t *nb_obj_found2=NULL;

	uint32_t *position= NULL;
	uint32_t *goodPoints = NULL;

	// Counter Declaration 
	int rowStep = 1;
	int colStep = 1;
	int width = 0;
	int height = 0;
	int detSizeR = 0;  
	int detSizeC = 0; 
	int tileWidth = 0;
	int tileHeight = 0;
	int irow = 0;
	int icol = 0;
	int nStages = 0; 
	int foundObj = 0;
	int nTileRows = 0;
	int nTileCols = 0;
	int iStage = 0;
	int iNode = 0;
	int nNodes = 0;
	int i = 0, j = 0;
	
	
	int real_height = 0, real_width = 0;
	int scale_index_found=0;
	int threshold_X=0;
	int threshold_Y=0;
	// int nb_obj_found=0;
	int * nb_obj_found= NULL;


	int count = 0;
	int offset_X = 0, offset_Y = 0;

	// Threshold Declaration 
	float thresh = 0.0;
	float a = 0.0;
	float b = 0.0;
	float scale_correction_factor = 0.0;
	
	// /**! Brief: For the circle calculation */
	float centerX_tmp=0.0;
	float centerY_tmp=0.0;
	float radius_tmp=0.0;
	float centerX=0.0;
	float centerY=0.0;
	float radius=0.0;

	// Factor Declaration 
	double scaleFactorMax = 0.0;
	double scaleStep = 1.1; // 10% increment of detector size per scale. Change this value to test other increments 
	double scaleFactor = 0.0;
	double varFact = 0.0;
	double featVal = 0.0;
	double detectionTime = 0.0;

	// Integral Image Declaration 
	float *restrict imgInt_f = NULL;
	float *restrict imgSqInt_f = NULL;

	// ACC declarations 
	float *restrict goodcenterX_tmp=NULL;
	float *restrict goodcenterY_tmp=NULL;
	uint32_t **restrict goodRadius_tmp=NULL;

	int * foundObj_test=NULL; 
	int * foundObj_test2=NULL; 
	int ** index_found=NULL; 

	if (argc <= 2)
	{
		TRACE_INFO(("Usage: %s classifier image1 image2 ...\n", argv[0]));
		return(0);
	}

	// Get the Image name and the Cascade file name from the console 
	haarFileName=argv[1];
	imgName=argv[2];


	// Get the Input Image informations 
	getImgDims(imgName, &width, &height);

	// Get back the Real Size of the Input Image 
	real_height=height;
	real_width=width;

	// Allocate the Cascade Memory 
	cascade = allocCascade_continuous();
	cascade_scaled = allocCascade_continuous();
	feature_scaled = malloc(sizeof(CvHaarFeature));

	// Get the Classifier informations 
	readClassifCascade(haarFileName, cascade, &detSizeR, &detSizeC, &nStages);

	TRACE_INFO(("detSizeR = %d\n", detSizeR));
	TRACE_INFO(("detSizeC = %d\n", detSizeC));

	// Determine the Max Scale Factor
	if (detSizeR != 0 && detSizeC != 0)
	{
		scaleFactorMax = min((int)floor(height/detSizeR), (int)floor(width/detSizeC));
	}

	// Give the Allocation size 
	img=alloc_1d_uint32_t(width*height);
	imgInt=alloc_1d_uint32_t(width*height);
	imgSq=alloc_1d_uint32_t(width*height);
	imgSqInt=alloc_1d_uint32_t(width*height);
	result2=alloc_2d_uint32_t(N_MAX_STAGES, width*height);
	position=alloc_1d_uint32_t(width*height);

	TRACE_INFO(("nStages: %d\n", nStages));
	TRACE_INFO(("NB_MAX_DETECTION: %d\n", NB_MAX_DETECTION));
	
	
	//nstages
	goodcenterX=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	goodcenterY=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	goodRadius=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	nb_obj_found2=alloc_1d_uint32_t(N_MAX_STAGES);
	nb_obj_found=alloc_1d_uint32_t(N_MAX_STAGES);

	goodcenterX_tmp=(float*)malloc(N_MAX_STAGES*sizeof(float)*NB_MAX_DETECTION);
	goodcenterY_tmp=(float*)malloc(N_MAX_STAGES*sizeof(float)*NB_MAX_DETECTION);
	goodRadius_tmp=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	foundObj_test=alloc_1d_uint32_t(N_MAX_STAGES*sizeof(int));
	foundObj_test2=alloc_1d_uint32_t(N_MAX_STAGES*sizeof(int));
	index_found=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);

	goodPoints=alloc_1d_uint32_t(width*height);
	imgInt_f=alloc_1d_float(width*height);
	imgSqInt_f=alloc_1d_float(width*height);

	TRACE_INFO(("Allocations finished!\n"));

	int total_scales=0; 

	for (scaleFactor = 1; scaleFactor <= scaleFactorMax; scaleFactor *= scaleStep)
		total_scales++;

	frame_start = clock();
	
	do // Infinite loop
	{
	for(int image_counter=0; image_counter < argc-2; image_counter++)
	{
	// int image_counter=1;
	// Task 1: Start Frame acquisition.
	// start = clock(); 
    start= clock(); 
    nNodes = 0;
    foundObj = 0;
    scale_index_found=0;
    varFact = 0.f;
    featVal = 0.f;

    memset(position, 0, width*height*sizeof (uint32_t));
    memset(nb_obj_found2, 0, N_MAX_STAGES*sizeof (uint32_t));
    memset(nb_obj_found, 0, N_MAX_STAGES*sizeof (uint32_t));
    memset(goodPoints, 0, width*height*sizeof (uint32_t));

    memset(goodcenterX_tmp, 0, NB_MAX_DETECTION*NB_MAX_DETECTION*sizeof(float));
    memset(goodcenterX_tmp, 0, NB_MAX_DETECTION*NB_MAX_DETECTION*sizeof(float));

    memset(foundObj_test, 0, N_MAX_STAGES*sizeof (uint32_t));
    memset(foundObj_test2, 0, N_MAX_STAGES*sizeof (uint32_t));

    for(int xx = 0; xx < N_MAX_STAGES; xx++){
        for(int yy = 0; yy < NB_MAX_DETECTION; yy++)
        {
            goodcenterX[xx][yy] = 0;
            goodcenterY[xx][yy] = 0;
            goodRadius[xx][yy] = 0;
        }
    }

    imgName=argv[image_counter+2];
    // load the Image in Memory 
    load_image_check((uint32_t *)img, (char *)imgName, width, height);
    printf("Load Image Done.\n");
    TRACE_INFO(("Load Image Done %s!\n", imgName));

    // Compute the Interal Image 
    computeIntegralImgRowACC((uint32_t*)img, (uint32_t*)imgInt, width, height);
    computeIntegralImgColACC((uint32_t*)imgInt, width, height);
    // computeIntegralImg((uint32_t *)img, (uint32_t *)imgInt, height, width);

    // Calculate the Image square 
    imgDotSquare((uint32_t *)img, (uint32_t *)imgSq, height, width);
    /* Compute the Integral Image square */
    // computeIntegralImg((uint32_t *)imgSq, (uint32_t *)imgSqInt, height, width);
    computeIntegralImgRowACC((uint32_t*)imgSq, (uint32_t*)imgSqInt, width, height);
    computeIntegralImgColACC((uint32_t*)imgSqInt, width, height);

    // Copy the Image to float array 
    imgCopy((uint32_t *)imgInt, (float *)imgInt_f, height, width);
    imgCopy((uint32_t *)imgSqInt, (float *)imgSqInt_f, height, width);

    TRACE_INFO(("Done with integral image\n"));

    TRACE_INFO(("scaleFactorMax = %f\n", scaleFactorMax));
    TRACE_INFO(("scaleStep = %f\n", scaleStep));
    TRACE_INFO(("nStages = %d\n", nStages));

	// Task 1: End Frame acquisition.
	int tmp=0; 

	// Task 2: Start Frame processing.

	int const cascade_size = N_MAX_STAGES + N_MAX_STAGES * N_MAX_CLASSIFIERS + N_MAX_STAGES * N_MAX_CLASSIFIERS;
	int const index_f_size = N_MAX_STAGES*NB_MAX_DETECTION, gR_size = total_scales*NB_MAX_DETECTION;


	// Launch the Main Loop 
	#pragma acc parallel loop seq independent firstprivate(detSizeC, detSizeR, height, width) 
	for (int s = 0; s < total_scales; s++){
		const float scaleFactor = (float) powf(scaleStep, (float)s);
	
		//TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
		const int tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
		const int tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);
		const int rowStep = max(2, (int)floor(scaleFactor + 0.5));
		const int colStep = max(2, (int)floor(scaleFactor + 0.5));

		//(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
		const float scale_correction_factor = (float)1.0/(float)((int)floor((detSizeC-2)*scaleFactor)*(int)floor((detSizeR-2)*scaleFactor)); 

		// compute number of tiles: difference between the size of the Image and the size of the Detector 
		const int nTileRows = height-tileHeight;
		const int nTileCols = width-tileWidth;

        float varFact, featVal; 
		// Operation used for every Stage of the Classifier 
		#pragma acc loop collapse(2) independent  private(varFact, featVal) firstprivate(nTileRows, nTileCols, rowStep, colStep)
		for (int irow = 0; irow < nTileRows; irow+=rowStep){
			for (int icol = 0; icol < nTileCols; icol+=colStep){
				int goodPoints_value = 255;
				/* Operation used for every Stage of the Classifier */
				if (goodPoints_value)
				{
					varFact=computeVariance((float *)imgInt_f, (float *)imgSqInt_f, irow, icol, tileHeight, tileWidth, real_height, real_width);

					if (varFact < 10e-15)
					{
						// this should not occur (possible overflow BUG)
						varFact = 1.0; 
						goodPoints_value = 0; 
						continue;
					}
					else
					{
						// Get the standard deviation 
						varFact = sqrt(varFact);
					}
				}

				#pragma acc loop seq independent
				for (int iStage = 0; iStage < nStages; iStage++)
				{
					int nNodes = cascade->stageClassifier[iStage].count;
					if (goodPoints_value)
					{	
						float sumClassif = 0.0;

						#pragma acc loop seq 
						for (int iNode = 0; iNode < nNodes; iNode++)
						{
							computeFeature((float *)imgInt_f, (float *)imgSqInt_f, cascade->stageClassifier[iStage].classifier[iNode].haarFeature,
											&featVal, irow, icol, tileHeight, tileWidth, scaleFactor, scale_correction_factor, real_height, real_width);
							// Get the thresholds for every Node of the stage 
							float thresh = cascade->stageClassifier[iStage].classifier[iNode].threshold;
							float a = cascade->stageClassifier[iStage].classifier[iNode].left;
							float b = cascade->stageClassifier[iStage].classifier[iNode].right;
							sumClassif += (float)(featVal < (thresh*varFact) ? a : b);
						}
						// Update goodPoints according to detection threshold 
						if (sumClassif < cascade->stageClassifier[iStage].threshold){
							goodPoints_value = 0;
						}else{	
							if (iStage == nStages - 1 )
							{ 
								int actu = 0; 

								float centerX=(((tileWidth-1)*0.5+icol));
								float centerY=(((tileHeight-1)*0.5+irow));
								float radius = sqrt(pow(tileHeight-1, 2)+pow(tileWidth-1, 2))/2;

								#pragma acc atomic capture 
								{
									actu=foundObj_test[s];	//solo funciona si esta init + en firstprivate / private ?
									foundObj_test[s]++;
								}

								goodcenterX_tmp[s*NB_MAX_DETECTION+actu]=centerX;
								goodcenterY_tmp[s*NB_MAX_DETECTION+actu]=centerY;
								goodRadius_tmp[s][actu]=radius;

								#pragma acc atomic update 
								nb_obj_found2[s]+=1; 
							}
						}	  
					}
				} // FOR STAGES
			}	// FOR COL
		}	// FOR ROW 
	} // FOR TOTAL SCALES 


	#pragma acc parallel loop reduction(max:scale_index_found) 
	for (int s = 0; s < total_scales; s++){
		
		const float scaleFactor = (float) powf(scaleStep, (float)s);
		const int tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
		const int tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);

		#pragma acc loop seq 
		for (int j=0; j < nb_obj_found2[s]; j++){

			float centerX = goodcenterX_tmp[s*NB_MAX_DETECTION+j];
			float centerY = goodcenterY_tmp[s*NB_MAX_DETECTION+j];

			int threshold_X=(int)((tileHeight-1)/(2*scaleFactor));
			int threshold_Y=(int)((tileWidth-1)/(2*scaleFactor));

			// printf("_________Evaluando[%d][%d] : centerX : %f vs centerX_tmp : %f\n", s,j, centerX, centerX_tmp); 

			if(centerX > (centerX_tmp+threshold_X) || centerX < (centerX_tmp-threshold_X) || (centerY > centerY_tmp+threshold_Y) || centerY < (centerY_tmp-threshold_Y))
			{
				int priv_indx = 0; 
				centerX_tmp = centerX;
				centerY_tmp = centerY;
				radius_tmp = goodRadius_tmp[s][j];

				#pragma acc atomic capture
				{
					priv_indx=nb_obj_found[s];
					nb_obj_found[s]+=1;	
				}

				// printf("Store values[%d] ::  %f %f at %d %d\n\n", s, centerX_tmp, centerY_tmp, s, priv_indx );
				goodcenterX[s][priv_indx]=centerX_tmp;
				goodcenterY[s][priv_indx]=centerY_tmp;
				goodRadius[s][priv_indx]=radius_tmp;

				// nb_obj_found2[s]++;

				#pragma acc atomic write 
				foundObj_test2[s]=s+1;

			}
		}

		scale_index_found = max(scale_index_found, foundObj_test2[s]);
	}
	scale_index_found++; 

	// Task 2: End Frame processing.
	// Task 3: Start Frame post-processing.

		// Multi-scale fusion and detection display (note: only a simple fusion scheme implemented
		int cnt_scale = 0, max_scale=0; 
		
		// Multi-scale fusion and detection display (note: only a simple fusion scheme implemented
		#pragma acc parallel loop reduction(+:cnt_scale) reduction(max: max_scale) present(nb_obj_found2[0:total_scales]) 
		for(i=0; i<scale_index_found; i++)
		{
			max_scale = max(0, nb_obj_found2[i]);
			if(nb_obj_found2[i])
				cnt_scale++; 
		}
		nb_obj_found2[scale_index_found] = max(max_scale, nb_obj_found2[scale_index_found-1]);

		
		// Keep the position of each circle from the bigger to the smaller 
		#pragma acc parallel private(offset_X, offset_Y) copy(result2[0:real_height][0:real_width]) create(position[0:NB_MAX_POINTS*NB_MAX_POINTS]) 
		{
		#pragma acc loop seq 
		for(i=scale_index_found; i>=0; i--)
		{
			#pragma acc loop seq 
			for(j=0; j<nb_obj_found2[scale_index_found]; j++)
			{
				// Normally if (goodcenterX=0 so goodcenterY=0) or (goodcenterY=0 so goodcenterX=0) 
				if(goodcenterX[i][j] !=0 || goodcenterY[i][j] !=0)
				{
					position[count]=goodcenterX[i][j];
					position[count+1]=goodcenterY[i][j];
					position[count+2]=goodRadius[i][j];
					count=count+3;
				}
			}
		}
		
		// Create the offset for X and Y 
		offset_X=(int)(real_width/(float)(cnt_scale*1.2));
		offset_Y=(int)(real_height/(float)(cnt_scale*1.2));

		// Delete detections which are too close 
		#pragma acc loop independent
		for(i=0; i<NB_MAX_POINTS; i+=3){
			#pragma acc loop independent
			for(j=3; j<NB_MAX_POINTS-i; j+=3){
				
				if(position[i] != 0 && position[i+j] != 0 && position[i+1] != 0 && position[i+j+1] != 0)
				{
					if(offset_X >= abs(position[i]-position[i+j]) && offset_Y >= abs(position[i+1]-position[i+j+1]))
					{
						position[i+j] = 0;
						position[i+j+1] = 0;
						position[i+j+2] = 0;
					}
				}
			}
		}	


		int finalNb = 0;
		
		#pragma acc loop reduction(+:finalNb)
		for(i=0; i<NB_MAX_POINTS; i+=3)
		{
			if (position[i]!=0) 
			{
				finalNb++;
				#if PRINT_OUTPUT
					printf("x:%d, y:%d, scale:%d, ValidOutput:%d\n", (int)position[i], (int)position[i+1], (int)(position[i+2]/2), finalNb);
					
				#endif
			}
		}

		}


		#if WRITE_IMG
        // Write the final result of the detection application 
		sprintf(result_name, "result_%d.pgm", image_counter);
		TRACE_INFO(("%s\n", result_name));
    
		// Draw detection
		for(i=0; i<NB_MAX_POINTS; i+=3)
		{
			if(position[i] != 0 && position[i+1] != 0 && position[i+2] != 0)
			{
				raster_rectangle(result2[scale_index_found], (int)position[i], (int)position[i+1], (int)(position[i+2]/2), real_width);
			}
			
		}


		// Re-build the result image with highlighted detections
            for(i=0; i<real_height; i++){
                for(j=0; j<real_width; j++)
                {
                    if(result2[scale_index_found][i*real_width+j]!= 255)
                    {
                        result2[scale_index_found][i*real_width+j] = img[i*real_width+j];
                    }
                }
            }
            
            imgWrite((uint32_t *)result2[scale_index_found], result_name, height, width);
		#endif

		TRACE_INFO(("END\n"));


		
	// Task 3: End Frame post-processing.
	
	// end = clock();
	// detectionTime = (double)(end-start)/CLOCKS_PER_SEC * 1000;
	// printf("\nTASK 3 : Execution time = %f ms.\n\n", detectionTime);
	
	} //for of all images
	} while(InProcessingLoop());

	#pragma acc wait
	frame_end = clock();
	const float frame_time = (double)((frame_end-frame_start))/CLOCKS_PER_SEC * 1000;
	printf("\nExecution time = %f for %d FRAMES ms.\n", frame_time, (argc-2));

	// FREE ALL the allocations 
	releaseCascade(cascade);
	releaseCascade(cascade_scaled);
	free(feature_scaled);
	free(img);
	free(imgInt);
	free(imgSq);
	free(imgSqInt);
	free(result2);

	free(goodcenterX);
	free(goodcenterY);
	free(goodRadius);
	free(nb_obj_found2);
	free(goodPoints);
	free(imgInt_f);
	free(imgSqInt_f);

	return 0;
}
