/* ## sin duplicar items 
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
*                       (Haar-like features, AdaBoost algorithm)
*                - learning data transcripted from OpenCV generated file
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
#include <cmath>

/* Static Library */
#include "violajones.h"

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

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

/* ********************************** SYCL DEVICE SELECTOR ********************************** */

class my_device_selector : public sycl::device_selector {
        public:
        int operator()(const sycl::device& dev) const override {
//              #ifdef GPU
                if ( dev.has(sycl::aspect::gpu)) {
                        return 1;
                }else {
                        return -1;
                }
//              #else
//              if ( dev.has(sycl::aspect::cpu)) {
//                      return 1;
//              }else {
//                      return -1;
//              }
//              #endif
                return -1;
        }
};
auto myQueue = sycl::queue{sycl::gpu_selector{}}; //{my_device_selector{}};
auto myQueue1 = sycl::queue{sycl::gpu_selector{}}; 

/* ********************************** FUNCTIONS ********************************** */

/*** Recover any pixel in the image by using the integral image ****/
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
                                // TRACE_INFO(("Error: \" This case is impossible!!!\"\n"));
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

void getRectangleParameters(CvHaarFeature *f, int iRectangle, int nRectangles, double scale, int rOffset, int cOffset, int *row, int *col, int *height, int *width)
{
        int r = 0, c = 0, h = 0, w = 0;

        w = f->rect[1].r.width;
        h = f->rect[1].r.height;

        if ((iRectangle > nRectangles) || (nRectangles < 2))
        {
                // TRACE_INFO(("Problem with rectangle index %d/%d or number of rectangles.\n", iRectangle, nRectangles));
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

float computeFeature(float *img, CvHaarFeature *f, float featVal, int irow, int icol, int height, int width, float scale, float scale_correction_factor, int real_height, int real_width)
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

//      *featVal = 0.0;

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
//      *featVal = val;
    return val;
//      writeInFeature(rowVect,colVect,hVect,wVect,rectWeight,nRects,f_scaled);
}
//end function: computeFeature *************************************************

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

        sprintf(filename, "%s", imgName);
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

        CvHaarClassifierCascade * cc;

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

        cc = (CvHaarClassifierCascade*) malloc(sizeof(CvHaarClassifierCascade));
        cc->stageClassifier = (CvHaarStageClassifier*) calloc(N_MAX_STAGES,sizeof(CvHaarStageClassifier));
        for (i = 0; i < N_MAX_STAGES; i++)
        {
                cc->stageClassifier[i].classifier = (CvHaarClassifier*) calloc(N_MAX_CLASSIFIERS,sizeof(CvHaarClassifier));
                for(j = 0; j < N_MAX_CLASSIFIERS; j++)
                {
                        cc->stageClassifier[i].classifier[j].haarFeature = (CvHaarFeature*) malloc(sizeof(CvHaarFeature));
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

void releaseCascade_continuous(CvHaarClassifierCascade *cc)
{
        free(cc);
        //cudaFreeHost(cc);
}

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
//      int irow = 0, icol = 0;

        for (int irow = 0; irow < height; irow++)
        {
                for (int icol = 0; icol < width; icol++)
                {
                        imgOut[irow*width+icol] = imgIn[irow*width+icol] * imgIn[irow*width+icol];
                }
        }
}

void imgDotSquareSYCL(uint32_t *imgIn, uint32_t *imgOut, unsigned int height, unsigned int width, sycl::id<1> idx)
{
//      int irow = 0, icol = 0;



                                int i = idx[0];

                                imgOut[i] = imgIn[i] * imgIn[i];
                        // });

}
//end function: imgDotSquare ***************************************************

void computeSquareImageSYCL_rows(uint32_t *imgIn, uint32_t *imgOut, unsigned int width, int height)
{

        myQueue.parallel_for<class computeSquareImageSYCL_rows>(
                        sycl::range<1>{width},[=]\
                        (sycl::id<1> idx){
                                int row = idx[0];

                                if (row<height)
                                {
                                        for(int i=0; i<width; i++)
                                                imgOut[row*width+i] = imgIn[row*width+i] * imgIn[row*width+i];
                                }

                        });
}

void computeSquareImageSYCL_cols(uint32_t *imgIn, uint32_t *imgOut, unsigned int height, int width)
{
        myQueue.parallel_for<class computeSquareImageSYCL_cols>(
                        sycl::range<1>{height},[=]\
                        (sycl::id<1> idx){
                                int col = idx[0];

                        if(col<width)
                                for(int i=0; i<height; i++)
                                                imgOut[col+width*i] = imgIn[col+width*i] * imgIn[col+width*i];

                        });
}


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

/*** Cast int image as double ****/
// void imgCopy(uint32_t *imgIn, double *imgOut, int height, int width)
// {
//      int irow = 0, icol = 0;

//      for (irow = 0; irow < height; irow++)
//      {
//              for (icol = 0; icol < width; icol++)
//              {
//                      imgOut[irow*width+icol] = (double)imgIn[irow*width+icol];
//              }
//      }
// }
void imgCopySYCL(uint32_t *imgIn, float *imgOut, unsigned int height, unsigned int width, sycl::nd_item<3> idx)
{
        // int i = idx[0];
        int i = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);

        if(i < height*width)
        {
                imgOut[i] = (float)imgIn[i];
        }

}

void initMatrSYCL(float *dev_goodcenterX_tmp,  float* dev_goodcenterY_tmp, uint32_t *dev_goodRadius_tmp, float *dev_goodcenterX,  float* dev_goodcenterY, uint32_t *dev_goodRadius, unsigned int N, unsigned int M)
{
//      int irow = 0, icol = 0;


        myQueue.parallel_for<class initMatr>(
                        sycl::range<1>{N*M},[=]\
                        (sycl::id<1> idx){
                                int i = idx[0];

                                dev_goodcenterX_tmp[i] = 0;
                                dev_goodcenterY_tmp[i] = 0;
                                dev_goodRadius_tmp[i] = 0;

                                dev_goodcenterX[i] = 0;
                                dev_goodcenterY[i] = 0;
                                dev_goodRadius[i] = 0;

                        });

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

void computeIntegralImgRowSYCL(uint32_t *imgIn, uint32_t *imgOut, int width, unsigned int height, sycl::nd_item<3> idx)
{
        // myQueue
        //    .parallel_for<class computeIntegralImgROW>(
        //              sycl::range<1>{height},[=]\
        //              (sycl::id<1> idx){
        // int row = idx[0];
         int row = idx.get_local_range(2) * idx.get_group(2)+idx.get_local_id(2);

        if(row < height)
        {
                int row_sum=0;
                for(int i=0; i<width; i++)
                {
                        row_sum += imgIn[row*width+i];
                        imgOut[row*width+i] = row_sum;
                }
        }

}

void computeIntegralImgColSYCL(uint32_t *imgOut, unsigned int width, int height, sycl::nd_item<3> idx)
{

        // int col = idx[0];
         int col = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);

        if(col < width)
        {
                int col_sum=0;
                for(int i=0; i<height; i++)
                {
                        col_sum += imgOut[col+i*width];
                        imgOut[col+i*width] = col_sum;
                }
        }
}

// /*** Compute integral image ****/
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
//end function: computeIntegralImg *********************************************




/*** Compute parameters for each rectangle in a feature:
****        upper-left corner, width, height, sign       ****/


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
// void computeFeature(double *img, double *imgSq, CvHaarFeature *f, double *featVal, int irow, int icol, int height, int width, double scale, float scale_correction_factor, CvHaarFeature *f_scaled, int real_height, int real_width)
// float computeFeature(float *img, CvHaarFeature *f, float featVal, int irow, int icol, int height, int width, float scale, float scale_correction_factor, int real_height, int real_width)
// {
//         int nRects = 0;
//         int col = 0;
//         int row = 0;
//         int wRect = 0;
//         int hRect = 0;
//         int i = 0;
//         int colVect[4] = {0};
//         int rowVect[4] = {0};
//         int wVect[4] = {0};
//         int hVect[4] = {0};

//         float w1 = 0.0;
//         float rectWeight[4] = {0};

//         float val = 0.0;
//         float s[N_RECTANGLES_MAX] = {0};

// //      *featVal = 0.0;

//         w1 = f->rect[0].weight * scale_correction_factor;

//   // Determine feature type (number of rectangles) according to weight
//         if (f->rect[2].weight == 2.0)
//         {
//                 nRects = 4;
//                 if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
//                 {
//                         rectWeight[0] = -w1;
//                         rectWeight[1] = w1;
//                         rectWeight[2] = w1;
//                         rectWeight[3] = -w1;
//                 }
//                 else
//                 {
//                         rectWeight[0] = w1;
//                         rectWeight[1] = -w1;
//                         rectWeight[2] = -w1;
//                         rectWeight[3] = w1;
//                 }
//         }
//         else
//         {
//                 if (f->rect[1].weight == 2.0)
//                 {
//                         nRects = 2;
//                         if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
//                         {
//                                 rectWeight[0] = -w1;
//                                 rectWeight[1] = w1;
//                         }
//                         else
//                         {
//                                 rectWeight[0] = w1;
//                                 rectWeight[1] = -w1;
//                         }
//                         rectWeight[2] = 0.0;
//                         rectWeight[3] = 0.0;
//                 }
//                 else
//                 {
//                         nRects = 3;
//                         rectWeight[0] = w1;
//                         rectWeight[1] = -2.0*w1;
//                         rectWeight[2] = w1;
//                         rectWeight[3] = 0.0;
//                 }
//         }
//         for (i = 0; i<nRects; i++)
//         {
//                 s[i] = 0.0;
//                 getRectangleParameters(f, i, nRects, scale, irow, icol, &row, &col, &hRect, &wRect);
//                 s[i] = computeArea((float *)img, row, col, hRect, wRect, real_height, real_width);

//                 if (sycl::fabs(rectWeight[i]) > 0.0)
//                 {
//                         val += rectWeight[i]*s[i];
//                 }
//     // test values for each rectangle
//                 rowVect[i] = row; colVect[i] = col; hVect[i] = hRect; wVect[i] = wRect;
//         }
// //      *featVal = val;
//     return val;
// //      writeInFeature(rowVect,colVect,hVect,wVect,rectWeight,nRects,f_scaled);
// }
// //end function: computeFeature *************************************************

/*** Calculate the Variance ****/
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
        uint32_t *new_arr;

        new_arr = (uint32_t *) malloc((n * sizeof (int)));
        if (new_arr == NULL) {
                TRACE_INFO(("ALLOC_1D_UINT_32T: Couldn't allocate array of integer\n"));
                return (NULL);
        }
        return (new_arr);
}
//end function: alloc_1d_uint32_t **********************************************

/*** Allocate one dimension double pointer ****/
double *alloc_1d_double(int n)
{
        double *new_arr;

        new_arr = (double *) malloc ((unsigned) (n * sizeof (double)));
        if (new_arr == NULL) {
                TRACE_INFO(("ALLOC_1D_DOUBLE: Couldn't allocate array of integer\n"));
                return (NULL);
        }
        return (new_arr);
}
//end function: alloc_1d_double ************************************************

/*** Allocate 2d array of integers ***/
uint32_t **alloc_2d_uint32_t(int m, int n)
{
        int i;
        uint32_t **new_arr;

        new_arr = (uint32_t **) malloc ((unsigned) (m * sizeof (uint32_t *)));
        if (new_arr == NULL) {
                TRACE_INFO(("ALLOC_2D_UINT_32T: Couldn't allocate array of integer ptrs\n"));
                return (NULL);
        }

        for (i = 0; i < m; i++) {
                new_arr[i] = alloc_1d_uint32_t(n);
        }

        return (new_arr);
}
//end function: alloc_2d_uint32_t **********************************************

/* Draws simple or filled square */
void raster_rectangle(uint32_t* img, int x0, int y0, int radius, int real_width)
{
        int i=0;
        for(i=-radius/2; i<radius/2; i++)
        {
                assert((i + x0 + (y0 + (int)(radius)) * real_width)>0);
                assert((i + x0 + (y0 + (int)(radius)) * real_width)<(480*640));
                assert((i + x0 + (y0 - (int)(radius)) * real_width)>0);
                assert((i + x0 + (y0 - (int)(radius)) * real_width)<(480*640));
                img[i + x0 + (y0 + (int)(radius)) * real_width]=255;
                img[i + x0 + (y0 - (int)(radius)) * real_width]=255;
        }
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


// void imgproc(float* dev_goodcenterX_tmp, float* dev_goodcenterY_tmp, uint32_t* dev_goodRadius_tmp, unsigned int ts_0, unsigned int rc, int *detSizeC, int *detSizeR, int *height,  int *width, uint32_t* dev_nb_obj_found2, double* dev_imgInt_f, double* dev_imgSqInt_f, CvHaarClassifierCascade* dev_cascade, int *real_height, int *real_width, int *nStages){

//           myQueue.submit([&] (sycl::handler& cgh){
//             sycl::stream out(1024, 256, cgh);

//             cgh.parallel_for(sycl::range<2>{ts_0,rc},[=] (sycl::id<2> idx){
//                 int s = idx[0];
//                              float scaleStep = 1.1;
//                 const float scaleFactor = (float) sycl::powr( scaleStep, (float)s);
//                 //printf("SCALEFACTOR: %f\n", scaleFactor);

//                 //TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
//                 const int tileWidth = (int)floor((*detSizeC) * scaleFactor + 0.5);
//                 const int tileHeight = (int)floor((*detSizeR) * scaleFactor + 0.5);
//                 const int rowStep = max(2, (int)floor(scaleFactor + 0.5));
//                 const int colStep = max(2, (int)floor(scaleFactor + 0.5));

//                 //(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
//                 float scale_correction_factor = (float)1.0/(float)((int)floor(((*detSizeC)-2)*scaleFactor)*(int)floor(((*detSizeR)-2)*scaleFactor));

//                 // compute number of tiles: difference between the size of the Image and the size of the Detector
//                 int nTileRows = (*height)-tileHeight;
//                 int nTileCols = (*width)-tileWidth;

//                 int foundObj = 0;
//                 int nb_obj_found=0;

//                 unsigned int irowiterations = (int)ceilf((float)nTileRows/rowStep);
//                 unsigned int icoliterations = (int)ceilf((float)nTileCols/colStep);

//                 const int stride = (nTileCols + colStep -1 )/colStep;

// //                 const int counter = idx[1];
//                 const unsigned int counter = idx[1];
//                 const int irow = (counter / stride)*rowStep;
//                 const int icol = (counter % stride)*colStep;


//                 sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_data (dev_nb_obj_found2[s]);
//                 double varFact, featVal;

// //                  if(s==0)
// //                         out << "s : " << s << " irow : " << irow << " icol : " << icol  << sycl::endl;

//                 if ((irow < nTileRows) && (icol < nTileCols))
//                 {
//                     int goodPoints_value = 255;


//                     if (goodPoints_value)
//                     {
// //                             /* Operation used for every Stage of the Classifier */
//                         varFact=computeVariance((double *)dev_imgInt_f, (double *)dev_imgSqInt_f, irow, icol, tileHeight, tileWidth, (*real_height), (*real_width));

//                         if (varFact < 10e-15)
//                         {
//                             // this should not occur (possible overflow BUG)
//                             varFact = 1.0;
//                             goodPoints_value = 0;
// //                                     continue; //NO LE GUSTA
//                         }
//                         else
//                         {
//                             // Get the standard deviation
//                             varFact = sqrt(varFact);
//                         }  // else END


//                     }  // goodPoints_value END

//                     if(goodPoints_value){
//                         for (int iStage = 0; iStage < (*nStages); iStage++)
//                         {
//                             int nNodes = dev_cascade->stageClassifier[iStage].count;

//                             if (goodPoints_value)
//                             {
//                                 double sumClassif = 0.0;

//                                 for (int iNode = 0; iNode < nNodes; iNode++)
//                                 {
//                                     featVal=computeFeature((double *)dev_imgInt_f, dev_cascade->stageClassifier[iStage].classifier[iNode].haarFeature,
//                                                     featVal, irow, icol, tileHeight, tileWidth, scaleFactor, scale_correction_factor, (*real_height), (*real_width));

//                                  // Get the thresholds for every Node of the stage
//                                     float thresh = dev_cascade->stageClassifier[iStage].classifier[iNode].threshold;
//                                     float a = dev_cascade->stageClassifier[iStage].classifier[iNode].left;
//                                     float b = dev_cascade->stageClassifier[iStage].classifier[iNode].right;
//                                     sumClassif += (featVal < (double)(thresh*varFact) ? a : b);
//                                 }  // FOR INODE END

//                                 // Update goodPoints according to detection threshold
//                                 if (sumClassif < dev_cascade->stageClassifier[iStage].threshold){
//                                     goodPoints_value = 0;

//                                 } else {
//                                     if (iStage == (*nStages) - 1)
//                                     {
//                                         float centerX=(((tileWidth-1)*0.5+icol));
//                                         float centerY=(((tileHeight-1)*0.5+irow));
//                                         uint32_t radius = sqrt(pow(tileHeight-1, 2)+pow(tileWidth-1, 2))/2;

//                                         int priv_indx = atomic_data++;
// //                                                 if(irow < 5 && icol < 5 && iStage < 3)
//                                         // out << "s : " << s << " -- priv_indx: " << 0 << " irow : " << irow << " icol : " << icol  << sycl::endl;
//                                         // printf("[%d] x: %f y: %f\n", scale, centerX, centerY);

//                                         dev_goodcenterX_tmp[s*NB_MAX_DETECTION+priv_indx]=centerX;
//                                         dev_goodcenterY_tmp[s*NB_MAX_DETECTION+priv_indx]=centerY;
//                                         dev_goodRadius_tmp[s*NB_MAX_DETECTION+priv_indx]=radius;
//                                     }
//                                 }  // else END

//                             } // GOODPOINTS_VALUE IF END
//                         } // Stages FOR END

//                     }
//                 }

//             });
//         });

// }

// imgproc(dev_goodcenterX_tmp, dev_goodcenterY_tmp, dev_goodRadius_tmp,  dev_detSizeC, dev_detSizeR, dev_height, dev_width, dev_nb_obj_found2,
//  dev_imgInt_f, dev_imgSqInt_f, dev_cascade, dev_real_height, dev_real_width, dev_nStages, out, idx);
// sycl::stream out,
void imgproc(float* dev_goodcenterX_tmp, float* dev_goodcenterY_tmp, uint32_t* dev_goodRadius_tmp,  int *detSizeC, int *detSizeR,  int *height,  int *width, uint32_t* dev_nb_obj_found2, float* dev_imgInt_f, float* dev_imgSqInt_f, CvHaarClassifierCascade* dev_cascade, int *real_height,  int *real_width, int *nStages, sycl::nd_item<3> idx){//, sycl::stream out, int image_counter


                // int s = idx[0];
                const int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
                const int s = idx.get_group(1);
                
                float scaleStep = 1.1;
                const float scaleFactor = (float) sycl::powr( scaleStep, (float)s);
                //printf("SCALEFACTOR: %f\n", scaleFactor);

                //TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
                const int tileWidth = (int)floor((*detSizeC) * scaleFactor + 0.5);
                const int tileHeight = (int)floor((*detSizeR) * scaleFactor + 0.5);
                const int rowStep = max(2, (int)floor(scaleFactor + 0.5));
                const int colStep = max(2, (int)floor(scaleFactor + 0.5));

                //(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
                float scale_correction_factor = (float)1.0/(float)((int)floor(((*detSizeC)-2)*scaleFactor)*(int)floor(((*detSizeR)-2)*scaleFactor));

                // compute number of tiles: difference between the size of the Image and the size of the Detector
                int nTileRows = (*height)-tileHeight;
                int nTileCols = (*width)-tileWidth;

                const int stride = (nTileCols + colStep -1 )/colStep;

                const int irow = (counter / stride)*rowStep;
                const int icol = (counter % stride)*colStep;


                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_data (dev_nb_obj_found2[s]);
                float varFact, featVal;

                if ((irow < nTileRows) && (icol < nTileCols))
                {
                    int goodPoints_value = 255;
                    if (goodPoints_value)
                    {
                        /* Operation used for every Stage of the Classifier */
                        varFact=computeVariance((float *)dev_imgInt_f, (float *)dev_imgSqInt_f, irow, icol, tileHeight, tileWidth, (*real_height), (*real_width));

                        if (varFact < 10e-15)
                        {
                            // this should not occur (possible overflow BUG)
                            varFact = 1.0;
                            goodPoints_value = 0;
//                                     continue; //NO LE GUSTA
                        }
                        else
                        {
                            // Get the standard deviation
                            varFact = sqrt(varFact);
                        }  // else END


                    }  // goodPoints_value END

                    if(goodPoints_value){
                        for (int iStage = 0; iStage < (*nStages); iStage++)
                        {
                            int nNodes = dev_cascade->stageClassifier[iStage].count;

                            if (goodPoints_value)
                            {
                                float sumClassif = 0.0;

                                for (int iNode = 0; iNode < nNodes; iNode++)
                                {
                                    featVal=computeFeature((float *)dev_imgInt_f, dev_cascade->stageClassifier[iStage].classifier[iNode].haarFeature,
                                                    featVal, irow, icol, tileHeight, tileWidth, scaleFactor, scale_correction_factor, (*real_height), (*real_width));

                                 // Get the thresholds for every Node of the stage
                                    float thresh = dev_cascade->stageClassifier[iStage].classifier[iNode].threshold;
                                    float a = dev_cascade->stageClassifier[iStage].classifier[iNode].left;
                                    float b = dev_cascade->stageClassifier[iStage].classifier[iNode].right;
                                    sumClassif += (featVal < (float)(thresh*varFact) ? a : b);
                                }  // FOR INODE END

                                // Update goodPoints according to detection threshold
                                if (sumClassif < dev_cascade->stageClassifier[iStage].threshold){
                                    goodPoints_value = 0;

                                } else {
                                    if (iStage == (*nStages) - 1)
                                    {
                                        float centerX=(((tileWidth-1)*0.5+icol));
                                        float centerY=(((tileHeight-1)*0.5+irow));
                                        uint32_t radius = sqrt(pow(tileHeight-1, 2)+pow(tileWidth-1, 2))/2;

                                        int priv_indx = atomic_data++;

                                        dev_goodcenterX_tmp[s*NB_MAX_DETECTION+priv_indx]=centerX;
                                        dev_goodcenterY_tmp[s*NB_MAX_DETECTION+priv_indx]=centerY;
                                        dev_goodRadius_tmp[s*NB_MAX_DETECTION+priv_indx]=radius;

                                    }
                                }  // else END

                            } // GOODPOINTS_VALUE IF END
                        } // Stages FOR END

                    }
                }



}


void k1_reduction(int* dev_scale_index_found, int *dev_nb_obj_found2, sycl::nd_item<3> idx)
{
        // int i = idx[0];
        int i = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);

        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_data (dev_nb_obj_found2[*dev_scale_index_found]);

        // printf("k1 : %d\n", *dev_scale_index_found); 
        if(i < (size_t)(*dev_scale_index_found)){
                int aux = atomic_data;
                // printf("_____kernel: MAX: %d - new : %d\n", aux, dev_nb_obj_found2[i]);
                atomic_data= max(atomic_data, dev_nb_obj_found2[i]);
                // aux = atomic_data;
                // printf("_____kernel FINAL: MAX: %d - new : %d\n", aux, dev_nb_obj_found2[i]);
                // sycl::atomic_fetch_max<uint32_t, sycl::access::address_space::generic_space>(atomic_data, dev_nb_obj_found2[i]);
        }
}



void k2(uint32_t *dev_position, int *dev_scale_index_found, int *dev_nb_obj_found2, uint32_t * dev_goodcenterX,uint32_t * dev_goodcenterY,uint32_t * dev_goodRadius)
{
        int count;
        count = 0;

        // printf("scale : %d__\n", *dev_scale_index_found);
        for(int i=(int)(*dev_scale_index_found); i>=0; i--)
        {
                for(int j=0; j<dev_nb_obj_found2[*dev_scale_index_found]; j++)
                {
                        // Normally if (goodcenterX=0 so goodcenterY=0) or (goodcenterY=0 so goodcenterX=0)
                        if(dev_goodcenterX[i*NB_MAX_DETECTION+j] !=0 || dev_goodcenterY[i*NB_MAX_DETECTION+j] !=0)
                        {
                                // dev_position[0]= 1;//dev_goodcenterX[i*NB_MAX_DETECTION+j];
                                // printf("gc X: %d - y: %d\n", dev_goodcenterX[i*NB_MAX_DETECTION+j], dev_goodcenterY[i*NB_MAX_DETECTION+j]); 
                                dev_position[count]=dev_goodcenterX[i*NB_MAX_DETECTION+j];
                                dev_position[(count)+1]=dev_goodcenterY[i*NB_MAX_DETECTION+j];
                                dev_position[(count)+2]=dev_goodRadius[i*NB_MAX_DETECTION+j];
                                (count)=(count)+3;
                        }
                }
        }

}

void k3(uint32_t *dev_position, int *dev_scale_index_found, /*int* total_indx,*/ int dev_real_width, int dev_real_height, sycl::nd_item<3> idx)
{       
        int total_indx = 10;     //change this to the number of different s found
        int offset_X=(int)(dev_real_width/(float)((total_indx)*1.2));
        int offset_Y=(int)(dev_real_height/(float)((total_indx)*1.2));
        // int i = idx[0]*3, j = (idx[1]*3)+3;

        int stride = (NB_MAX_POINTS + 2 )/3;
        int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
        int i = (counter / stride)*3;
        int j = ((counter % stride)*3) + 3;

        if(i<NB_MAX_POINTS && j <NB_MAX_POINTS-i){
                if(dev_position[i] != 0 && dev_position[i+j] != 0 && dev_position[i+1] != 0 && dev_position[i+j+1] != 0){
                        if(offset_X >= (int)sycl::abs(dev_position[i]-dev_position[i+j]) && offset_Y >= (int)sycl::abs(dev_position[i+1]-dev_position[i+j+1])){
                                // printf("i: %d j:%d\n", i,j); 
                                dev_position[i+j] = 0;
                                dev_position[i+j+1] = 0;
                                dev_position[i+j+2] = 0;
                        }
                }
        }


}


void subwindow_find_candidates(int nStages, CvHaarClassifierCascade *dev_cascade, int real_width, float *dev_imgInt_f, float *dev_imgSqInt_f, int real_height, int *dev_foundObj, int *dev_nb_obj_found, int detSizeC, int detSizeR, uint32_t *dev_goodcenterX, uint32_t *dev_goodcenterY, uint32_t *dev_goodRadius, int *dev_scale_index_found, uint32_t *dev_nb_obj_found2, sycl::nd_item<3> idx){
	const int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
                const int s = idx.get_group(1);
                
                float scaleStep = 1.1;
                const float scaleFactor = (float) sycl::powr( scaleStep, (float)s);
                //printf("SCALEFACTOR: %f\n", scaleFactor);

                //TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
                const int tileWidth = (int)floor((detSizeC) * scaleFactor + 0.5);
                const int tileHeight = (int)floor((detSizeR) * scaleFactor + 0.5);
                const int rowStep = max(2, (int)floor(scaleFactor + 0.5));
                const int colStep = max(2, (int)floor(scaleFactor + 0.5));

                //(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
                float scale_correction_factor = (float)1.0/(float)((int)floor(((detSizeC)-2)*scaleFactor)*(int)floor(((detSizeR)-2)*scaleFactor));

                // compute number of tiles: difference between the size of the Image and the size of the Detector
                int nTileRows = (real_height)-tileHeight;
                int nTileCols = (real_width)-tileWidth;

                const int stride = (nTileCols + colStep -1 )/colStep;

                const int irow = (counter / stride)*rowStep;
                const int icol = (counter % stride)*colStep;

                int goodPoints_value = 0;

                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_foundObj (dev_foundObj[s]);
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_nb_obj_found (dev_nb_obj_found[s]);
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_nb_obj_found2 (dev_nb_obj_found2[s]);
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_scale_found (*dev_scale_index_found);

                float varFact, featVal;

                if ((irow < nTileRows) && (icol < nTileCols))
                {
                    goodPoints_value = 255;
                    if (goodPoints_value)
                    {
                        /* Operation used for every Stage of the Classifier */
                        varFact=computeVariance((float *)dev_imgInt_f, (float *)dev_imgSqInt_f, irow, icol, tileHeight, tileWidth, (real_height), (real_width));

                        if (varFact < 10e-15)
                        {
                            // this should not occur (possible overflow BUG)
                            varFact = 1.0;
                            goodPoints_value = 0;
//                                     continue; //NO LE GUSTA
                        }
                        else
                        {
                            // Get the standard deviation
                            varFact = sqrt(varFact);
                        }  // else END
                    }  // goodPoints_value END

                    if(goodPoints_value){
                        for (int iStage = 0; iStage < (nStages); iStage++)
                        {
                            int nNodes = dev_cascade->stageClassifier[iStage].count;

                            if (goodPoints_value)
                            {
                                float sumClassif = 0.0;

                                for (int iNode = 0; iNode < nNodes; iNode++)
                                {
                                    featVal=computeFeature((float *)dev_imgInt_f, dev_cascade->stageClassifier[iStage].classifier[iNode].haarFeature,
                                                    featVal, irow, icol, tileHeight, tileWidth, scaleFactor, scale_correction_factor, (real_height), (real_width));


                                 // Get the thresholds for every Node of the stage
                                    float thresh = dev_cascade->stageClassifier[iStage].classifier[iNode].threshold;
                                    float a = dev_cascade->stageClassifier[iStage].classifier[iNode].left;
                                    float b = dev_cascade->stageClassifier[iStage].classifier[iNode].right;
                                    sumClassif += (featVal < (float)(thresh*varFact) ? a : b);
                                }  // FOR INODE END

                                // Update goodPoints according to detection threshold
                                if (sumClassif < dev_cascade->stageClassifier[iStage].threshold){
                                    goodPoints_value = 0;

                                } else {
                                    if (iStage == (nStages) - 1)
                                    {
                                        float centerX=(((tileWidth-1)*0.5+icol));
                                        float centerY=(((tileHeight-1)*0.5+irow));

                                        atomic_foundObj++;
                                    }
                                }  // else END

                            } // GOODPOINTS_VALUE IF END
                        } // Stages FOR END

                    }
                }

	float centerX=0.0;
	float centerY=0.0;
	float radius=0.0;
       	
	float centerX_tmp=0.0;
	float centerY_tmp=0.0;
	float radius_tmp=0.0;
	
	int threshold_X=0;
	int threshold_Y=0;


	// Determine used object 
       	if (atomic_foundObj)
       	{	
                // Only the detection is used 
                if (goodPoints_value)
                {
                        // Calculation of the Center of the detection 
                        centerX=(((tileWidth-1)*0.5+icol));
                        centerY=(((tileHeight-1)*0.5+irow));
                        
                        //Calculation of the radius of the circle surrounding object
                        radius = sycl::sqrt(pow((float)tileHeight-1, 2)+pow((float)tileWidth-1, 2))/2;

                        //Threshold calculation: proportionnal to the size of the Detector 
                        threshold_X=(int)((tileHeight-1)/(2*scaleFactor));
                        threshold_Y=(int)((tileWidth-1)/(2*scaleFactor));
                        
                        //Reduce number of detections in a given range 

                        // int dev_nb_obj_found_tmp = (atomic_scale_found)*NB_MAX_DETECTION + ((atomic_nb_obj_found)?(atomic_nb_obj_found)-1:0); // teoricamente este es el bueno
                        int dev_nb_obj_found_tmp = (s)*NB_MAX_DETECTION + ((atomic_nb_obj_found)?(atomic_nb_obj_found)-1:0); 
        // 			int dev_nb_obj_found_tmp = (*dev_scale_index_found)*NB_MAX_DETECTION + ((dev_nb_obj_found[scale])?(dev_nb_obj_found[scale])-1:0);

                        if(centerX > (dev_goodcenterX[dev_nb_obj_found_tmp]+threshold_X) || centerX < (dev_goodcenterX[dev_nb_obj_found_tmp]-threshold_X) || centerY > (dev_goodcenterY[dev_nb_obj_found_tmp]+threshold_Y) || centerY < (dev_goodcenterY[dev_nb_obj_found_tmp]-threshold_Y))
                        {
                                // printf("[%d,%d] SCALE : %d - x : %f  y: %f\n", irow, icol, dev_nb_obj_found_tmp, centerX, centerY); 
                                centerX_tmp=centerX;
                                centerY_tmp=centerY;
                                radius_tmp=radius;

                                // Get only the restricted Good Points and get back the size for each one
                                int nb_obj_found_tmp=atomic_nb_obj_found++;
                                // int dev_scale_index_found_tmp = ((atomic_scale_found)?(atomic_scale_found)-1:0)*NB_MAX_DETECTION + (nb_obj_found_tmp);

        // 			int nb_obj_found_tmp =dpct::atomic_fetch_add<int, sycl::access::address_space::generic_space>(&(dev_nb_obj_found[scale]), 1);
                                // int dev_scale_index_found_tmp = ((*dev_scale_index_found)?(*dev_scale_index_found)-1:0)*NB_MAX_DETECTION + (nb_obj_found_tmp);

                                dev_goodcenterX[s*NB_MAX_DETECTION+nb_obj_found_tmp]=centerX_tmp;
                                dev_goodcenterY[s*NB_MAX_DETECTION+nb_obj_found_tmp]=centerY_tmp;
                                dev_goodRadius[s*NB_MAX_DETECTION+nb_obj_found_tmp]=radius_tmp;

                                
                                // atomic_nb_obj_found2=max(atomic_nb_obj_found2, nb_obj_found_tmp+1 ); //= max(atomic_nb_obj_found2, (uint32_t)(nb_obj_found_tmp + 1)); 
                                int aux = atomic_nb_obj_found; 
                                atomic_scale_found= max(atomic_scale_found, s); 
                                int aux2 = atomic_scale_found; 

                                // printf("Guardando %f,%f en [%d,%d] -- scale found : %d of[%d] = %d\n",centerX_tmp,centerY_tmp, s, nb_obj_found_tmp, aux2, s, nb_obj_found_tmp);
                                // printf("Por ahora hay %d guardados en %d -- maxima escala : %d\n", aux, s, aux2 );
// 				dpct::atomic_fetch_max<uint32_t,sycl::access::address_space::generic_space>( &(dev_nb_obj_found2[(*dev_scale_index_found)? (*dev_scale_index_found) -1: 0]),(uint32_t)(nb_obj_found_tmp + 1));
                        }
                }
//                 idx.barrier();
// //                 // sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
//                 if(irow==0 && icol==0)
//                         atomic_scale_found++; 
//         //              dpct::atomic_fetch_add<int, sycl::access::address_space::generic_space>(dev_scale_index_found, 1);
        }
	
}

////////// MAIN FUNCIONAL 


/* ********************************** MAIN ********************************** */
int main( int argc, char** argv )
{
        // Timer declaration
        time_t frame_start, frame_end;

        // Pointer declaration
        CvHaarClassifierCascade* cascade = NULL;
        CvHaarFeature *feature_scaled = NULL;

        char *imgName_1 = NULL, *imgName_2 = NULL;
        char *haarFileName = NULL;
        char result_name_1[MAX_BUFFERSIZE]={0};
        char result_name_2[MAX_BUFFERSIZE]={0};

        uint32_t *img_1 = NULL, *img_2 = NULL;
        
        uint32_t *dev_img_2 = NULL;
        uint32_t *dev_imgInt_2 = NULL;
        uint32_t *dev_imgSq_2 = NULL;
        uint32_t *dev_imgSqInt_2 = NULL;

        uint32_t *position= NULL;

        // Counter Declaration
        int rowStep = 1;
        int colStep = 1;
        int width = 0;
        int height = 0;
        int detSizeR = 0;
        int detSizeC = 0;
        int tileWidth = 0;
        int tileHeight = 0;
        int nStages = 0;
        int nTileRows = 0;
        int nTileCols = 0;

        int real_height = 0, real_width = 0;
        int *scale_index_found_1, *scale_index_found_2;
        scale_index_found_1 = (int*)malloc(1*sizeof(int));
        scale_index_found_2 = (int*)malloc(1*sizeof(int));

        // Threshold Declaration

        // Factor Declaration
        float scaleFactorMax = 0.0;
        float scaleStep = 1.1; // 10% increment of detector size per scale. Change this value to test other increments
        float scaleFactor = 0.0;

        // Integral Image Declaration
        float *dev_imgInt_f_2 = NULL;
        float *dev_imgSqInt_f_2 = NULL;


        if (argc <= 2)
        {
                TRACE_INFO(("Usage: %s classifier image1 image2 ...\n", argv[0]));
                return(0);
        }

        // Get the Image name and the Cascade file name from the consoledev_goodcenterX
        haarFileName=argv[1];
        imgName_1=argv[2];
        imgName_2=argv[2];


        // Get the Input Image informations
        getImgDims(imgName_1, &width, &height);

        // Get back the Real Size of the Input Image
        real_height=height;
        real_width=width;

        // Allocate the Cascade Memory
        cascade = allocCascade_continuous();
        feature_scaled = (CvHaarFeature*) malloc(sizeof(CvHaarFeature));

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        // ALLOCATE DEVICE CASCADE AND FEATURED_SCALE SYCL

        CvHaarClassifierCascade * dev_cascade = static_cast<CvHaarClassifierCascade *>(malloc_device(sizeof(CvHaarClassifierCascade)
                                        + N_MAX_STAGES * sizeof(CvHaarStageClassifier)
                                        + N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
                                        + N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature), myQueue));

        auto dev_feature_scaled = sycl::malloc_device<CvHaarFeature>(1, myQueue);

        uint32_t* dev_goodcenterX_1 = sycl::malloc_device<uint32_t>(2*N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        uint32_t* dev_goodcenterX_2 = dev_goodcenterX_1+N_MAX_STAGES*NB_MAX_DETECTION;
        uint32_t* dev_goodcenterY_1 = sycl::malloc_device<uint32_t>(2*N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        uint32_t* dev_goodcenterY_2 = dev_goodcenterY_2+N_MAX_STAGES*NB_MAX_DETECTION;
        uint32_t* dev_goodRadius_1 = sycl::malloc_device<uint32_t>(2*N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        uint32_t* dev_goodRadius_2 = dev_goodRadius_1+N_MAX_STAGES*NB_MAX_DETECTION;

        uint32_t* dev_img_1 = sycl::malloc_device<uint32_t>(2*height*width, myQueue);
        dev_img_2 = dev_img_1+height*width;
        uint32_t* dev_imgInt_1 = sycl::malloc_device<uint32_t>(2*height*width, myQueue);
        dev_imgInt_2 = dev_imgInt_1+height*width;
        uint32_t* dev_imgSq_1 = sycl::malloc_device<uint32_t>(2*height*width, myQueue);
        dev_imgSq_2 = dev_imgSq_1+height*width;
        uint32_t* dev_imgSqInt_1 = sycl::malloc_device<uint32_t>(2*height*width, myQueue);
        dev_imgSqInt_2 = dev_imgSqInt_1+height*width;
        float* dev_imgInt_f_1 = sycl::malloc_device<float>(2*height*width, myQueue);
        dev_imgInt_f_2 = dev_imgInt_f_1+height*width;
        float* dev_imgSqInt_f_1 = sycl::malloc_device<float>(2*height*width, myQueue);
        dev_imgSqInt_f_2=dev_imgSqInt_f_1+height*width;

        int* dev_nb_obj_found_1 = sycl::malloc_device<int>(2*N_MAX_STAGES, myQueue);
        int* dev_nb_obj_found_2 = dev_nb_obj_found_1+N_MAX_STAGES;
        int* dev_scale_index_found_1 = sycl::malloc_device<int>(2, myQueue);
        int* dev_scale_index_found_2 = dev_scale_index_found_1+1;

        int* dev_foundObj_1 = sycl::malloc_device<int>(2*N_MAX_STAGES, myQueue);
        int* dev_foundObj_2 = dev_foundObj_1+N_MAX_STAGES;
        uint32_t* dev_nb_obj_found2_1 = sycl::malloc_device<uint32_t>(2*(N_MAX_STAGES+1), myQueue);
        uint32_t* dev_nb_obj_found2_2 = dev_nb_obj_found2_1+(N_MAX_STAGES+1);

        uint32_t* dev_position_1 = sycl::malloc_device<uint32_t>(2*width*height, myQueue);
        uint32_t* dev_position_2 = dev_position_1+width*height;
        int* dev_finalNb_1 = sycl::malloc_device<int>(2, myQueue);
        int* dev_finalNb_2 = dev_finalNb_1+1;

        ///////////////////////////////////////////////////////////////////////////////////////////////////

        // Get the Classifier informations
        readClassifCascade(haarFileName, cascade, &detSizeR, &detSizeC, &nStages);

        //////////////// MEMCPY HOST TO DEVICE SYCL //////////////////////
        myQueue.memcpy(dev_cascade, cascade, sizeof(CvHaarClassifierCascade)
                        + N_MAX_STAGES * sizeof(CvHaarStageClassifier)
                        + N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
                        + N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature)).wait();


        myQueue.single_task<class devFixLinks>([=](){
        int i, j;

        dev_cascade->stageClassifier = (CvHaarStageClassifier*)(((char*)dev_cascade) + sizeof(CvHaarClassifierCascade));
        for (i = 0; i < N_MAX_STAGES; i++){
            dev_cascade->stageClassifier[i].classifier = (CvHaarClassifier*)(((char*)dev_cascade->stageClassifier) + (N_MAX_STAGES * sizeof(CvHaarStageClassifier)) + (i*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)));
            for(j = 0; j < N_MAX_CLASSIFIERS; j++){
                dev_cascade->stageClassifier[i].classifier[j].haarFeature = (CvHaarFeature*)(((char*)&(dev_cascade->stageClassifier[N_MAX_STAGES])) \
                                                                                             + (N_MAX_STAGES*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)) + (((i*N_MAX_CLASSIFIERS)+j)*sizeof(CvHaarFeature)));
            }
        }
        }).wait();

        uint32_t* position_1=(uint32_t*) alloc_1d_uint32_t(width*height);
        uint32_t* position_2=(uint32_t*) alloc_1d_uint32_t(width*height);
        uint32_t* nb_obj_found2_1 = alloc_1d_uint32_t(N_MAX_STAGES);
        uint32_t* nb_obj_found2_2 = alloc_1d_uint32_t(N_MAX_STAGES);
        uint32_t** result2_1=alloc_2d_uint32_t(N_MAX_STAGES, width*height);
        uint32_t** result2_2=alloc_2d_uint32_t(N_MAX_STAGES, width*height);

        /////////////////////////////////////////////////////////////////

        TRACE_INFO(("detSizeR = %d\n", detSizeR));
        TRACE_INFO(("detSizeC = %d\n", detSizeC));

        // Determine the Max Scale Factor
        if (detSizeR != 0 && detSizeC != 0)
        {
                scaleFactorMax = min((int)floor(height/detSizeR), (int)floor(width/detSizeC));
        }

        // // Give the Allocation size
        printf("img alloc\n");
        img_1= (uint32_t*) alloc_1d_uint32_t(2*width*height);
        img_2= img_1+width*height;

        TRACE_INFO(("nStages: %d\n", nStages));
        TRACE_INFO(("NB_MAX_DETECTION: %d\n", NB_MAX_DETECTION));



        TRACE_INFO(("Allocations finished!\n"));

        int total_scales=0;
        for (scaleFactor = 1; scaleFactor <= scaleFactorMax; scaleFactor *= scaleStep)
                total_scales++;

    ///////////////// PRINT DEVICE SELECTOR /////////////////
        std::cout << "Chosen device: "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    ////////////////////////////////////////////////////////

        frame_start = clock();

        sycl::event e1_1, e2_1, e3_1, e4_1, e5_1, e6_1, e7_1, e8_1, e9_1, e10_1, e11_1, e12_1, e13_1, e14_1, e15_1, e16_1, e17_1, e18_1, e19_1, e20_1, e21_1, e22_1, e23_1;
        sycl::event e1_2, e2_2, e3_2, e4_2, e5_2, e6_2, e7_2, e8_2, e9_2, e10_2, e11_2, e12_2, e13_2, e14_2, e15_2, e16_2, e17_2, e18_2, e19_2, e20_2, e21_2, e22_2, e23_2;

        sycl::event *e1, *e2, *e3, *e4, *e5, *e6, *e7, *e8, *e9, *e10, *e11, *e12, *e13, *e14, *e15, *e16, *e17, *e18, *e19, *e20, *e21, *e22, *e23;

        sycl::queue *current_queue;

        scaleFactor=1.1; 
        tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
        tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);
        rowStep = max(2, (int)floor(scaleFactor + 0.5));
        colStep = max(2, (int)floor(scaleFactor + 0.5));

        //(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
        // compute number of tiles: difference between the size of the Image and the size of the Detector
        nTileRows = height-tileHeight;
        nTileCols = width-tileWidth;

        // TRACE_INFO(("Inside scale for and before stage for!\n"));
        int irowiterations = (int)ceilf((float)nTileRows/rowStep);
        int icoliterations = (int)ceilf((float)nTileCols/colStep);

        // Task 2:  Frame processing.
        // Launch the Main Loop
        time_t start, end; 
        int number_of_threads = irowiterations*icoliterations;
        double frame_time_img, frame_time_work; 
        do // Infinite loop
        { //(argc-2)
        for(int image_counter=0; image_counter <(argc-2); image_counter++)
        {

        int *dev_scale_index_found;
        // // int *dev_count;
        uint32_t *dev_goodcenterX;
        uint32_t *dev_goodcenterY;
        uint32_t *dev_goodRadius;
        uint32_t *dev_position;
        // uint32_t *dev_result2;
        uint32_t **result2;
        char *imgName;

        uint32_t *img;
        uint32_t *dev_img;
        uint32_t *dev_imgInt;
        uint32_t *dev_imgSq;
        uint32_t *dev_imgSqInt;
        float *dev_imgInt_f;
        float *dev_imgSqInt_f;

        int *dev_foundObj;
        uint32_t *dev_nb_obj_found2;

        int *dev_nb_obj_found;
        int *dev_finalNb ;
        int *scale_index_found;
        char *result_name;
        uint32_t* position;
        uint32_t* nb_obj_found2;

        
        // if((image_counter%4)==0 || (image_counter%4)==1)
        if((image_counter%4)==0 || (image_counter%4)==1)
        {       
                imgName=imgName_1;
                current_queue = &myQueue;
                position = position_1;
                nb_obj_found2=nb_obj_found2_1;

                dev_scale_index_found = dev_scale_index_found_1;
                // // dev_count = dev_count_1;

                dev_goodcenterX = dev_goodcenterX_1;
                dev_goodcenterY = dev_goodcenterY_1;
                dev_goodRadius = dev_goodRadius_1;

                dev_position = dev_position_1;
                // dev_result2 = dev_result2_1;
                result2 = result2_1;

                img=img_1;
                dev_img=dev_img_1;
                dev_imgInt = dev_imgInt_1;
                dev_imgSq = dev_imgSq_1;
                dev_imgSqInt = dev_imgSqInt_1;
                dev_imgInt_f = dev_imgInt_f_1;
                dev_imgSqInt_f = dev_imgSqInt_f_1;

                dev_foundObj = dev_foundObj_1;
                dev_nb_obj_found2 = dev_nb_obj_found2_1;

                dev_nb_obj_found = dev_nb_obj_found_1;
                dev_finalNb = dev_finalNb_1;
                // finalNb = finalNb_1;
                scale_index_found = scale_index_found_1;
                result_name =result_name_1;

                e1=&e1_1; e2=&e2_1;  e3=&e3_1;  e4=&e4_1;  e5=&e5_1;
                e6=&e6_1; e7=&e7_1;  e8=&e8_1;  e9=&e9_1;  e10=&e10_1;
                e11=&e11_1;  e12=&e12_1;  e13=&e13_1;  e14=&e14_1; e15=&e15_1;
                e16=&e16_1; e17=&e17_1;  e18=&e18_1;  e19=&e19_1;  e20=&e20_1;
                e21=&e21_1; e22=&e22_1; e23=&e23_1;
        }
        else
        {       
                imgName=imgName_2;
                current_queue = &myQueue; 
                position = position_2;
                nb_obj_found2=nb_obj_found2_2;

                dev_scale_index_found = dev_scale_index_found_2;
                // // dev_count = dev_count_2;
                dev_goodcenterX = dev_goodcenterX_2;
                dev_goodcenterY = dev_goodcenterY_2;
                dev_goodRadius = dev_goodRadius_2;

                dev_position = dev_position_2;
                // dev_result2 = dev_result2_2;
                result2 = result2_2;

                img=img_2;
                dev_img=dev_img_2;
                dev_imgInt = dev_imgInt_2;
                dev_imgSq = dev_imgSq_2;
                dev_imgSqInt = dev_imgSqInt_2;
                dev_imgInt_f = dev_imgInt_f_2;
                dev_imgSqInt_f = dev_imgSqInt_f_2;

                dev_foundObj = dev_foundObj_2;
                dev_nb_obj_found2 = dev_nb_obj_found2_2;

                dev_nb_obj_found = dev_nb_obj_found_2;

                dev_finalNb = dev_finalNb_2;
                // finalNb = finalNb_2;
                scale_index_found = scale_index_found_2;
                result_name =result_name_2;

                e1=&e1_2; e2=&e2_2;  e3=&e3_2;  e4=&e4_2;  e5=&e5_2;
                e6=&e6_2; e7=&e7_2;  e8=&e8_2;  e9=&e9_2;  e10=&e10_2;
                e11=&e11_2; e12=&e12_2;  e13=&e13_2;  e14=&e14_2; e15=&e15_2;
                e16=&e16_2; e17=&e17_2;  e18=&e18_2;  e19=&e19_2;  e20=&e20_2;
                e21=&e21_2; e22=&e22_2;  e23=&e23_2;
        }

        // // Task 0: Queue init
        // // End Task 0
        // // Start Task 1
        imgName=argv[image_counter+2];
        load_image_check((uint32_t *)img, (char *)imgName, width, height);

        ///////////////// MEMCPY HOST TO DEVICE SYCL //////////////////////
        *e1 = current_queue->memcpy(dev_img, img, width*height*sizeof(uint32_t));

        *e2 = current_queue->memset(dev_foundObj, 0, N_MAX_STAGES*sizeof(int), *e16);
        *e3 = current_queue->memset(dev_nb_obj_found2, 0, N_MAX_STAGES*sizeof (uint32_t), *e16);
        *e4 = current_queue->memset(dev_goodcenterX, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t), *e16);
        *e5 = current_queue->memset(dev_goodcenterY, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t), *e16);
        *e6 = current_queue->memset(dev_goodRadius, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t), *e16);
        *e7 = current_queue->memset(dev_nb_obj_found, 0, N_MAX_STAGES*sizeof (int), *e16);
        *e8 = current_queue->memset(dev_scale_index_found,  0, 1*sizeof(int), *e16);

        

        int block_size = 16;
        sycl::range<3> block(1, 1, block_size);
        sycl::range<3> grid_row(1, 1, (height + block_size - 1) / block_size);
        sycl::range<3> grid_column(1, 1, (width + block_size - 1) / block_size);

        /* Compute the Interal Image */
        //SYCL GPU
        *e9 = current_queue->parallel_for(
        sycl::nd_range<3>(grid_row * block, block), {*e1}, [=](sycl::nd_item<3> idx){
                computeIntegralImgRowSYCL((uint32_t *)dev_img, (uint32_t *) dev_imgInt, width, height, idx);
        });

        *e10 = current_queue->parallel_for(
        sycl::nd_range<3>(grid_column * block, block), {*e9}, [=](sycl::nd_item<3> idx){
                computeIntegralImgColSYCL((uint32_t *)dev_imgInt, width, height, idx);
        });

        // Calculate the Image square
        *e11 = current_queue->parallel_for(
        sycl::range<1>{size_t(height*width)}, {*e1}, [=](sycl::id<1> idx){
                imgDotSquareSYCL((uint32_t *)dev_img, (uint32_t *)dev_imgSq, width, height, idx);
        });

        /* Compute the Integral Image square */
        *e12 = current_queue->parallel_for(sycl::nd_range<3>(grid_row * block, block), {*e11}, [=](sycl::nd_item<3> idx){
                computeIntegralImgRowSYCL((uint32_t *)dev_imgSq, (uint32_t *) dev_imgSqInt, width, height, idx);
        });

        *e13 = current_queue->parallel_for(sycl::nd_range<3>(grid_column * block, block), {*e12}, [=](sycl::nd_item<3> idx){
                computeIntegralImgColSYCL((uint32_t *)dev_imgSqInt, width, height, idx);
        });

        // e16->wait();
        // Copy the Image to float array
        *e14 = current_queue->parallel_for( sycl::nd_range<3>(sycl::range<3>(1, 1, (width * height) / 128) *
        sycl::range<3>(1, 1, 128),sycl::range<3>(1, 1, 128)), {*e10, *e16},  [=](sycl::nd_item<3> idx) {
                imgCopySYCL((uint32_t *)dev_imgInt, (float *)dev_imgInt_f, height, width, idx);
        });

        *e15 = current_queue->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (width * height) / 128) *
                sycl::range<3>(1, 1, 128),sycl::range<3>(1, 1, 128)), {*e13, *e16}, [=](sycl::nd_item<3> idx) {
                imgCopySYCL((uint32_t *)dev_imgSqInt, (float *)dev_imgSqInt_f, height, width, idx);
        });

        // // Task 1: End Frame acquisition.
        scaleFactor = 1;
        //TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
        
        int block_size_sub = 64;
        sycl::range<3> block_subwindow(1, total_scales, (number_of_threads + (block_size_sub - 1)) / block_size_sub);
        sycl::range<3> thread_subwindow(1, 1, block_size_sub);

        size_t ts_0 = total_scales;


        *e16 = current_queue->parallel_for(sycl::nd_range<3>(block_subwindow * thread_subwindow,thread_subwindow), {*e2, *e3, *e4, *e5, *e6, *e7, *e8, *e14, *e15}, [=](sycl::nd_item<3> idx) {
                subwindow_find_candidates( nStages, dev_cascade, real_width, dev_imgInt_f, dev_imgSqInt_f, real_height, dev_foundObj, dev_nb_obj_found, detSizeC, detSizeR, dev_goodcenterX, dev_goodcenterY, dev_goodRadius, dev_scale_index_found, dev_nb_obj_found2, idx);
        });
        // myQueue.memcpy(scale_index_found, dev_scale_index_found, 1*sizeof(int), *e16);
        
        *e17 = current_queue->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ts_0), sycl::range<3>(1, 1, ts_0)), {*e16, *e20},
                [=](sycl::nd_item<3> idx) {
                k1_reduction(dev_scale_index_found, dev_nb_obj_found, idx);
        });

        *e18 = current_queue->memset(dev_position, 0, width*height*sizeof(uint32_t), *e22);

        // // Keep the position of each circle from the bigger to the smaller
        *e19 = current_queue->single_task({*e17, *e18}, [=](){
                k2(dev_position, dev_scale_index_found, dev_nb_obj_found, dev_goodcenterX, dev_goodcenterY, dev_goodRadius);
        });

        // Delete detections which are too close
        *e20 = current_queue->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (number_of_threads + 127) / 128) * sycl::range<3>(1, 1, 128),
                sycl::range<3>(1, 1, 128)), {*e19},   [=](sycl::nd_item<3> idx) {
                        k3(dev_position, dev_scale_index_found, real_width, real_height, idx);
        });

        *e21 = current_queue->memset(dev_finalNb, 0, 1*sizeof(int), *e22);
        e22->wait(); 

        *e22 = current_queue->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (irowiterations + 127) / 128) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), \ 
        {*e20, *e21}, [=](sycl::nd_item<3> idx) {
                // int i = idx[0]*3;
                int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
                int i = counter * 3;

                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_nb (dev_finalNb[0]);

                if (dev_position[i]!=0)
                {
                        int c = atomic_nb++;
                        printf("[%d] x: %d, y: %d, scale: %d, ValidOutput: %d\n", image_counter, (int)dev_position[i], (int)dev_position[i+1], (int)(dev_position[i+2]/2), c);
                }
        });
        
        #if WRITE_IMG
        current_queue->memcpy(position, dev_position, width*height*sizeof(uint32_t)).wait();
        current_queue->memcpy(dev_nb_obj_found2, nb_obj_found2, N_MAX_STAGES*sizeof(uint32_t)).wait();
        // Re-build the result image with highlighted detections
        for (int i = 0; i < real_height; i++){
                for (int j = 0; j < real_width; j++){
                result2[scale_index_found][i * real_width + j] = img[i * real_width + j];
                }
        }

        // Draw detection
        for (int i = 0; i < NB_MAX_POINTS; i += 3){
                if (position[i] != 0 && position[i + 1] != 0 && position[i + 2] != 0)
                raster_rectangle(result2[scale_index_found], (int)position[i], (int)position[i + 1], (int)(position[i + 2] / 2), real_width);
        }

        //   Write the final result of the detection application
                sprintf(result_name, "result_%d.pgm", image_counter);
                imgWrite((uint32_t *)result2[scale_index_found], result_name, height, width);
        #endif

        // Task 3: End Frame post-processing.
        } //for of all images
        } while(InProcessingLoop());

        myQueue.wait();
        // myQueue2.wait();

        frame_end = clock();
        float frame_time = (float)(frame_end-frame_start)/CLOCKS_PER_SEC * 1000;
        printf("\n TOTAL Antes WAIT Execution time = %f for %d FRAMES ms.\n", frame_time, (argc-2));

        // FREE ALL the allocations
        releaseCascade_continuous(cascade);
        free(feature_scaled);

        return 0;
}
