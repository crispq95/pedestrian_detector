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
// auto myQueue = sycl::queue{my_device_selector{}};

auto myQueue = sycl::queue{ sycl::gpu_selector{}, {sycl::property::queue::in_order{}}};
auto myQueue2 = sycl::queue{ sycl::gpu_selector{}, {sycl::property::queue::in_order{}}};



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
            // }).wait();


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
            // }).wait();
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

/*** Recover any pixel in the image by using the integral image ****/
inline float getImgIntPixel(float *img, int row, int col, int real_height, int real_width)
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
inline float computeArea(float *img, int row, int col, int height, int width, int real_height, int real_width)
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
inline void getRectangleParameters(CvHaarFeature *f, int iRectangle, int nRectangles, double scale, int rOffset, int cOffset, int *row, int *col, int *height, int *width)
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
// void computeFeature(double *img, double *imgSq, CvHaarFeature *f, double *featVal, int irow, int icol, int height, int width, double scale, float scale_correction_factor, CvHaarFeature *f_scaled, int real_height, int real_width)
inline float computeFeature(float *img, CvHaarFeature *f, float featVal, int irow, int icol, int height, int width, float scale, float scale_correction_factor, int real_height, int real_width)
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
    #pragma clang loop unroll_count(4)
        for (i = 0; i<nRects; i++)
        {
                s[i] = 0.0;
                getRectangleParameters(f, i, nRects, scale, irow, icol, &row, &col, &hRect, &wRect);
                s[i] = computeArea((float *)img, row, col, hRect, wRect, real_height, real_width);

                if (sycl::fabs(rectWeight[i]) > 0.0)
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

/*** Calculate the Variance ****/
inline float computeVariance(float *img, float *imgSq, int irow, int icol, int height, int width, int real_height, int real_width)
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

                const int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
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


void k1_reduction(uint32_t *dev_scale_index_found, uint32_t *dev_nb_obj_found2, sycl::nd_item<3> idx)
{
        // int i = idx[0];
        int i = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);

        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_data (dev_nb_obj_found2[*dev_scale_index_found]);

        if(i < (size_t)(*dev_scale_index_found)){
                max(atomic_data, dev_nb_obj_found2[i]);
                // sycl::atomic_fetch_max<uint32_t, sycl::access::address_space::generic_space>(atomic_data, dev_nb_obj_found2[i]);
        }
}



void k2(uint32_t *dev_position, uint32_t *dev_scale_index_found, uint32_t *dev_nb_obj_found2, uint32_t * dev_goodcenterX,uint32_t * dev_goodcenterY,uint32_t * dev_goodRadius)
{
        int count;
        count = 0;

        for(int i=(int)(*dev_scale_index_found); i>=0; i--)
        {
                for(int j=0; j<dev_nb_obj_found2[*dev_scale_index_found]; j++)
                {
                        // Normally if (goodcenterX=0 so goodcenterY=0) or (goodcenterY=0 so goodcenterX=0)
                        if(dev_goodcenterX[i*NB_MAX_DETECTION+j] !=0 || dev_goodcenterY[i*NB_MAX_DETECTION+     j] !=0)
                        {
                                // dev_position[0]= 1;//dev_goodcenterX[i*NB_MAX_DETECTION+j];
                                dev_position[count]=dev_goodcenterX[i*NB_MAX_DETECTION+j];
                                dev_position[(count)+1]=dev_goodcenterY[i*NB_MAX_DETECTION+j];
                                dev_position[(count)+2]=dev_goodRadius[i*NB_MAX_DETECTION+j];
                                (count)=(count)+3;
                        }
                }
        }

}

void k3(uint32_t *dev_position, uint32_t *dev_scale_index_found, int* total_indx, int *dev_real_width, int *dev_real_height, sycl::nd_item<3> idx)
{
        int offset_X=(int)(*dev_real_width/(float)((*total_indx)*1.2));
        int offset_Y=(int)(*dev_real_height/(float)((*total_indx)*1.2));
        // int i = idx[0]*3, j = (idx[1]*3)+3;

        int stride = (NB_MAX_POINTS + 2 )/3;
        int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
        int i = (counter / stride)*3;
        int j = ((counter % stride)*3) + 3;

        if(i<NB_MAX_POINTS && j <NB_MAX_POINTS-i){
                if(dev_position[i] != 0 && dev_position[i+j] != 0 && dev_position[i+1] != 0 && dev_position[i+j+1] != 0){
                        if(offset_X >= (int)sycl::abs(dev_position[i]-dev_position[i+j]) && offset_Y >= (int)sycl::abs(dev_position[i+1]-dev_position[i+j+1])){
                                dev_position[i+j] = 0;
                                dev_position[i+j+1] = 0;
                                dev_position[i+j+2] = 0;
                        }
                }
        }


}

/* ********************************** MAIN ********************************** */
int main( int argc, char** argv )
{
        // Timer declaration
        time_t frame_start, frame_end;

        // Pointer declaration
        CvHaarClassifierCascade* cascade = NULL;
        CvHaarFeature *feature_scaled = NULL;

        char *imgName = NULL;
        char *haarFileName = NULL;
        char result_name[MAX_BUFFERSIZE]={0};

        uint32_t *img = NULL;
        uint32_t *imgInt = NULL;
        uint32_t *imgSq = NULL;
        uint32_t *imgSqInt = NULL;
        uint32_t **result2 = NULL;

        uint32_t *goodcenterX=NULL;
        uint32_t *goodcenterY=NULL;
        uint32_t *goodRadius=NULL;
        uint32_t *nb_obj_found2=NULL;

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
        int scale_index_found=0;


        // Threshold Declaration


        // Factor Declaration
        double scaleFactorMax = 0.0;
        double scaleStep = 1.1; // 10% increment of detector size per scale. Change this value to test other increments
        double scaleFactor = 0.0;

        // Integral Image Declaration
        double *imgInt_f = NULL;
        double *imgSqInt_f = NULL;


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
        feature_scaled = (CvHaarFeature*) malloc(sizeof(CvHaarFeature));

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        // ALLOCATE DEVICE CASCADE AND FEATURED_SCALE SYCL

        CvHaarClassifierCascade * dev_cascade = static_cast<CvHaarClassifierCascade *>(malloc_device(sizeof(CvHaarClassifierCascade)
                                        + N_MAX_STAGES * sizeof(CvHaarStageClassifier)
                                        + N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
                                        + N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature), myQueue));

        auto dev_feature_scaled = sycl::malloc_device<CvHaarFeature>(1, myQueue);

        auto dev_goodcenterX_tmp = sycl::malloc_device<float>(N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        auto dev_goodcenterY_tmp = sycl::malloc_device<float>(N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        auto dev_goodRadius_tmp = sycl::malloc_device<uint32_t>(N_MAX_STAGES*NB_MAX_DETECTION, myQueue);

        auto dev_goodcenterX = sycl::malloc_device<uint32_t>(N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        auto dev_goodcenterY = sycl::malloc_device<uint32_t>(N_MAX_STAGES*NB_MAX_DETECTION, myQueue);
        auto dev_goodRadius = sycl::malloc_device<uint32_t>(N_MAX_STAGES*NB_MAX_DETECTION, myQueue);

        auto dev_nb_obj_found2 = sycl::malloc_device<uint32_t>(N_MAX_STAGES, myQueue);

        auto dev_img = sycl::malloc_device<uint32_t>(height*width, myQueue);
        auto dev_imgInt = sycl::malloc_device<uint32_t>(height*width, myQueue);
        auto dev_imgSq = sycl::malloc_device<uint32_t>(height*width, myQueue);
        auto dev_imgSqInt = sycl::malloc_device<uint32_t>(height*width, myQueue);
        auto dev_imgInt_f = sycl::malloc_device<float>(height*width, myQueue);
        auto dev_imgSqInt_f = sycl::malloc_device<float>(height*width, myQueue);

        auto dev_nb_obj_found = sycl::malloc_device<uint32_t>(N_MAX_STAGES, myQueue);
        auto dev_scale_index_found = sycl::malloc_device<uint32_t>(1, myQueue);

        auto dev_foundObj = sycl::malloc_device<int>(N_MAX_STAGES, myQueue);

        auto dev_detSizeR = sycl::malloc_device<int>(1, myQueue);
        auto dev_detSizeC = sycl::malloc_device<int>(1, myQueue);
        auto dev_height = sycl::malloc_device<int>(1, myQueue);
        auto dev_width = sycl::malloc_device<int>(1, myQueue);
        auto dev_real_height = sycl::malloc_device<int>(1, myQueue);
        auto dev_real_width = sycl::malloc_device<int>(1, myQueue);
        auto dev_nStages = sycl::malloc_device<int>(1, myQueue);

        auto dev_position = sycl::malloc_device<uint32_t>(width*height, myQueue); //revisar size
        auto dev_finalNb = sycl::malloc_device<int>(1, myQueue);

        auto dev_last_nb_obj_found2 = sycl::malloc_device<uint32_t>(N_MAX_STAGES, myQueue);
        auto dev_last_scale_index_found = sycl::malloc_device<uint32_t>(1, myQueue);
        auto dev_nIndexFound = sycl::malloc_device<int>(1, myQueue);



        ///////////////////////////////////////////////////////////////////////////////////////////////////

        // Get the Classifier informations
        readClassifCascade(haarFileName, cascade, &detSizeR, &detSizeC, &nStages);

        //////////////// MEMCPY DEVICE TO HOST SYCL //////////////////////
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

        myQueue.memcpy(dev_detSizeR, &detSizeR, sizeof(int));
        myQueue.memcpy(dev_detSizeC, &detSizeC, sizeof(int));
        myQueue.memcpy(dev_height, &height, sizeof(int));
        myQueue.memcpy(dev_width, &width, sizeof(int));
        myQueue.memcpy(dev_real_height, &real_height, sizeof(int));
        myQueue.memcpy(dev_real_width, &real_width, sizeof(int));
        myQueue.memcpy(dev_nStages, &nStages, sizeof(int));

        /////////////////////////////////////////////////////////////////

        TRACE_INFO(("detSizeR = %d\n", detSizeR));
        TRACE_INFO(("detSizeC = %d\n", detSizeC));

        // Determine the Max Scale Factor
        if (detSizeR != 0 && detSizeC != 0)
        {
                scaleFactorMax = min((int)floor(height/detSizeR), (int)floor(width/detSizeC));
        }

        // Give the Allocation size
        img= (uint32_t*) alloc_1d_uint32_t(width*height);

        imgInt= (uint32_t*) alloc_1d_uint32_t(width*height);
        imgSq=(uint32_t*) alloc_1d_uint32_t(width*height);
        imgSqInt=(uint32_t*) alloc_1d_uint32_t(width*height);
        position=(uint32_t*) alloc_1d_uint32_t(width*height);

        TRACE_INFO(("nStages: %d\n", nStages));
        TRACE_INFO(("NB_MAX_DETECTION: %d\n", NB_MAX_DETECTION));

        //nstages
        result2=alloc_2d_uint32_t(N_MAX_STAGES, width*height);
        goodcenterX=alloc_1d_uint32_t(N_MAX_STAGES*NB_MAX_DETECTION);
        goodcenterY=alloc_1d_uint32_t(N_MAX_STAGES*NB_MAX_DETECTION);
        goodRadius=alloc_1d_uint32_t(N_MAX_STAGES*NB_MAX_DETECTION);
        nb_obj_found2=alloc_1d_uint32_t(N_MAX_STAGES);

        imgInt_f=alloc_1d_double(width*height);
        imgSqInt_f=alloc_1d_double(width*height);


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

        do // Infinite loop
        {
        for(int image_counter=0; image_counter < argc-2; image_counter++)
        {

            // Task 0: Queue init

            // End Task 0
            // Start Task 1
            scale_index_found=0;
            memset(position, 0, width*height*sizeof (uint32_t));
            memset(nb_obj_found2, 0, N_MAX_STAGES*sizeof (uint32_t));

            imgName=argv[image_counter+2];
            load_image_check((uint32_t *)img, (char *)imgName, width, height);

            ///////////////// MEMCPY HOST TO DEVICE SYCL //////////////////////
            myQueue.memcpy(dev_img, img, width*height*sizeof(uint32_t));

            int block_size = 16;
            sycl::range<3> block(1, 1, block_size);
            sycl::range<3> grid_row(1, 1, (height + block_size - 1) / block_size);
            sycl::range<3> grid_column(1, 1, (width + block_size - 1) / block_size);

            // Compute the Interal Image
            //SYCL GPU
            myQueue.parallel_for(
                sycl::nd_range<3>(grid_row * block, block),[=](sycl::nd_item<3> idx){
                    computeIntegralImgRowSYCL((uint32_t *)dev_img, (uint32_t *) dev_imgInt, width, height, idx);
                });

            myQueue.parallel_for(
                sycl::nd_range<3>(grid_column * block, block), [=](sycl::nd_item<3> idx){
                    computeIntegralImgColSYCL((uint32_t *)dev_imgInt, width, height, idx);
                });

                // Calculate the Image square
            myQueue.parallel_for(
                sycl::range<1>{size_t(height*width)}, [=](sycl::id<1> idx){
                    imgDotSquareSYCL((uint32_t *)dev_img, (uint32_t *)dev_imgSq, width, height, idx);
                });

            /* Compute the Integral Image square */
            myQueue.parallel_for(
                    sycl::nd_range<3>(grid_row * block, block), [=](sycl::nd_item<3> idx){
                            computeIntegralImgRowSYCL((uint32_t *)dev_imgSq, (uint32_t *) dev_imgSqInt, width, height, idx);
                });

            myQueue.parallel_for(
                sycl::nd_range<3>(grid_column * block, block), [=](sycl::nd_item<3> idx){
                        computeIntegralImgColSYCL((uint32_t *)dev_imgSqInt, width, height, idx);
                });

                // Copy the Image to float array
            myQueue.parallel_for( sycl::nd_range<3>(sycl::range<3>(1, 1, (width * height) / 128) *
                sycl::range<3>(1, 1, 128),sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> idx) {
                    imgCopySYCL((uint32_t *)dev_imgInt, (float *)dev_imgInt_f, height, width, idx);
                });

            myQueue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (width * height) / 128) *
                    sycl::range<3>(1, 1, 128),sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> idx) {
                        imgCopySYCL((uint32_t *)dev_imgSqInt, (float *)dev_imgSqInt_f, height, width, idx);
                });

        // Task 1: End Frame acquisition.
        scaleFactor = 1;

        //TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
        tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
        tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);
        rowStep = max(2, (int)floor(scaleFactor + 0.5));
        colStep = max(2, (int)floor(scaleFactor + 0.5));

        //(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
        // compute number of tiles: difference between the size of the Image and the size of the Detector
        nTileRows = height-tileHeight;
        nTileCols = width-tileWidth;

        TRACE_INFO(("Inside scale for and before stage for!\n"));

        int irowiterations = (int)ceilf((float)nTileRows/rowStep);
        int icoliterations = (int)ceilf((float)nTileCols/colStep);

            // Task 2:  Frame processing.
            // Launch the Main Loop
            int number_of_threads = irowiterations*icoliterations;
            int block_size_sub = 64;
            sycl::range<3> block_subwindow(1, total_scales, (number_of_threads + (block_size_sub - 1)) / block_size_sub);
            sycl::range<3> thread_subwindow(1, 1, block_size_sub);

            size_t ts_0 = total_scales;

            myQueue.memset(dev_goodRadius_tmp, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t)).wait();
            myQueue.memset(dev_goodcenterX_tmp, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(float)).wait();
            myQueue.memset(dev_goodcenterY_tmp, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(float)).wait();
            myQueue.memcpy(dev_nb_obj_found2, nb_obj_found2, N_MAX_STAGES*sizeof (uint32_t)).wait();
            
            myQueue.memset(dev_goodcenterX, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t)).wait();
            myQueue.memset(dev_goodcenterY, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t)).wait();
            myQueue.memset(dev_goodRadius, 0, NB_MAX_DETECTION*N_MAX_STAGES*sizeof(uint32_t)).wait();

            myQueue.memcpy(dev_nb_obj_found, nb_obj_found2, N_MAX_STAGES*sizeof (uint32_t)).wait();
            myQueue.memcpy(dev_scale_index_found, &scale_index_found, 1*sizeof(uint32_t)).wait();
            myQueue.memcpy(dev_nIndexFound, &scale_index_found, 1*sizeof(uint32_t));

            myQueue.submit([&](sycl::handler &cgh) {cgh.parallel_for(
                sycl::nd_range<3>(block_subwindow * thread_subwindow, thread_subwindow),
                [=](sycl::nd_item<3> idx) {
                            imgproc(dev_goodcenterX_tmp, dev_goodcenterY_tmp, dev_goodRadius_tmp, dev_detSizeC, dev_detSizeR, dev_height, dev_width, dev_nb_obj_found2, dev_imgInt_f, dev_imgSqInt_f, dev_cascade, dev_real_height, dev_real_width, dev_nStages,  idx); //out, image_counter
                        });
                });

            // Done processing all scales myQueue.submit([&] (sycl::handler& cgh){
            myQueue.submit([&] (sycl::handler& cgh){
                cgh.parallel_for(sycl::range<1>{size_t(ts_0)},[=] (sycl::id<1> idx){
                    int s = idx[0];
                    float centerX_tmp, centerY_tmp, radius_tmp;

                    //change this for a single number
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_indx (dev_nb_obj_found[s]);
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_scl (dev_scale_index_found[0]);
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_total_idx (dev_nIndexFound[0]);

                    // const float scaleFactor = (float) powf(scaleStep, (float)s);
                    float ss = (float)scaleStep;
                    const float scaleFactor = (float) sycl::powr( ss, (float)s);
                    int tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
                    int tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);

                        if(s < ts_0){
                            for (int j=0; j < dev_nb_obj_found2[s]; j++){
                                float centerX = dev_goodcenterX_tmp[s*NB_MAX_DETECTION+j];
                                float centerY = dev_goodcenterY_tmp[s*NB_MAX_DETECTION+j];

                                int threshold_X=(int)((tileHeight-1)/(2*scaleFactor));
                                int threshold_Y=(int)((tileWidth-1)/(2*scaleFactor));

                                if(centerX > (centerX_tmp+threshold_X) || centerX < (centerX_tmp-threshold_X) || (centerY > centerY_tmp+threshold_Y) || centerY < (centerY_tmp-threshold_Y))
                                {
                                    centerX_tmp = centerX;
                                    centerY_tmp = centerY;
                                    radius_tmp = dev_goodRadius_tmp[s*NB_MAX_DETECTION+j];

                                    int priv_indx = atomic_indx++;

                                    dev_goodcenterX[s*NB_MAX_DETECTION+(priv_indx)]=(uint32_t)centerX_tmp;
                                    dev_goodcenterY[s*NB_MAX_DETECTION+(priv_indx)]=(uint32_t)centerY_tmp;
                                    dev_goodRadius[s*NB_MAX_DETECTION+(priv_indx)]=(uint32_t)radius_tmp;
                                }
                            }

                            if(dev_nb_obj_found2[s]){
                                atomic_scl=(s > atomic_scl)? s: atomic_scl;
                                atomic_total_idx++;
                            }
                     }
                });
            });
            scale_index_found = ts_0;
        // Task 2: End Frame processing.

        // Task 3:  Frame post-processing.
            // Multi-scale fusion and detection display (note: only a simple fusion scheme implemented
            myQueue.single_task([=](){
                    for(int i = 0; i< N_MAX_STAGES; i++){
                        dev_last_nb_obj_found2[i] = dev_nb_obj_found2[i];
                    }
                    *dev_last_scale_index_found = *dev_scale_index_found;
            });

            myQueue.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, scale_index_found),
                    sycl::range<3>(1, 1, scale_index_found)), [=](sycl::nd_item<3> idx) {
                    k1_reduction(dev_last_scale_index_found, dev_last_nb_obj_found2, idx);
            });

            myQueue.memset(dev_position, 0, width*height*sizeof(uint32_t));
            
            // Keep the position of each circle from the bigger to the smaller
            myQueue.single_task([=](){
                    k2(dev_position, dev_last_scale_index_found, dev_last_nb_obj_found2, dev_goodcenterX, dev_goodcenterY, dev_goodRadius);
            });


            // Delete detections which are too close
            myQueue.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (number_of_threads + 127) / 128) * sycl::range<3>(1, 1, 128),
                            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> idx) {
                            k3(dev_position, dev_last_scale_index_found, dev_nIndexFound, dev_real_width, dev_real_height, idx);
            });

            myQueue.memset(dev_finalNb, 0, 1*sizeof(int));
            myQueue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (irowiterations + 127) / 128) *
                    sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> idx) {

                    // int i = idx[0]*3;
                    int counter = idx.get_local_range(2) * idx.get_group(2) + idx.get_local_id(2);
                    int i = counter * 3;

                    auto v = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(dev_finalNb[0]);

                    if (dev_position[i]!=0)
                    {
                        int c = v.fetch_add(1);
                        printf("[%d] x: %d, y: %d, scale: %d, ValidOutput: %d\n", image_counter, (int)dev_position[i], (int)dev_position[i+1], (int)(dev_position[i+2]/2), c);
                    }
            });

            #if WRITE_IMG
                myQueue.memcpy(position, dev_position, width*height*sizeof(uint32_t)).wait();
                myQueue.memcpy(dev_nb_obj_found2, nb_obj_found2, N_MAX_STAGES*sizeof(uint32_t)).wait();
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

        frame_end = clock();
        float frame_time = (double)(frame_end-frame_start)/CLOCKS_PER_SEC * 1000;
        printf("\n TOTAL Antes WAIT Execution time = %f for %d FRAMES ms.\n", frame_time, (argc-2));

        myQueue.wait();
        myQueue2.wait();
        
        // FREE ALL the allocations
        releaseCascade_continuous(cascade);
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
        free(imgInt_f);
        free(imgSqInt_f);

    // SYCL FREE
    sycl::free(dev_cascade, myQueue);
        sycl::free(dev_feature_scaled, myQueue);

        sycl::free(dev_img, myQueue);
        sycl::free(dev_imgInt, myQueue);
        sycl::free(dev_imgSq, myQueue);
        sycl::free(dev_imgSqInt, myQueue);
        sycl::free(dev_imgInt_f, myQueue);


        sycl::free(dev_goodcenterX_tmp, myQueue);
    sycl::free(dev_goodcenterY_tmp, myQueue);
        sycl::free(dev_goodRadius_tmp, myQueue);

    sycl::free(dev_goodcenterX, myQueue);
    sycl::free(dev_goodcenterY, myQueue);
        sycl::free(dev_goodRadius, myQueue);

        sycl::free(dev_nb_obj_found2, myQueue);
    sycl::free(dev_nb_obj_found, myQueue);
    sycl::free(dev_scale_index_found, myQueue);

        sycl::free(dev_foundObj, myQueue);
        sycl::free(dev_finalNb, myQueue);

        return 0;
}