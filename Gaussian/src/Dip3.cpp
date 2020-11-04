//============================================================================
// Name    : Dip3.cpp
// Author   : Ronny Haensch
// Version    : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip3.h"

// returns a window around the pixel with the coordinates (x, y)
/*
src:            input image
x:              x coordinate of the center of the window
y:              y coordinate of the center of the window
width:          width of the window
height:         height of the window
return:         window
*/
Mat buildWindow(const Mat& src, int x, int y, int width, int height) {
    Mat window = Mat::zeros(height, width, src.type());
    int imageX, imageY;

    for(int i=0; i<window.cols; i++) {
        for(int j=0; j<window.rows; j++) {
            // Handle left & right image border via mirroring
            if(x-width/2+i < 0) {                    // left border
                imageX = x + width/2 + i;
            } else if (x-width/2+i >= src.cols) {    // right border
                imageX = x + width/2 - i;
            } else {                                        // inside the image
                imageX = x - width/2 + i;
            }

            // Handle top & bottom image border via mirroring
            if(y-height/2+j < 0) {                    // top border
                imageY = y + height/2 + j;
            } else if (y-height/2+j >= src.rows) {    // bottom border
                imageY = y + height/2 - j;
            } else {                                        // inside the image
                imageY = y - height/2 + j;
            }

            window.at<float>(j, i) = src.at<float>(imageY, imageX);
        }
    }

    return window;
}


float clip(float num, float low, float high) {
    return max(low, min(num, high));
}


// Generates gaussian filter kernel of given size
/*
kSize:     kernel size (used to calculate standard deviation)
return:    the generated filter kernel
*/
Mat Dip3::createGaussianKernel(int kSize){
    double sigma = kSize / 5.0;
    Mat kernel = Mat::zeros(kSize, kSize, CV_32FC1);
    
    for(int x=0; x<kSize; x++) {
        for(int y=0; y<kSize; y++) {
            kernel.at<float>(y, x) = exp(-(pow(x-kSize/2, 2) + pow(y-kSize/2, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * sigma * sigma);
        }
    }

    // normalize kernel
    kernel /= cv::sum(kernel)[0];

    return kernel;
}


// Performes a circular shift in (dx,dy) direction
/*
in       input matrix
dx       shift in x-direction
dy       shift in y-direction
return   circular shifted matrix
*/
Mat Dip3::circShift(const Mat& in, int dx, int dy){
    Mat output = Mat::zeros(in.size(), CV_32FC1);
    int newX, newY;

    for(int x=0; x<in.cols; x++) {
        newX = (x + dx + in.cols) % in.cols;

        for(int y=0; y<in.rows; y++) {
            newY = (y + dy + in.rows) % in.rows;

            output.at<float>(newY, newX) = in.at<float>(y, x);
        }
    }

    return output;
}

//Performes convolution by multiplication in frequency domain
/*
in       input image
kernel   filter kernel
return   output image
*/
Mat Dip3::frequencyConvolution(const Mat& in, const Mat& kernel){
    Mat kernelLarge = Mat::zeros(in.size(), CV_32FC1);
    Mat input = in.clone();
    Mat output(in.size(), in.type());
    
    // copy kernel to larger matrix
    //copyMakeBorder(kernel, kernelLarge, 0, kernelLarge.rows-kernel.rows, 0, kernelLarge.cols-kernel.cols, BORDER_CONSTANT, 0);
    for(int x=0; x<kernel.cols; x++) {
        for(int y=0; y<kernel.rows; y++) {
            kernelLarge.at<float>(y, x) = kernel.at<float>(y, x);
        }
    }

    // center the kernel
    kernelLarge = circShift(kernelLarge, -kernel.cols/2, -kernel.rows/2);

    // perform dft
    dft(input, input);
    dft(kernelLarge, kernelLarge);

    // calculate convolution
    mulSpectrums(input, kernelLarge, output, 0);

    // perform idft
    dft(output, output, DFT_INVERSE|DFT_SCALE);

    return output;
}

// Performs UnSharp Masking to enhance fine image structures
/*
in       the input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain; 2 <==> seperable filter; 3 <==> integral image
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::usm(const Mat& in, int type, int size, double thresh, double scale){
    Mat output = in.clone();
    Mat tmp(in.size(), CV_32FC1);

    // smooth original image
    switch(type) {
        case 0:         // spatial convolution
            tmp = mySmooth(in, size, 0);
            break;
        case 1:         // convolution in frequency domain
            tmp = mySmooth(in, size, 1);
            break;
        case 2:         // separable filter
            tmp = mySmooth(in, size, 2);
            break;
        case 3:         // integral image
            tmp = mySmooth(in, size, 3);
            break;
        default:        // cv::GaussianBlur
            GaussianBlur(in, tmp, Size(floor(size/2)*2+1, floor(size/2)*2+1), size/5., size/5.);
    }

    // subtract smoothed image from original image
    tmp = in - tmp;

    // thresholding
    threshold(tmp, tmp, thresh, 0, THRESH_TOZERO);

    // scale difference to further enhance edges
    tmp *= scale;

    // add to original image
    output += tmp;

    return output;
}

// convolution in spatial domain
/*
src:    input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip3::spatialConvolution(const Mat& src, const Mat& kernel){
    Mat output = Mat::zeros(src.size(), src.type());
    Mat kernelFlipped = kernel.clone();

    // flip the kernel
    // flip(kernel, kernel, -1);
    for(int x=0; x<kernel.cols; x++) {
        for(int y=0; y<kernel.rows; y++) {
            kernelFlipped.at<float>(y, x) = kernel.at<float>(kernel.rows-y-1, kernel.cols-x-1);
        }
    }

    // perform convolution
    for(int x=0; x<output.cols; x++) {
        for(int y=0; y<output.rows; y++) {
            Mat window = buildWindow(src, x, y, kernel.cols, kernel.rows);
            window = window.mul(kernelFlipped);
            output.at<float>(y, x) = cv::sum(window)[0];
        }
    }

    return output;
}

// convolution in spatial domain by seperable filters
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::seperableFilter(const Mat& src, int size){
    Mat temp = Mat::zeros(src.size(), CV_32FC1);
    Mat output = Mat::zeros(src.size(), CV_32FC1);
    Mat kernel = Mat::zeros(size, 1, CV_32FC1);
    float sigma = size / 5.0;

    // create kernel
    for(int x=0; x<size; x++) {
        kernel.at<float>(0, x) = exp(-pow(x-size/2, 2) / (2 * pow(sigma, 2))) / (sqrt(2*CV_PI) * sigma);
    }

    // normalize kernel
    kernel /= cv::sum(kernel)[0];

    // convolution with column vector
    for(int x=0; x<output.cols; x++) {
        for(int y=0; y<output.rows; y++) {
            Mat window = buildWindow(src, x, y, 1, size);
            window = window.mul(kernel);
            
            temp.at<float>(y, x) = cv::sum(window)[0];
        }
    }

    // transpose kernel
    kernel = kernel.t();

    // convolution with row vector
    for(int x=0; x<output.cols; x++) {
        for(int y=0; y<output.rows; y++) {
            Mat window = buildWindow(temp, x, y, size, 1);
            window = window.mul(kernel);
            output.at<float>(y, x) = cv::sum(window)[0];
        }
    }

    return output;
}

// convolution in spatial domain by integral images
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::satFilter(const Mat& src, int size){
    Mat input = src.clone() / pow(size, 2);
    Mat output = Mat::zeros(src.size(), CV_32FC1);
    Mat integralImage = Mat::zeros(src.size(), CV_32FC1);
    float topLeft, topRight, bottomLeft, bottomRight;

    integral(input, integralImage, CV_32F);

    for(int x=1; x<=output.cols; x++) {
        for(int y=1; y<=output.rows; y++) {
            topLeft = integralImage.at<float>(clip(y-size/2-1, 0, src.rows), clip(x-size/2-1, 0, src.cols));
            topRight = integralImage.at<float>(clip(y-size/2-1, 0, src.rows), clip(x+size/2, 0, src.cols));
            bottomLeft = integralImage.at<float>(clip(y+size/2, 0, src.rows), clip(x-size/2-1, 0, src.cols));
            bottomRight = integralImage.at<float>(clip(y+size/2, 0, src.rows), clip(x+size/2, 0, src.cols));

            // calculate scale factor to fix the color for pixels near the edges
            float scale = (min(x-1+size/2, src.cols-1) - max(x-1-size/2, 0) + 1) * (min(y-1+size/2, src.rows-1) - max(y-1-size/2, 0) + 1) / pow(size, 2);

            output.at<float>(y-1, x-1) = (bottomRight + topLeft - topRight - bottomLeft) / scale;
        }
    }

    return output;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in       input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::run(const Mat& in, int smoothType, int size, double thresh, double scale){

   return usm(in, smoothType, size, thresh, scale);

}


// Performes smoothing operation by convolution
/*
in       input image
size     size of filter kernel
type     how is smoothing performed?
return   smoothed image
*/
Mat Dip3::mySmooth(const Mat& in, int size, int type){

   // create filter kernel
   Mat kernel = createGaussianKernel(size);
 
   // perform convoltion
   switch(type){
     case 0: return spatialConvolution(in, kernel);	// 2D spatial convolution
     case 1: return frequencyConvolution(in, kernel);	// 2D convolution via multiplication in frequency domain
     case 2: return seperableFilter(in, size);	// seperable filter
     case 3: return satFilter(in, size);		// integral image
     default: return frequencyConvolution(in, kernel);
   }
}

// function calls basic testing routines to test individual functions for correctness
void Dip3::test(void){

   test_createGaussianKernel();
   test_circShift();
   test_frequencyConvolution();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip3::test_createGaussianKernel(void){

   Mat k = createGaussianKernel(11);
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel(): Sum of all kernel elements is not one!" << endl;
      return;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel(): Seems like kernel is not centered!" << endl;
      return;
   }
   cout << "Message: Dip3::createGaussianKernel() seems to be correct" << endl;
}

void Dip3::test_circShift(void){
   
   Mat in = Mat::zeros(3,3,CV_32FC1);
   in.at<float>(0,0) = 1;
   in.at<float>(0,1) = 2;
   in.at<float>(1,0) = 3;
   in.at<float>(1,1) = 4;
   Mat ref = Mat::zeros(3,3,CV_32FC1);
   ref.at<float>(0,0) = 4;
   ref.at<float>(0,2) = 3;
   ref.at<float>(2,0) = 2;
   ref.at<float>(2,2) = 1;
   
   if (sum((circShift(in, -1, -1) == ref)).val[0]/255 != 9){
      cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip3::circShift() seems to be correct" << endl;
}

void Dip3::test_frequencyConvolution(void){
   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = frequencyConvolution(input, kernel);
   
   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
}
