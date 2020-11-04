//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip5.h"

// uses structure tensor to define interest points (foerstner)
void Dip5::getInterestPoints(const Mat& img, double sigma, vector<KeyPoint>& points){
    Mat gradX, gradY, gradXX, gradYY, gradXY, weight, isotropy, trSq;
    Mat tr = Mat::zeros(img.size(), CV_32FC1);
    Mat det = Mat::zeros(img.size(), CV_32FC1);

    Mat kernel = createFstDevKernel(sigma);

    // 1. Gradient in x- and y-direction
    filter2D(img, gradX, -1, kernel);
    filter2D(img, gradY, -1, kernel.t());

    showImage(gradX, "01_gradX", -1, false, true);
    showImage(gradY, "01_gradY", -1, false, true);

    // 2. gradX * gradX, gradY * gradY, gradX * gradY
    multiply(gradX, gradX, gradXX);
    multiply(gradY, gradY, gradYY);
    multiply(gradX, gradY, gradXY);

    showImage(gradXX, "02_gradXX", -1, false, true);
    showImage(gradYY, "02_gradYY", -1, false, true);
    showImage(gradXY, "02_gradXY", -1, false, true);

    // 3. Average (Gaussian Window)
    GaussianBlur(gradXX, gradXX, Size(0, 0), sigma);
    GaussianBlur(gradYY, gradYY, Size(0, 0), sigma);
    GaussianBlur(gradXY, gradXY, Size(0, 0), sigma);

    showImage(gradXX, "03_gradXX_blur", -1, false, true);
    showImage(gradYY, "03_gradYY_blur", -1, false, true);
    showImage(gradXY, "03_gradXY_blur", -1, false, true);

    // 4. Trace of structure tensor
    for(int x=0; x<tr.cols; x++) {
        for(int y=0; y<tr.rows; y++) {
            tr.at<float>(y, x) = gradXX.at<float>(y, x) + gradYY.at<float>(y, x);
        }
    }

    showImage(tr, "04_trace", -1, false, true);

    // 5. Determinant of structure tensor
    for(int x=0; x<det.cols; x++) {
        for(int y=0; y<det.rows; y++) {
            det.at<float>(y, x) = gradXX.at<float>(y, x) * gradYY.at<float>(y, x) - 2 * gradXY.at<float>(y, x);
        }
    }

    showImage(det, "05_determinant", -1, false, true);

    // 6. Weight calculation
    divide(det, tr, weight);

    showImage(weight, "06_weight", -1, false, true);

    // 7. Weight non-max suppression
    weight = nonMaxSuppression(weight);

    showImage(weight, "07_weight_suppressed", -1, false, true);

    // 8. Weight tresholding
    double thresh = 50 * mean(weight)[0];
    threshold(weight, weight, thresh, 255, THRESH_BINARY);

    showImage(weight, "08_weight_threshold", -1, false, true);

    // 9. Isotropy calculation
    multiply(tr, tr, trSq);
    divide(4*det, trSq, isotropy);

    showImage(isotropy, "09_isotropy", -1, false, true);

    // 10. Isotropy non-max suppression
    isotropy = nonMaxSuppression(isotropy);

    showImage(isotropy, "10_isotropy_suppressed", -1, false, true);

    // 11. Isotropy thresholding
    threshold(isotropy, isotropy, 0.75, 255, THRESH_BINARY);

    showImage(isotropy, "11_isotropy_threshold", -1, false, true);

    // 12. Keypoints found
    for(int x=0; x<weight.cols; x++) {
        for(int y=0; y<weight.rows; y++) {
            if(weight.at<float>(y, x) == 255 && isotropy.at<float>(y, x) == 255) {
                points.push_back(KeyPoint(Point(x, y), 1.f));
            }
        }
    }
}

// creates kernel representing fst derivative of a Gaussian kernel in x-direction
/*
sigma	standard deviation of the Gaussian kernel
return	the calculated kernel
*/
Mat Dip5::createFstDevKernel(double sigma){
    // sigma = 0.3*((kSize-1)*0.5 - 1) + 0.8
    int kSize = max(0, (int)((sigma - 0.8) / 0.15)) / 2 * 2 + 3;
    Mat kernel = Mat::zeros(kSize, kSize, CV_32FC1);
    
    for(int x=0; x<kSize; x++) {
        for(int y=0; y<kSize; y++) {
            kernel.at<float>(y, x) = -(x-kSize/2) * exp(-(pow(x-kSize/2, 2) + pow(y-kSize/2, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 4));
        }
    }

    return kernel;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in		:  input image
points	:	detected keypoints
*/
void Dip5::run(const Mat& in, vector<KeyPoint>& points){
   this->getInterestPoints(in, this->sigma, points);
}

// non-maxima suppression
// if any of the pixel at the 4-neighborhood is greater than current pixel, set it to zero
Mat Dip5::nonMaxSuppression(const Mat& img){

    Mat out = img.clone();
    
    for(int x=1; x<out.cols-1; x++){
        for(int y=1; y<out.rows-1; y++){
            if ( img.at<float>(y-1, x) >= img.at<float>(y, x) ){
                out.at<float>(y, x) = 0;
                continue;
            }
            if ( img.at<float>(y, x-1) >= img.at<float>(y, x) ){
                out.at<float>(y, x) = 0;
                continue;
            }
            if ( img.at<float>(y, x+1) >= img.at<float>(y, x) ){
                out.at<float>(y, x) = 0;
                continue;
            }
            if ( img.at<float>( y+1, x) >= img.at<float>(y, x) ){
                out.at<float>(y, x) = 0;
                continue;
            }
        }
    }
    return out;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void Dip5::showImage(const Mat& img, const char* win, int wait, bool show, bool save){
  
    Mat aux = img.clone();

    // scale and convert
    if (img.channels() == 1)
        normalize(aux, aux, 0, 255, CV_MINMAX);
        aux.convertTo(aux, CV_8UC1);
    // show
    if (show){
      imshow( win, aux);
      waitKey(wait);
    }
    // save
    if (save)
      imwrite( (string(win)+string(".png")).c_str(), aux);
}
