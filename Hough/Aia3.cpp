//============================================================================
// Name        : Aia3.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia3.h"


void Aia3::plotHough(vector< vector<Mat> > const& houghSpace){
   Mat houghImage = Mat::zeros(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1);

    for (int i = 0; i < houghSpace.size(); ++i) {
        for (int j = 0; j < houghSpace.at(i).size(); ++j) {
            Mat current = houghSpace.at(i).at(j);
            for (int r = 0; r < current.rows; ++r) {
                for (int c = 0; c < current.cols; ++c) {
                    houghImage.at<float>(r, c) += current.at<float>(r, c);
                }
            }
        }
    }
    //normalize(houghImage, houghImage, 0, 1, CV_MINMAX);
    showImage(houghImage, "Hough Space", 0);
	imwrite( "houghspacetwo.png", houghImage);
//imwrite(g_test_flag+"8_Hough.png", houghImage*255);
}

// creates the fourier-spectrum of the scaled and rotated template
/*
  templ:	the object template; binary image in templ[0], complex gradient in templ[1]
  scale:	the scale factor to scale the template
  angle:	the angle to rotate the template
  fftMask:	the generated fourier-spectrum of the template
*/
void Aia3::makeFFTObjectMask(vector<Mat> const& templ, double scale, double angle, Mat& fftMask){
   // DONE !!!
	
	//cout << "Dimensionen der fftMask:" << fftMask.rows << " x " << fftMask.cols << endl;
	//cout << "fftMask-type:" << fftMask.type()<< endl;
	
	//angle = -CV_PI/4;
	//scale = 0.5;
	
	//get familiar with the given objects:
   vector<Mat> gradient(2);
   cv::split(templ[1], gradient);
  
   //showImage(templ[0],"binary_edge",0);
   //showImage(gradient[0],"gradientx",0);
   //showImage(gradient[1],"gradienty",0);
   

   Mat edge_gradient_x = Mat(gradient[0].rows, gradient[0].cols,gradient[0].type());
   Mat edge_gradient_y = Mat(gradient[0].rows, gradient[0].cols,gradient[0].type());



	//1. Get Oi*Ob: i.e. multiply the gradient images with the binary edge image:
	cv::multiply(gradient[0],templ[0],edge_gradient_x);
	cv::multiply(gradient[1],templ[0],edge_gradient_y);
	
	//show intermediate result:
     //showImage(edge_gradient_x,"gradientx",0);
     // showImage(edge_gradient_y,"gradienty",0);




   //2. rotate and scale the result
   Mat rotated_binary, rotated_x, rotated_y;

   rotated_x = rotateAndScale(edge_gradient_x, angle, scale);
   rotated_y = rotateAndScale(edge_gradient_y, angle, scale);
   
	//show intermediate Result:
   //showImage(rotated_x, "rotated, before gradient-correction",0);



   //3. correct magnitude of gradients directions: 
   Mat magn, phase = Mat(rotated_x.rows, rotated_x.cols, rotated_x.type());
   
    cv::cartToPolar(rotated_x, rotated_y, magn, phase);  
	Mat shift =  Mat::ones(phase.rows,phase.cols, phase.type())*((angle));    
	cv::add(phase,shift,phase);
	cv::polarToCart(magn, phase, rotated_x, rotated_y);

    //show intermediate Result:
	//showImage(rotated_x, "rotated, after gradient-correction",0);



   //4. normalize it:
   double normalisation_const = cv::sum(magn)[0];
   rotated_x = rotated_x/normalisation_const;
   rotated_y = rotated_y/normalisation_const;
  
	//show intermediate Result:
   //showImage(rotated_x, "rotated, after normalisation",0);
   
   /*
    //EXERCISE: access a two channel matrix:
	 * 
	 * 
    Mat a = Mat::zeros(10,10, CV_32FC1);
    Mat b = Mat::ones(10,10, CV_32FC1);
    
    int i = 0;
    
	for(int c = 0; c< a.cols; c++){
		for(int r = 0; r< a.rows; r++){
			a.at<float>(r,c) = i;
			i++;
		}
		}
    
    //cout << "matrix a:"<< endl;
	//cout << a<<endl;
	
    
    vector<Mat> vec;
    vec.push_back(a);
    vec.push_back(b);
   
	

	Mat merged;
	merge(vec, merged);

	
	cout << "type of merged matrix:" << merged.type()<< endl; 
	
	cout << merged << endl;  
	
	cout <<"Access at (9,0): " <<merged.at<Vec2f>(9,0)<< endl;
	cout <<"Access at (0,9): " <<merged.at<Vec2f>(0,9)<< endl;
	cout <<"Access at (2,2): " <<merged.at<Vec2f>(2,2)<< endl;
   */
      
   
	// 5.copy it into a larger objekt (the fftmask):
	for(int r = 0; r < rotated_x.rows; r++){
		for(int c = 0; c < rotated_x.cols; c++){
			fftMask.at<Vec2f>(r,c)[0] = rotated_x.at<float>(r,c);
			fftMask.at<Vec2f>(r,c)[1] = rotated_y.at<float>(r,c); 
		}
	}
   
	//show intermediate result:
	Mat planes[2];
	cv::split(fftMask, planes);
	//showImage(planes[0], "after copying in fftMask-Object",0);



   //6. center it:
   circShift(fftMask, fftMask, -rotated_x.rows/2, -rotated_x.cols/2);
   
   //show intermediate result:
   cv::split(fftMask, planes);
   //showImage(planes[0], "after circular shift",0);



   //7. transform it to frequency-domain:
	cv::dft(fftMask, fftMask);
}

// computes the hough space of the general hough transform
/*
  gradImage:	the gradient image of the test image
  templ:		the template consisting of binary image and complex-valued directional gradient image
  scaleSteps:	scale resolution
  scaleRange:	range of investigated scales [min, max]
  angleSteps:	angle resolution
  angleRange:	range of investigated angles [min, max)
  return:		the hough space: outer vector over scales, inner vector of angles
*/
vector< vector<Mat> > Aia3::generalHough(Mat const& gradImage, vector<Mat> const& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange){
   // TODO !!!
	double scaleMin = scaleRange[0];
    double scaleMax = scaleRange[1];
    double angleMin = angleRange[0];
    double angleMax = angleRange[1];
    
    double scalestep = (scaleMax - scaleMin)/ scaleSteps;
    double anglestep = (angleMax - angleMin)/ angleSteps;
    
    cout << "scalestep: " << scalestep << endl;
    cout << "scale-min: "<< scaleMin<<endl;
	cout << "scale-max: "<< scaleMax<<endl;
	cout << "number of steps:"<< scaleSteps << endl;
	
	cout << "anglestep: " << anglestep << endl;
    cout << "angle-min: "<< angleMin<<endl;
	cout << "scale-max: "<< angleMax<<endl;
	cout << "number of angle-steps:"<< angleSteps<< endl;
    

  //calculate convolution:
	Mat grad_fft;
	dft(gradImage, grad_fft);			
	
	
	
	/*//THIS PART the hough vote for fixed angle and scale!
	double angle = 30/180.*CV_PI;
	double scale = 0.5;
	makeFFTObjectMask(templ, scale, angle,fftMask);
	
	
	//show intermediate results:
	Mat fft[2];
	Mat test[2];
	cv::split(fftMask, fft);
	cv:split(grad_fft, test);
	showImage(fft[0], "fftmask", 0);
	showImage(test[0], "gradImage", 0);

	Mat convoluted;


	mulSpectrums(grad_fft, grad_fft, convoluted, 0, true);
	//show intermediate result:
	Mat planes[2];
	//split(convoluted, planes);
	//showImage(planes[0], "real-part", 0 ); //real-part
	//showImage(planes[1], "imaginary", 0 ); //imaginary-part
	
	//set imaginary-part to zero:
	//planes[1] = Mat::zeros(planes[1].rows, planes[1].cols, planes[1].type());
			
			//merge it again:
	//		vector<Mat> vec;
	//		vec.push_back(planes[0]);
	//		vec.push_back(planes[1]);
	//		merge(vec, convoluted);
	
	Mat hough ;
	dft(convoluted,hough, DFT_INVERSE | DFT_SCALE);
	
	//cout<< planes[0].rows << " x "<< planes[0].cols <<endl;
	//cout<< planes[0].at<float>(0,0)<< endl;
	//cout<< planes[0].at<float>(90,90)<< endl;
	
	//Calculate absolute value of each real-entry:
	split(hough, planes);
	Mat testhough = abs(planes[0]);
	
	//dft(hough, hough, DFT_INVERSE | DFT_SCALE);
	showImage(testhough, "hough-test", 0 ); 
*/

	
	/*EXERCISE create and iterate over a matric, composed as a vector as vectors:
	vector <vector<double> > matrix;
	scale = scaleMin;
	angle = angleMin;
	
	
	//compute convolution:
	//for each scale in scalerange:
	while(scale <= scaleMax){
		vector<double> temp;
		//for each rotation in rotationrange:
		while(angle <= angleMax){
			temp.push_back(scale+angle);
			angle = angle + anglestep;
		}
		
		matrix.push_back(temp);
		scale = scale + scalestep;
		angle = angleMin;
	}
	iterate over all the indices:
	for(int s = 0; s< matrix.size(); s++){
		for(int a = 0; a< matrix[s].size(); a++){
				cout <<matrix[s][a] << " ";
			}
		cout << endl;
		}
		 */


//COMPUTE THE HOUGH SPACE:
	vector <vector<Mat> > houghSpace;
	vector<Mat> temp;
	double scale = scaleMin;
	double angle  = angleMin;
	
	//scale = scaleMin;
	//angle  = angleMin;
	
	Mat convoluted, hough, fftMask;
	
	//for each scale in scalerange:
	for(int i = 0; i < scaleSteps; ++i){
		scale = scale + scalestep;
		//for each rotation in rotationrange:
		for(int j = 0; j < angleSteps; ++j){
			angle = angleMin + anglestep;
			//compute Mask for given scale and rotation:
			fftMask= Mat::zeros(grad_fft.rows, grad_fft.cols, grad_fft.type());
			makeFFTObjectMask(templ, scale, angle,fftMask);
			
			//compute convolution with the objectmask:
			mulSpectrums(grad_fft, fftMask, convoluted, 0, true );
			
			dft(convoluted, convoluted, DFT_INVERSE);
			
			Mat hough(convoluted.rows, convoluted.cols, CV_32FC1);
            for (int r = 0; r < hough.rows; ++r) {
                for (int c = 0; c < hough.cols; ++c) {
                    hough.at<float>(r, c) = abs(convoluted.at<Vec2f>(r, c)[0]);
				}
			}
			temp.push_back(hough);
			
		}
		houghSpace.push_back(temp);
		//angle = angleMin;
		
	} 
	
	/*
	//for each scale in scalerange:
	for(int i = 0; i < scaleSteps; ++i){
		scale = scale + scalestep;
		//for each rotation in rotationrange:
		for(int j = 0; j < angleSteps; ++j){
			angle = angleMin + anglestep;
			//compute Mask for given scale and rotation:
			fftMask= Mat::zeros(grad_fft.rows, grad_fft.cols, grad_fft.type());
			makeFFTObjectMask(templ, scale, angle,fftMask);
			
			//compute convolution with the objectmask:
			mulSpectrums(grad_fft, fftMask, convoluted, 0, true );
			
			Mat planes[2];
			split(convoluted, planes);
			
			planes[1] = Mat::zeros(planes[1].rows, planes[1].cols, planes[1].type());
			
			//merge it again:
			vector<Mat> vec;
			vec.push_back(planes[0]);
			vec.push_back(planes[1]);
			merge(vec, hough);
			
			
			dft(hough, hough, DFT_INVERSE);
			
			split(hough,planes);
			
			temp.push_back(abs(planes[0]));
			
		}
		houghSpace.push_back(temp);
		//angle = angleMin;
	}
	 */

    return houghSpace;
}

// creates object template from template image
/*
  templateImage:	the template image
  sigma:			standard deviation of directional gradient kernel
  templateThresh:	threshold for binarization of the template image
  return:			the computed template
*/
vector<Mat> Aia3::makeObjectTemplate(Mat const& templateImage, double sigma, double templateThresh){
   //get complex gradients:
    vector<Mat> gradient(2);
    cv::split(calcDirectionalGrad(templateImage, sigma), gradient);
   
   //showImage(gradient[0], "Gradient in x-direction", 0);
   //showImage(gradient[1], "Gradient in y-direction", 0);
   //cout << gradient[0].rows << gradient[0].cols << endl;
   
   //get binary edges:
	Mat  magn(gradient[0].rows, gradient[0].cols, gradient[0].type());
	Mat  binary_edge(gradient[0].rows, gradient[0].cols, CV_8UC2);
	
	
	//the easy way:
	magnitude(gradient[0],gradient[1],magn);
	
	//calculate maximum magnitude:
	double max;
	minMaxLoc(magn, NULL, &max, NULL, NULL);
	//cout<< "maximum: "<< max<< endl;
	
	//calculate threshold:
	int thresh = templateThresh*max;
	
	//showImage(binary_edge,"before_threshold",0);
	threshold(magn, binary_edge, thresh, 1, THRESH_BINARY); 
	
	//showImage(binary_edge,"binary_edge",0);
	
	//combine both:
	vector<Mat> result;
	result.push_back(binary_edge);
	result.push_back(calcDirectionalGrad(templateImage, sigma));
    return result;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// loads template and test images, sets parameters and calls processing routine
/*
tmplImg:	path to template image
testImg:	path to test image
*/
void Aia3::run(string tmplImg, string testImg){

    // processing parameter
    double sigma 			= 1;		// standard deviation of directional gradient kernel
    double templateThresh 	= 0.3;		// relative threshold for binarization of the template image
    // TO DO !!!
    // ****
    // Set parameters to reasonable values
    double objThresh 		= 0.9;		// relative threshold for maxima in hough space
    double scaleSteps 		= 20;		// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];				// scale of angles [min, max]
    scaleRange[0] 			= 0.5;
    scaleRange[1] 			= 2;
    double angleSteps 		= 20;		// angle resolution in terms of number of angles to be investigated
    double angleRange[2];				// range of angles [min, max)
    angleRange[0] 			= 0;
    angleRange[1] 			= 2*CV_PI;
    // ****
	
    Mat params = (Mat_<float>(1,9) << sigma, templateThresh, objThresh, scaleSteps, scaleRange[0], scaleRange[1], angleSteps, angleRange[0], angleRange[1]);
  
    // load template image as gray-scale, paths in argv[1]
    Mat templateImage = imread( tmplImg, 0);
    if (!templateImage.data){
		cerr << "ERROR: Cannot load template image from\n" << tmplImg << endl;
	    cerr << "Continue with pressing any key... (if you can't find the \"any\"-key, press enter)" << endl;
	    cin.get();
		exit(-1);
    }
    // convert 8U to 32F
    templateImage.convertTo(templateImage, CV_32FC1);
    // show template image
    showImage(templateImage, "Template image", 0);
    
    // load test image
    Mat testImage = imread( testImg, 0);
	if (!testImage.data){
		cerr << "ERROR: Cannot load test image from\n" << testImg << endl;
	    cerr << "Continue with pressing any key... (if you can't find the \"any\"-key, press enter)" << endl;
	    cin.get();
		exit(-1);
	}
	// and convert it from 8U to 32F
	testImage.convertTo(testImage, CV_32FC1);
    // show test image
    showImage(testImage, "testImage", 0);
    
    // start processing
    process(templateImage, testImage, params);
    
}

// loads template and create test image, sets parameters and calls processing routine
/*
tmplImg:	path to template image
angle:		rotation angle in degree of the test object
scale:		scale of the test object
*/
void Aia3::test(string tmplImg, float angle, float scale){

	// angle to rotate template image (in radian)
	double testAngle = angle/180.*CV_PI;
	// scale to scale template image
	double testScale = scale;

    // processing parameter
    double sigma 			= 1;		// standard deviation of directional gradient kernel
    double templateThresh 	= 0.9;		// relative threshold for binarization of the template image
    double objThresh		= 0.85;		// relative threshold for maxima in hough space
    double scaleSteps 		= 3;		// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];				// scale of angles [min, max]
	scaleRange[0] 			= 1;
	scaleRange[1] 			= 2;
    double angleSteps 		= 12;		// angle resolution in terms of number of angles to be investigated
    double angleRange[2];				// range of angles [min, max)
	angleRange[0] 			= 0;
	angleRange[1] 			= 2*CV_PI;
	
	Mat params = (Mat_<float>(1,9) << sigma, templateThresh, objThresh, scaleSteps, scaleRange[0], scaleRange[1], angleSteps, angleRange[0], angleRange[1]);
		  
    // load template image as gray-scale, paths in argv[1]
    Mat templateImage = imread( tmplImg, 0);
    if (!templateImage.data){
		cerr << "ERROR: Cannot load template image from\n" << tmplImg << endl;
		cerr << "Continue with pressing any key... (if you can't find the \"any\"-key, press enter)" << endl;
	    cin.get();
		exit(-1);
    }
    // convert 8U to 32F
    templateImage.convertTo(templateImage, CV_32FC1);
    // show template image
    showImage(templateImage, "Template Image", 0);
    
    // generate test image
    Mat testImage = makeTestImage(templateImage, testAngle, testScale, scaleRange);
    // show test image
    showImage(testImage, "Test Image", 0);
	
	// start processing
    process(templateImage, testImage, params);
}
    
void Aia3::process(Mat const& templateImage, Mat const& testImage, Mat const& params){
	
	// processing parameter
    double sigma			= params.at<float>(0);		// standard deviation of directional gradient kernel
    double templateThresh 	= params.at<float>(1);		// relative threshold for binarization of the template image
    double objThresh 		= params.at<float>(2);		// relative threshold for maxima in hough space
    double scaleSteps 		= params.at<float>(3);		// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];								// scale of angles [min, max]
	scaleRange[0] 			= params.at<float>(4);
	scaleRange[1] 			= params.at<float>(5);
    double angleSteps 		= params.at<float>(6);		// angle resolution in terms of number of angles to be investigated
	double angleRange[2];								// range of angles [min, max)
    angleRange[0] 			= params.at<float>(7);
	angleRange[1] 			= params.at<float>(8);

	// calculate directional gradient of test image as complex numbers (two channel image)
    Mat gradImage = calcDirectionalGrad(testImage, sigma);
    
    // generate template from template image
    // templ[0] == binary image
    // templ[0] == directional gradient image
    vector<Mat> templ = makeObjectTemplate(templateImage, sigma, templateThresh);
    
    // show binary image
    //showImage(templ[0], "Binary part of template", 0);
    
    //Mat fftMask;
    //makeFFTObjectMask(templ, 0.5, 0, fftMask);
    //show gradient of test-image:
    Mat test[2];
    split(gradImage, test);
    showImage(test[0],"testimage",0);

    // perfrom general hough transformation
    vector< vector<Mat> > houghSpace = generalHough(gradImage, templ, scaleSteps, scaleRange, angleSteps, angleRange);
	
	// plot hough space (max over angle- and scale-dimension)
	plotHough(houghSpace);
    
    // find maxima in hough space
    vector<Scalar> objList;
    findHoughMaxima(houghSpace, objThresh, objList);

    // print found objects on screen
    cout << "Number of objects: " << objList.size() << endl;
    int i=0;
		for(vector<Scalar>::iterator it = objList.begin(); it != objList.end(); it++, i++){
		cout << i << "\tScale:\t" << (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
		cout << "\tAngle:\t" << ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0])/CV_PI*180;
		cout << "\tPosition:\t(" << (*it).val[2] << ", " << (*it).val[3] << " )" << endl;
    }

    // show final detection result
    plotHoughDetectionResult(testImage, templ, objList, scaleSteps, scaleRange, angleSteps, angleRange);

}
// computes directional gradients
/*
  image:	the input image
  sigma:	standard deviation of the kernel
  return:	the two-channel gradient image
*/
Mat Aia3::calcDirectionalGrad(Mat const& image, double sigma){

  // compute kernel size
  int ksize = max(sigma*3,3.);
  if (ksize % 2 == 0)  ksize++;
  double mu = ksize/2.0;

  // generate kernels for x- and y-direction
  double val, sum=0;
  Mat kernel(ksize, ksize, CV_32FC1);
  //Mat kernel_y(ksize, ksize, CV_32FC1);
  for(int i=0; i<ksize; i++){
      for(int j=0; j<ksize; j++){
		val  = pow((i+0.5-mu)/sigma,2);
		val += pow((j+0.5-mu)/sigma,2);
		val = exp(-0.5*val);
		sum += val;
		kernel.at<float>(i, j) = -(j+0.5-mu)*val;
     }
  }
  kernel /= sum;
  // use those kernels to compute gradient in x- and y-direction independently
  vector<Mat> grad(2);
  filter2D(image, grad[0], -1, kernel);
  filter2D(image, grad[1], -1, kernel.t());
  // combine both real-valued gradient images to a single complex-valued image
  Mat output;
  merge(grad, output);
  
  return output; 
}

// rotates and scales a given image
/*
  image:	the image to be scaled and rotated
  angle:	rotation angle in radians
  scale:	scaling factor
  return:	transformed image
*/
Mat Aia3::rotateAndScale(Mat const& image, double angle, double scale){
    
    // create transformation matrices
    // translation to origin
    Mat T = Mat::eye(3, 3, CV_32FC1);    
    T.at<float>(0, 2) = -image.cols/2.0;
    T.at<float>(1, 2) = -image.rows/2.0;
    // rotation
    Mat R = Mat::eye(3, 3, CV_32FC1);
    R.at<float>(0, 0) =  cos(angle);
    R.at<float>(0, 1) = -sin(angle);
    R.at<float>(1, 0) =  sin(angle);
    R.at<float>(1, 1) =  cos(angle);
    // scale
    Mat S = Mat::eye(3, 3, CV_32FC1);    
    S.at<float>(0, 0) = scale;
    S.at<float>(1, 1) = scale;
    // combine
    Mat H = R*S*T;

    // compute corners of warped image
    Mat corners(1, 4, CV_32FC2);
    corners.at<Vec2f>(0, 0) = Vec2f(0,0);
    corners.at<Vec2f>(0, 1) = Vec2f(0,image.rows);
    corners.at<Vec2f>(0, 2) = Vec2f(image.cols,0);
    corners.at<Vec2f>(0, 3) = Vec2f(image.cols,image.rows);
    perspectiveTransform(corners, corners, H);
    
    // compute size of resulting image and allocate memory
    float x_start = min( min( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), min( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float x_end   = max( max( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), max( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float y_start = min( min( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), min( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );
    float y_end   = max( max( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), max( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );
       
    // create translation matrix in order to copy new object to image center
    T.at<float>(0, 0) = 1;
    T.at<float>(1, 1) = 1;
    T.at<float>(2, 2) = 1;
    T.at<float>(0, 2) = (x_end - x_start + 1)/2.0;
    T.at<float>(1, 2) = (y_end - y_start + 1)/2.0;
    
    // change homography to take necessary translation into account
    H = T * H;
    // warp image and copy it to output image
    Mat output;
    warpPerspective(image, output, H, Size(x_end - x_start + 1, y_end - y_start + 1), CV_INTER_LINEAR);

    return output;
  
}

// generates the test image as a transformed version of the template image
/*
  temp:		the template image
  angle:	rotation angle
  scale:	scaling factor
  scaleRange:	scale range [min,max], used to determine the image size
*/
Mat Aia3::makeTestImage(Mat const& temp, double angle, double scale, double* scaleRange){
 
    // rotate and scale template image
    Mat small = rotateAndScale(temp, angle, scale);
    
    // create empty test image
    Mat testImage = Mat::zeros(temp.rows*scaleRange[1]*2, temp.cols*scaleRange[1]*2, CV_32FC1);
   

    // copy new object into test image
    Mat tmp;
    Rect roi;
    roi = Rect( (testImage.cols - small.cols)*0.5, (testImage.rows - small.rows)*0.5, small.cols, small.rows);
    tmp = Mat(testImage, roi);
    small.copyTo(tmp);

    return testImage;
}

// shows the detection result of the hough transformation
/*
  testImage:	the test image, where objects were searched (and hopefully found)
  templ:		the template consisting of binary image and complex-valued directional gradient image
  objList:		list of objects as defined by findHoughMaxima(..)
  scaleSteps:	scale resolution
  scaleRange:	range of investigated scales [min, max]
  angleSteps:	angle resolution
  angleRange:	range of investigated angles [min, max)
*/
void Aia3::plotHoughDetectionResult(Mat const& testImage, vector<Mat> const& templ, vector<Scalar> const& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange){

    // some matrices to deal with color
    Mat red = testImage.clone();
    Mat green = testImage.clone();
    Mat blue = testImage.clone();
    Mat tmp = Mat::zeros(testImage.rows, testImage.cols, CV_32FC1);
    
    // scale and angle of current object
    double scale, angle;
    
    // for all objects
    for(vector<Scalar>::const_iterator it = objList.begin(); it != objList.end(); it++){
		// compute scale and angle of current object
		scale = (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
		angle = ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0]);    
		
		// use scale and angle in order to generate new binary mask of template
		Mat binMask = rotateAndScale(templ[0], angle, scale);

		// perform boundary checks
		Rect binArea = Rect(0, 0, binMask.cols, binMask.rows);
		Rect imgArea = Rect((*it).val[2]-binMask.cols/2., (*it).val[3]-binMask.rows/2, binMask.cols, binMask.rows);
		if ( (*it).val[2]-binMask.cols/2 < 0 ){
			binArea.x = abs( (*it).val[2]-binMask.cols/2 );
			binArea.width = binMask.cols - binArea.x;
			imgArea.x = 0;
			imgArea.width = binArea.width;
		}
		if ( (*it).val[3]-binMask.rows/2 < 0 ){
			binArea.y = abs( (*it).val[3]-binMask.rows/2 );
			binArea.height = binMask.rows - binArea.y;
			imgArea.y = 0;
			imgArea.height = binArea.height;
		}
		if ( (*it).val[2]-binMask.cols/2 + binMask.cols >= tmp.cols ){
			binArea.width = binMask.cols - ( (*it).val[2]-binMask.cols/2 + binMask.cols - tmp.cols );
			imgArea.width = binArea.width;
		}
		if ( (*it).val[3]-binMask.rows/2 + binMask.rows >= tmp.rows ){
			binArea.height = binMask.rows - ( (*it).val[3]-binMask.rows/2 + binMask.rows - tmp.rows );
			imgArea.height = binArea.height;
		}
		// copy this object instance in new image of correct size
		tmp.setTo(0);
		Mat binRoi = Mat(binMask, binArea);
		Mat imgRoi = Mat(tmp, imgArea);
		binRoi.copyTo(imgRoi);

		// delete found object from original image in order to reset pixel values with red (which are white up until now)
		binMask = 1 - binMask;
		imgRoi = Mat(red, imgArea);
		multiply(imgRoi, binRoi, imgRoi);
		imgRoi = Mat(green, imgArea);
		multiply(imgRoi, binRoi, imgRoi);
		imgRoi = Mat(blue, imgArea);
		multiply(imgRoi, binRoi, imgRoi);

		// change red channel
		red = red + tmp*255;
    }
    // generate color image
    vector<Mat> color;
    color.push_back(blue);
    color.push_back(green);
    color.push_back(red);
    Mat display;
    merge(color, display);
    // display color image
    showImage(display, "result", 0);
    // save color image
    imwrite("detectionResult.png", display);
}

// seeks for local maxima within the hough space
/*
  a local maxima has to be larger than all its 8 spatial neighbors, as well as the largest value at this position for all scales and orientations
  houghSpace:	the computed hough space
  objThresh:	relative threshold for maxima in hough space
  objList:	list of detected objects
*/
void Aia3::findHoughMaxima(vector< vector<Mat> > const& houghSpace, double objThresh, vector<Scalar>& objList){

    // get maxima over scales and angles
    Mat maxImage = Mat::zeros(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1 );
    for(vector< vector<Mat> >::const_iterator it = houghSpace.begin(); it != houghSpace.end(); it++){
	for(vector<Mat>::const_iterator img = (*it).begin(); img != (*it).end(); img++){
	    max(*img, maxImage, maxImage);
	}
    }
    // get global maxima
    double min, max;
    minMaxLoc(maxImage, &min, &max);

    // define threshold
    double threshold = objThresh * max;

    // spatial non-maxima suppression
    Mat bin = Mat(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1, -1);
    for(int y=0; y<maxImage.rows; y++){
		for(int x=0; x<maxImage.cols; x++){
			// init
			bool localMax = true;
			// check neighbors
			for(int i=-1; i<=1; i++){
				int new_y = y + i;
				if ((new_y < 0) or (new_y >= maxImage.rows)){
					continue;
				}
				for(int j=-1; j<=1; j++){
					int new_x = x + j;
					if ((new_x < 0) or (new_x >= maxImage.cols)){
					continue;
					}
					if (maxImage.at<float>(new_y, new_x) > maxImage.at<float>(y, x)){
					localMax = false;
					break;
					}
				}
				if (!localMax)
					break;
			}
			// check if local max is larger than threshold
			if ( (localMax) and (maxImage.at<float>(y, x) > threshold) ){
				bin.at<float>(y, x) = maxImage.at<float>(y, x);
			}
		}
    }
    
    // loop through hough space after non-max suppression and add objects to object list
    double scale, angle;
    scale = 0;
    for(vector< vector<Mat> >::const_iterator it = houghSpace.begin(); it != houghSpace.end(); it++, scale++){
		angle = 0;
		for(vector<Mat>::const_iterator img = (*it).begin(); img != (*it).end(); img++, angle++){
			for(int y=0; y<bin.rows; y++){
				for(int x=0; x<bin.cols; x++){
					if ( (*img).at<float>(y, x) == bin.at<float>(y, x) ){
					// create object list entry consisting of scale, angle, and position where object was detected
					Scalar cur;
					cur.val[0] = scale;
					cur.val[1] = angle;
					cur.val[2] = x;
					cur.val[3] = y;	    
					objList.push_back(cur);
					}
				}
			}
		}
    }   
}

// shows the image
/*
img:	the image to be displayed
win:	the window name
dur:	wait number of ms or until key is pressed
*/
void Aia3::showImage(Mat const& img, string win, double dur){
  
    // use copy for normalization
    Mat tempDisplay;
    if (img.channels() == 1)
	normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    else
	tempDisplay = img.clone();
    
    tempDisplay.convertTo(tempDisplay, CV_8UC1);
    
    // create window and display omage
    namedWindow( win.c_str(), 0 );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) cvWaitKey(dur);
    // be tidy
    destroyWindow(win.c_str());
    
}

// Performes a circular shift in (dx,dy) direction
/*
in:		input matrix
out:	circular shifted matrix
dx:		shift in x-direction
dy:		shift in y-direction
*/
void Aia3::circShift(Mat const& in, Mat& out, int dx, int dy){

	Mat tmp = Mat::zeros(in.rows, in.cols, in.type());
 
	int x, y, new_x, new_y;
	
	for(y=0; y<in.rows; y++){

	      // calulate new y-coordinate
	      new_y = y + dy;
	      if (new_y<0)
		  new_y = new_y + in.rows;
	      if (new_y>=in.rows)
		  new_y = new_y - in.rows;
	      
	      for(x=0; x<in.cols; x++){

		  // calculate new x-coordinate
		  new_x = x + dx;
		  if (new_x<0)
			new_x = new_x + in.cols;
		  if (new_x>=in.cols)
			new_x = new_x - in.cols;
 
		  tmp.at<Vec2f>(new_y, new_x) = in.at<Vec2f>(y, x);
		  
	    }
	}
	out = tmp;
}
