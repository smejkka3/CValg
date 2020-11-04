//============================================================================
// Name        : Aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description :
//============================================================================

#include "Aia2.h"
#include <math.h>

// calculates the contour line of all objects in an image
/*
img			the input image
objList		vector of contours, each represented by a two-channel matrix
thresh		threshold used to binarize the image
k			number of applications of the erosion operator
*/
void Aia2::getContourLine(const Mat& img, vector<Mat>& objList, int thresh, int k){

   // namedWindow("Window", CV_WINDOW_AUTOSIZE);
   // namedWindow("after erosion", CV_WINDOW_AUTOSIZE);
   // namedWindow("after threshold", CV_WINDOW_AUTOSIZE);
    Mat output;

    //Threshold input:
    cv::threshold(img,  output, thresh, 255, THRESH_BINARY_INV);

    //imshow( "after threshold", output);
    //waitKey(0);
    //erode input:
    Mat erosionMatrix =  Mat();
    cv::erode(output, output, erosionMatrix,  Point(-1,-1),k);

    //imshow( "after erosion", output);
    //waitKey(0);

    //get contourlines of input and save them in objList:
    vector<Vec4i> hierarchy;
    findContours(output, objList, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE ,Point(0, 0));
    int idx = 0;

     for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( output, objList , idx, color, 1, 8, hierarchy );
    }
    //out << objList[0].dims <<endl;        //Get dimensions of objlist-elements
   // imshow( "Window", output);
    //waitKey(0);
}

// calculates the (unnormalized!) fourier descriptor from a list of points
/*
contour		1xN 2-channel matrix, containing N points (x in first, y in second channel)
out		fourier descriptor (not normalized)
*/
Mat Aia2::makeFD(const Mat& contour){

    Mat converted = Mat();

    if(contour.type() != CV_32FC2){
        contour.convertTo(converted, CV_32FC2); // OutputArray of Converted needs to be empty!
    }
    else{
        converted = contour.clone();
    }

   //Create matrix for the fourier descriptor
   Mat descriptor = Mat(contour.rows, contour.cols,  CV_32FC2);

   //create fourier descriptor:
    dft(converted, descriptor);
    cout << "Vector in frequency domain: "<< endl;
    cout <<  descriptor << endl;
    waitKey(0);
    
    return descriptor;
}

// normalize a given fourier descriptor
/*
fd		the given fourier descriptor
n		number of used frequencies (should be even)
out		the normalized fourier descriptor
*/
Mat Aia2::normFD(const Mat& fd, int n){
    int dur = 100;
    plotFD(fd, "fd not normalized.png", dur);

    //TRANSLATION INVARIANCE
    Mat fd_trans_inv = fd.clone();
    fd_trans_inv.row(0) = 0.0; //set FD(0) to zero!

    // cout << fd_trans_inv << endl;
    plotFD(fd_trans_inv, "fd translation invariant.png", dur);


    //SCALE INVARIANCE
    Mat fd_scale_inv = fd_trans_inv.clone();
    float norm;
    float fd1_1, fd1_2;
    int rows = fd_scale_inv.rows;

    //calculate norm of second entry (F(1)):
    fd1_1 = fd_scale_inv.at<float>(1, 0);
    fd1_2 = fd_scale_inv.at<float>(1, 1);
    norm = sqrt(pow(fd1_1,2)+pow(fd1_2,2));

    //if the norm of the second entry is equal to zero
    if(norm == 0){
        //calculate norm of the last Frequency (FD(-1))
        fd1_1 = fd_scale_inv.at<float>(rows-1, 0);
        fd1_2 = fd_scale_inv.at<float>(rows-1, 1);
        norm = sqrt(pow(fd1_1,2)+pow(fd1_2,2));
    }

   // cout<< "Scale invarianve object :"<<endl;
    // cout << fd_scale_inv << endl;
    // cout << "LAst entry_"<< fd_scale_inv.at<float>(fd_scale_inv.rows-1,0) << fd_scale_inv.at<float>(fd_scale_inv.rows-1,1) <<endl;

    //Scale the FD by dividing each entry through the calculated norm:
    for(int i=0; i<rows; i++){
        fd_scale_inv.at<float>(i,0 ) = fd_scale_inv.at<float>(i,0)/norm;
        fd_scale_inv.at<float>(i,1) = fd_scale_inv.at<float>(i,1)/norm;

    }
      //  fd_scale_inv.at<float>(rows-1,0 ) = fd_scale_inv.at<float>(rows-1,0)/norm;
      //  fd_scale_inv.at<float>(rows-1,1) = fd_scale_inv.at<float>(rows-1,1)/norm;
     //cout<< "Normed Scale invarianve object :"<<endl;
     //cout << fd_scale_inv << endl;


    //ROTATION INVARIANCE
    Mat planes[2];
    split(fd_scale_inv, planes);

    //Calculate the magnitude and phase of 2D vectors
    Mat magnitude, phase;
    cartToPolar(planes[0], planes[1], magnitude, phase);

    /*check if carttopolar works properly:
    int r = magnitude.rows;

    //cout << "check it: "<< planes[0].at<float>(r-1)<< "y= "<< planes[1].at<float>(r-1)<<"magnitude= "<< magnitude.at<float>(r-1)<<"phase= "<< phase.at<float>(r-1) <<endl;
    yes it works!!!! */

    Mat fd_rot_inv = magnitude;
    plotFD(fd_rot_inv, "fd translation, scale, and rotation invariant.png", dur);
    //cout << fd_rot_inv.type()<< "Type of fd_rot_inv"<<endl;
    //cout << fd_scale_inv<<endl;
    //cout << fd_scale_inv.row(fd_scale_inv.rows-1) << "last entry not rotation incariant"<< endl;
    //cout << "norm:" << norm<<endl;

    //NOISE INVARIANCE
    //Keep just the n lowest frequencies
    int half = static_cast<int>(n/2);

    Mat result = Mat::zeros(n,1, CV_32FC1);

    for(int i=0; i<= half; i++){
         result.at<float>(i) = fd_rot_inv.at<float>(i);
         result.at<float>(i+half) = fd_rot_inv.at<float>(fd_rot_inv.rows-half+i);
    }
    //cout << "Normalized fds:" << result<<endl;
    plotFD(result,"Fully Normalized DFT.png",dur);
  return result;
}

// plot fourier descriptor
/*
fd	the fourier descriptor to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/

void Aia2::plotFD(const Mat& fd, string win, double dur){

    //check for ROTATION INVARIANCE:
    Mat ifd;

    //if the input is a one channel matrix (i.e. it is rotation invariant)
    if(fd.type()==5){
        //Add a second channel with zeros:
        Mat g, fd_expanded;
        g = Mat::zeros(fd.rows, 1, CV_32FC1);
        //cout << g <<endl;
        // cout << fd  << endl;

        // Merge the two channels
        vector<Mat> channels;
        channels.push_back(fd);
        channels.push_back(g);
        merge(channels, fd_expanded);

        //apply inverse DFT:
        dft(fd_expanded, ifd, DFT_INVERSE | DFT_SCALE);
        //cout<< "gschafft"<<endl;
        //cout << ifd<< endl;
    }else{
        dft(fd, ifd, DFT_INVERSE | DFT_SCALE);
    }


    //CHECK FOR NEGATIVE COORDINATES:
    float min_x= HUGE_VAL;
    float min_y= HUGE_VAL;

    //find minimum coordinates:
     for(int i = 0; i<ifd.rows;i++)
    {
        min_x= min(ifd.at<float>(i,0),min_x);
        min_y = min(ifd.at<float>(i,1),min_y);
    }
    //cout << "Minimum-coordinates are (x,y) = (" << min_x << "," << min_y << ")." <<endl;

    //Add an OFFSET:
    if (min_x < 0){
    //cout << "Negative X-Coordinates! Adding Offset..."<<endl;
        for(int i = 0; i<ifd.rows;i++)
        {
        ifd.at<float>(i, 0) = ifd.at<float>(i, 0) - min_x;
        }
    }
    if (min_y < 0){
    //cout << "Negative Y-Coordinates! Adding Offset..."<<endl;
        for(int i = 0; i<ifd.rows;i++)
        {
        ifd.at<float>(i, 1) = ifd.at<float>(i, 1) - min_y;
        }
    }


    //CHECK IF FD IS SCALED:
    //float norm1 = sqrt(pow(fd.at<float>(1, 0),2)+pow(fd.at<float>(1, 1),2));
    //if( norm1 == 1 || (norm1 == 0 &&  sqrt(pow(fd.at<float>(2, 0),2)+pow(fd.at<float>(2, 1),2)) == 1)){


    //Scale the plot back:
     float fmax_x= -HUGE_VAL;
     float fmax_y= -HUGE_VAL;

    for(int i = 0; i<ifd.rows;i++) //find maximum coordinates
    {
        fmax_x= max(ifd.at<float>(i, 0),fmax_x);
        fmax_y = max(ifd.at<float>(i, 1),fmax_y);
    }

    //cout << "Maximum-scale  = (" << fmax_x << "," << fmax_y << ")." <<endl;
    waitKey(dur);

    //Rescale coordinates:
    float m = 100/fmax_x;

    for(int i = 0; i<ifd.rows;i++) //find maximum coordinates
    {
        ifd.at<float>(i, 0) = ifd.at<float>(i, 0)*m;
        ifd.at<float>(i, 1) =   ifd.at<float>(i, 1)*m;
    }
    ifd.convertTo(ifd, CV_32SC2);

    //find max x and y values
    int max_x= -HUGE_VAL;
    int max_y= -HUGE_VAL;

    for(int i = 0; i<ifd.rows;i++) //find maximum coordinates
    {
        max_x= max(ifd.at<Point>(i).x,max_x);
        max_y = max(ifd.at<Point>(i).y,max_y);
    }

    //cout << "Maximum-coordinates are (x,y) = (" << max_x << "," << max_y << ")." <<endl;

    //generate picture with 1 pixels more:
    max_x = max_x+1;
    max_y = max_y+1;

    Mat plot = Mat::zeros(max_y, max_x, CV_8UC3);

    //draw contour:
    Scalar color( 255, 255, 255 );
    drawContours( plot, ifd , -1, color, 1, 8 );

    namedWindow(win);
    imshow(win,plot);
    //imwrite(win, plot);
    waitKey(dur);

}


/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing functions, and saves result
// in particular extracts FDs and compares them to templates
/*
img			path to query image
template1	path to template image of class 1
template2	path to template image of class 2
*/


void Aia2::run(string img, string template1, string template2){

	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	Mat exC1 = imread( template1, 0);
	Mat exC2  = imread( template2, 0);
	if ( (!exC1.data) || (!exC2.data) ){
	    cout << "ERROR: Cannot load class examples in\n" << template1 << "\n" << template2 << endl;
	    cout << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// parameters
	// these two will be adjusted below for each image indiviudally
	int binThreshold;				      // threshold for image binarization
	int numOfErosions;			   	// number of applications of the erosion operator
	// these two values work fine, but it might be interesting for you to play around with them
	int steps = 32;					   // number of dimensions of the FD
	double detThreshold = 0.01;		// threshold for detection

	// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// TO DO !!!
	// --> Adjust threshold and number of erosion operations

	//class examples:
	binThreshold = 170;
	numOfErosions = 8;


	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	int mSize = 0, mc1 = 0, mc2 = 0, i = 0;
	for(vector<Mat>::iterator c = contourLines1.begin(); c != contourLines1.end(); c++,i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc1 = i;
		}
	}
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);
	for(vector<Mat>::iterator c = contourLines2.begin(); c != contourLines2.end(); c++, i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc2 = i;
		}
	}
	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.at(mc1));
	Mat fd2 = makeFD(contourLines2.at(mc2));

	// normalize  fourier descriptor
	Mat fd1_norm = normFD(fd1, steps);
	Mat fd2_norm = normFD(fd2, steps);

	// process query image
	// load image as gray-scale, path in argv[1]
	Mat query = imread( img, 0);
	if (!query.data){
	    cout << "ERROR: Cannot load query image in\n" << img << endl;
	    cout << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// get contour lines from image
	vector<Mat> contourLines;
	// TO DO !!!
	// --> Adjust threshold and number of erosion operations
	binThreshold = 140;
	numOfErosions = 4;
	getContourLine(query, contourLines, binThreshold, numOfErosions);

	cout << "Found " << contourLines.size() << " object candidates" << endl;

	// just to visualize classification result
	Mat result(query.rows, query.cols, CV_8UC3);
	vector<Mat> tmp;
	tmp.push_back(query);
	tmp.push_back(query);
	tmp.push_back(query);
	merge(tmp, result);

	// loop through all contours found
	i = 1;
	for(vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++){

	    cout << "Checking object candidate no " << i << " :\t";

		// color current object in yellow
	  	Vec3b col(0,255,255);
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    showImage(result, "result", 0);

	    // if fourier descriptor has too few components (too small contour), then skip it (and color it in blue)
	    if (c->rows < steps){
			cout << "Too less boundary points (" << c->rows << " instead of " << steps << ")" << endl;
			col = Vec3b(255,0,0);
	    }else{
			// calculate fourier descriptor
			Mat fd = makeFD(*c);
			// normalize fourier descriptor
			Mat fd_norm = normFD(fd, steps);
			// compare fourier descriptors
			double err1 = norm(fd_norm, fd1_norm)/steps;
			double err2 = norm(fd_norm, fd2_norm)/steps;
			// if similarity is too small, then reject (and color in cyan)
			if (min(err1, err2) > detThreshold){
				cout << "No class instance ( " << min(err1, err2) << " )" << endl;
				col = Vec3b(255,255,0);
			}else{
				// otherwise: assign color according to class
				if (err1 > err2){
					col = Vec3b(0,0,255);
					cout << "Class 2 ( " << err2 << " )" << endl;
				}else{
					col = Vec3b(0,255,0);
					cout << "Class 1 ( " << err1 << " )" << endl;
				}
			}
		}
		// draw detection result
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    // for intermediate results, use the following line
	    showImage(result, "result", 0);

	}
	// save result
	imwrite("result.png", result);
	// show final result
	showImage(result, "result", 0);
	//imwrite("result.png",result);
}

// shows the image
/*
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::showImage(const Mat& img, string win, double dur){

    // use copy for normalization
    Mat tempDisplay = img.clone();
    if (img.channels() == 1) normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    // create window and display omage
    namedWindow( win.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) waitKey(dur);

}

// function loads input image and calls processing function
// output is tested on "correctness"
void Aia2::test(void){

	test_getContourLine();
	test_makeFD();
	test_normFD();

}

void Aia2::test_getContourLine(void){

   // creates a black square on a white background
	Mat img(100, 100, CV_8UC1, Scalar(255));
	Mat roi(img, Rect(40,40,20,20));
	roi.setTo(0);
   // test correctness for #erosions=1
   // creates correct outline
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
   // computes outline
	vector<Mat> objList;
	getContourLine(img, objList, 128, 1);
   // compares correct and computed outlines
   // there should be only one object
	if ( objList.size() > 1 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> found too many contours (#erosions=1)" << endl;
		cin.get();
	}
	if ( max(objList.at(0).rows, objList.at(0).cols) != 68 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> contour has wrong number of pixels (#erosions=1)" << endl;
		cin.get();
	}
	if ( sum(cline != objList.at(0)).val[0] != 0 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> computed wrong contour (#erosions=1)" << endl;
		cin.get();
	}
   // test correctness for #erosions=3
   // re-init
   objList.resize(0);
   // creates correct outline
   cline = Mat(52,1,CV_32SC2);
   k=0;
	for(int i=43; i<56; i++) cline.at<Vec2i>(k++) = Vec2i(43,i);
	for(int i=43; i<56; i++) cline.at<Vec2i>(k++) = Vec2i(i,56);
	for(int i=56; i>43; i--) cline.at<Vec2i>(k++) = Vec2i(56, i);
	for(int i=56; i>43; i--) cline.at<Vec2i>(k++) = Vec2i(i,43);
   // computes outline
	getContourLine(img, objList, 128, 3);
   // compares correct and computed outlines
	if ( objList.size() > 1 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> found too many contours (#erosions=3)" << endl;
		cin.get();
	}
	if ( max(objList.at(0).rows, objList.at(0).cols) != 52 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> contour has wrong number of pixels (#erosions=3)" << endl;
		cin.get();
	}
	if ( sum(cline != objList.at(0)).val[0] != 0 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> computed wrong contour (#erosions=3)" << endl;
		cin.get();
	}
}

void Aia2::test_makeFD(void){

   // create example outline
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);

   // test for correctness
	Mat fd = makeFD(cline);
	if (fd.rows != cline.rows){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "--> The number of frequencies does not match the number of contour points!" << endl;
		cin.get();
	}
	if (fd.channels() != 2){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "--> The fourier descriptor is supposed to be a two-channel, 1D matrix!" << endl;
		cin.get();
	}
   if (fd.type() != CV_32FC2){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "--> Frequency amplitudes are not computed with floating point precision!" << endl;
		cin.get();
	}
}

void Aia2::test_normFD(void){

	double eps = pow(10,-3);

   // create example outline
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);

	Mat fd = makeFD(cline);
	Mat nfd = normFD(fd, 32);
   // test for correctness
	if (nfd.channels() != 1){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The normalized fourier descriptor is supposed to be a one-channel, 1D matrix" << endl;
		cin.get();
	}
   if (nfd.type() != CV_32FC1){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The normalized fourier descriptor is supposed to be in floating point precision" << endl;
		cin.get();
	}
	if (abs(nfd.at<float>(0)) > eps){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The F(0)-component of the normalized fourier descriptor F is supposed to be 0" << endl;
		cin.get();
	}
	if ((abs(nfd.at<float>(1)-1.) > eps) && (abs(nfd.at<float>(nfd.rows-1)-1.) > eps)){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The F(1)-component of the normalized fourier descriptor F is supposed to be 1" << endl;
		cout << "--> But what if the unnormalized F(1)=0?" << endl;
		cin.get();
	}
	if (nfd.rows != 32){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The number of components does not match the specified number of components" << endl;
		cin.get();
	}
}
