//============================================================================
// Name        : Aia5.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia4.h"
//#include <limits.h>

// Computes the log-likelihood of each feature vector in every component of the supplied mixture model.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return: 	the log-likelihood log(p(x_i|y_i=j))
*/


Mat Aia5::calcCompLogL(const vector<struct comp*>& model, const Mat& features){
	// To Do !!!
	
	//Get familiar with the inputs:
	/*
	 for(int i = 0; i < model.size(); ++i){ 
		cout << "alpha: " << model.at(i)->weight<<endl;
		cout << "mean: " << model.at(i)->mean<<endl;
		cout << "cov: " << model.at(i)->covar<<endl;
	}	
		
	cout << "Feature-Matrix: "<< features.rows << "x" << features.cols << ", type: " << features.type()<<endl;	
	for(int r = 0; r < features.rows; ++r){
		for(int c = features.cols-15; c < features.cols; c++){
			cout << features.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	*/


	//get number of dimensions:
	int d = features.rows;
	//get number of samples:
	int n = features.cols;
	//get number of components:
	int K = model.size();
	//cout << "size: "<< C;
	
	//create log-likelihood-data-frame of appropriate type:
	Mat logLikelihood = Mat::zeros(K, n, CV_32FC1); 	//TODO: this could be changed to double precision
	
	//start computing the likelihoods:
	
	//not necessary for discriminating the classes, as this value is the same for each class
	//float z = -(d/2)*log(2*CV_PI);
	float z = 1;
	//cout << "z: "<< z<< endl;
	logLikelihood = logLikelihood*z;
	/*
	cout << "loglikelihood matrix:"<< endl;
	for(int r = 0; r < logLikelihood.rows; ++r){
		for(int c = 0; c < 15; c++){
			cout << logLikelihood.at<float>(r,c)<<" ";
			}
			cout << endl;
	}
	*/
	
	
	//for each component:
	for(int k = 0; k < K; ++k){
		//initialise the parameters of the given component:
		Mat mean =  model.at(k)->mean;
		Mat cov = model.at(k)->covar;
		
		//do some precomputations:
		double determinante = determinant(cov); 
		double log_det = -0.5*log(determinante);
		
		Mat inv_cov = Mat(cov.rows, cov.cols, cov.type());
		double check = invert(cov, inv_cov);
		
		if(check == 0){
			cout << "++++++++++ATTENTION! YOU TRIED TO INVERT A SINGULAR MATRIX++++++++++"<< endl;
		}
		
		//print the resuls:
		/*
		cout <<"Mean " << k << ":"<<endl;
		cout << mean<< endl;
		cout <<"Covariance-Matrix " << k<< ":"<<endl;
		cout << cov<< endl;
		cout <<"Inverted Covariance-Matrix " << k<< ":"<<endl;
		cout << inv_cov<< endl;
		cout <<"Determinant of Covariance-Matrix " << k<< ":"<<endl;
		cout << determinante<< endl;
		*/

		//now, fill in the matrix:

		
		
		//mean = Mat::ones(2,1,CV_32FC1);
		//inv_cov = Mat::eye(2,2,CV_32FC1);
		
		
		//Firstly, add -0.5log(det(sigma))
		
		
		//cout << "vector and mahalanobis:" <<endl;
		for(int c = 0; c < n; c++){
			//get x_i
			Mat vec = features.col(c);
	
			
			//compute -0.5(x-mu).TSigma_inv(x-mu) using the mahalanobis distance:
			double mahal = Mahalanobis(vec, mean, inv_cov);
			mahal = -0.5*mahal*mahal;				//square it and multiply it by -0.5
			
			//now set the entry:
			logLikelihood.at<float>(k,c) = log_det + mahal;
			
			}
		/*
		//print result:
		cout << "loglikelihood matrix:"<< endl;
	for(int r = 0; r < logLikelihood.rows; ++r){
		for(int c = 0; c < 15; c++){
			cout << logLikelihood.at<float>(r,c)<<" ";
			}: "<< features.rows << "x" << features.cols << ", type: " << features.type()<<endl;	
	for(int r =
			cout << endl;
		}
*/
	}
	//print the loglikelihood-matrix:
	/*
	for(int r = 0; r < logLikelihood.rows; ++r){
		for(int c = 0; c < 15; c++){
			cout << logLikelihood.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	
*/
	
	
		
	return logLikelihood;
}

// Computes the log-likelihood of each feature by combining the likelihoods in all model components.
/*
model:     structure containing model parameters
features:  matrix of feature vectors
return:	   the log-likelihood of feature number i in the mixture model (the log of the summation of alpha_j p(x_i|y_i=j) over j)
*/
Mat Aia5::calcMixtureLogL(const vector<struct comp*>& model, const Mat& features){
	//This Method WORKS FINE!!!!

	//get started:
	Mat logL = calcCompLogL(model, features);
	Mat likelihood;
	Mat result = Mat::zeros(1, features.cols, CV_32FC1);		//1xn - result-matrix
	Mat alphas(model.size(), 1, CV_32FC1);
	Mat one = Mat::ones(1,features.cols,CV_32FC1);
	
	//get alphas:
	for(int i = 0; i < model.size(); ++i){ 
		alphas.at<float>(0,i) = model.at(i)->weight;
	}
		
	/*
		cout << "loglikelihood- matrix:"<< endl;
	for(int r = 0; r < logL.rows; ++r){
		for(int c = logL.cols-15; c < logL.cols; c++){
			cout << logL.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	*/
	
	
	//1. compute ln(alpha_j)+logLikelihood:
	Mat ln_alpha, addition;
	log(alphas,ln_alpha);
	addition = ln_alpha * one;
	
	/*
	cout<<"ln_alphas:"<<endl;
	cout << ln_alpha<< endl;
	cout << "ln_alpha- matrix:"<< endl;
	for(int r = 0; r < addition.rows; ++r){
		for(int c = addition.cols-15; c < addition.cols; c++){
			cout << addition.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	*/
	
	add(logL, addition, logL); 
	/*
	cout << "ln_Lik + ln_alpha- matrix:"<< endl;
	for(int r = 0; r < logL.rows; ++r){
		for(int c = logL.cols-15; c < logL.cols; c++){
			cout << logL.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	*/
		
		
	//2. find the maximum value:
	double max;
	minMaxLoc(logL, NULL, &max , NULL, NULL);
	//cout <<"maximum-vlaue:"<< max<< endl;
	
	
	//3. subtract this value from the logL:
	subtract(logL,max,logL);	
	
	/*
	cout << "shifted ln_Lik + ln_alpha- matrix:"<< endl;
	for(int r = 0; r < logL.rows; ++r){
		for(int c = logL.cols-15; c < logL.cols; c++){
			cout << logL.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	*/
	
	
	//4. calculate exponentials:
	exp(logL,logL);
	/*
	cout << "exponential:"<< endl;
	for(int r = 0; r < logL.rows; ++r){
		for(int c = logL.cols-15; c < logL.cols; c++){
			cout << logL.at<float>(r,c)<<" ";
			}
			cout << endl;
		}
	*/
		
		
	//5. do the summation:
	for(int c=0; c<logL.cols; c++){
		float sum = 0;
		//sum the likelihoods over the components:
		for(int k = 0; k<  logL.rows; k++){
			sum = sum + logL.at<float>(k,c);
			}
		result.at<float>(0,c) = result.at<float>(0,c) + sum;
	}
	/*
	cout << "normalization- constants:"<< endl;
	for(int c = result.cols - 15; c < result.cols; ++c){
		cout << result.at<float>(0,c)<<" ";
		}
	cout << endl;
	*/
	
	
	//6. calculate log of the sum, and reverse logSumExpTrick:
	log(result,result);
	add(result, max, result);
	/*
	cout << "result:"<< endl;
	for(int c = result.cols - 15; c < result.cols; ++c){
		cout << result.at<float>(0,c)<<" ";
		}
	cout << endl;
	*/
	
	return result;

}

// Computes the posterior over components (the degree of component membership) for each feature.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return:		the posterior p(y_i=j|x_i)
*/
Mat Aia5::gmmEStep(const vector<struct comp*>& model, const Mat& features){
	// To Do !!!
	
	    //we call the first function
	Mat LogL = calcCompLogL(model, features);
	//we call the second function
    Mat likelihood = calcMixtureLogL(model, features);
    //we declare the output
	Mat gmm = Mat::zeros(model.size(), features.cols, features.type());
    for(unsigned i=0; i<model.size(); i++){ //i=row = componetn

        for(unsigned j=0; j<features.cols; j++){	//j = column = observation

        //we calculate the output as in the equation
           gmm.at<float>(i,j) = LogL.at<float>(i,j) + log((*model[i]).weight) - likelihood.at<float>(j);
        }	
    }
	exp(gmm,gmm);
	
	
	/*
	//print result:
	cout << "posterior- matrix:"<< endl;
	for(int r = 0; r< gmm.rows;r++){
		for(int c = gmm.cols-15; c < gmm.cols; c++){
			cout << gmm.at<float>(r,c)<<" ";
		}
		cout << endl;		
	}
	
	cout << "Sum of posteriors:"<< endl;
	for(int c = gmm.cols-100; c < gmm.cols; c++){
		float sum = 0;
	for(int r = 0; r< gmm.rows;r++){
		
			sum = sum+ gmm.at<float>(r,c);
		}
		cout << sum << " ";
		
	}
	cout << endl;
	
	  
	 THIS FUNCTION ALSO SEEMS TO BE FINE!!
	*/
	return gmm;
}

// Updates a given model on the basis of posteriors previously computed in the E-Step.
/*
model:     structure containing model parameters, will be updated in-place
           new model structure in which all parameters have been updated to reflect the current posterior distributions.
features:  matrix of feature vectors
posterior: the posterior p(y_i=j|x_i)
*/
void Aia5::gmmMStep(vector<struct comp*>& model, const Mat& features, const Mat& posterior){
   
	
	//1.Update N_j
	Mat N = Mat(posterior.rows, 1, CV_32FC1);
	
	float sum;
	for(int k= 0; k < posterior.rows; k++){
		sum = 0;
		for(int c = 0; c <posterior.cols;c++){
			sum = sum + posterior.at<float>(k,c);
		}
		N.at<float>(k,0) = sum;
	}
	//cout << "summe der NJ:"<< cv::sum(N)[0]<<endl;
	
	
	//2. Update alpha_j
		for(int j = 0; j < model.size(); ++j){ 
			//cout << "N an stelle "<<j<< " "<< N.at<float>(j) << endl;
			model.at(j)->weight = N.at<float>(j)/features.cols;
			//cout << "alpha_new an stelle "<<j<< " "<< model.at(j)->weight<< endl;
		}
		
		
	//3. Update mu_j
	//for each component:
	for(int j = 0; j< model.size();j++){
		//calculate corresponding mu:
		Mat mu = Mat(features.rows, 1, CV_32FC1);
		
		float sum;
		//for each dimension in feature-space:
		for(int dim = 0; dim < features.rows; dim++){
			sum = 0;
			//calculate the sum of posterior*x_i over the observations:
			for(int obs = 0; obs < features.cols; obs++){//
				sum = sum + features.at<float>(dim,obs)*posterior.at<float>(j,obs);
			}
		mu.at<float>(dim,0) = sum/N.at<float>(j);
		}
		//cout <<"neues mu an stelle "<<j<<": "<<mu<<endl;
		model.at(j)->mean = mu;
		}
	
	
	
	//4. Update Sigma_j
	for(int j = 0; j< model.size();j++){
		Mat centered = Mat(features.rows, features.cols, features.type());
		Mat mu = model.at(j)->mean;
		Mat cov = Mat::zeros(features.rows, features.rows, features.type());
		//calculate x - mu:
		for (int c = 0; c < features.cols; c++) {
			centered.col(c) = features.col(c) - mu;
		}
		
		Mat dst;
		for(int c = 0; c < centered.cols; c++){
			
			mulTransposed(centered.col(c), dst, 0, noArray(), 1 , -1);
			cov = cov + dst*posterior.at<float>(j,c);
		}
		//cout <<"neue cov an stelle "<<j<<": "<<cov<<endl;
		//	cout <<cov<<endl;
		model.at(j)->covar = cov/N.at<float>(j);
		
	}
	
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// sets parameters and generates data for EM clustering
void Aia5::run(void){

	// *********************************//============================================================================
// Name        : Aia4.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for fourth AIA assignment
//============================================================================



	//this was used in order to test the gmm for diffrent pc /cluster - combinations!
	//string s = "PCs Components Accuracy\n";
	//for(int pc = 2; pc <= 10; pc=pc +2){
		//for(int comp = 12; comp <= 16; comp = comp +2){
		
		
	
		
		
		
		// To Do: Adjust number of princ. comp. and number of clusters and evaluate performance
		// dimensionality of the generated feature vectors (number of principal components)	
		//*****
    int vectLen = 24;	
	//int vectLen = pc;
	
    // number of components (clusters) to be used by the model
    int numberOfComponents = 10; 	
	//int numberOfComponents = comp; 
    
    // *********************************

     // read training data
    cout << "Reading training data..." << endl;
    vector<Mat> trainImgDB;
    readImageDatabase("./img/train/in/", trainImgDB);
    cout << "Done\n" << endl;
      
    // generate PCA-basis
    cout << "Generate PCA-basis from data:" << endl;
    vector<PCA> featProjection;
    genFeatureProjection(trainImgDB, featProjection, vectLen);
    cout << "Done\n" << endl;

    // this is gonna contain the individual models (one per category)
    vector< vector<struct comp*> > models;
    
    // start learning
    cout << "Start learning..." << endl;
    for(int c=0; c<10; c++){
	cout << "\nTrain GMM of category " << c << endl;
	cout << " > Project on principal components of category " << c << " :" << endl;
	Mat fea = Mat(trainImgDB.at(c).rows, vectLen, CV_32FC1);
	featProjection.at(c).project( trainImgDB.at(c), fea );
	fea = fea.t();
	cout << "> Done" << endl;
	
	cout << " > Estimate probability density of category " << c << " :" << endl;
	// train the corresponding mixture model using EM...
	vector<struct comp*> model;
	trainGMM(fea, numberOfComponents, model);
	models.push_back(model);
	cout << "> Done" << endl;
    }
    cout << "Done\n" << endl;

    // read testing data
    cout << "Reading test data...:\t";
    vector<Mat> testImgDB;
    readImageDatabase("./img/test/in/" , testImgDB);
    cout << "Done\n" << endl;
    
    cout << "Test GMM: Start" << endl;
    Mat confMatrix = Mat::zeros(10, 10, CV_32FC1);
    int dtr = 0, n=0;
    // for each category within the test data
    for(int c_true=0; c_true<10; c_true++){
	n += testImgDB.at(c_true).rows;
	// init likelihood
	Mat maxMixLogL = Mat(1, testImgDB.at(c_true).rows, CV_32FC1);
	// estimated class
	Mat est = Mat::zeros(1, testImgDB.at(c_true).rows, CV_8UC1);
	
	for(int c_est=0; c_est<10; c_est++){
	    cout << " > Project on principal components of category " << c_est << " :\t";
	    Mat fea = Mat(testImgDB.at(c_true).rows, vectLen, CV_32FC1);
	    featProjection.at(c_est).project( testImgDB.at(c_true), fea );
	    fea = fea.t();
	    cout << "Done" << endl;
	
	    cout << " > Estimate class likelihood of category " << c_est << " :" << endl;
	    
	    // get data log
	    Mat mixLogL = calcMixtureLogL(models.at(c_est), fea);

	    // compare to current max
	    for(int i=0; i<fea.cols; i++){
		if ( ( maxMixLogL.at<float>(0,i) < mixLogL.at<float>(0,i) ) or (c_est == 0) ){
		    maxMixLogL.at<float>(0,i) = mixLogL.at<float>(0,i);
		    est.at<uchar>(0,i) = c_est;
		}
	    }
	    cout << "Done\n" << endl;
	}
	// make corresponding entry in confusion matrix
	for(int i=0; i<testImgDB.at(c_true).rows; i++){
	    //cout << (int)est.at<uchar>(0,i) << "\t" << c_true << endl;
	    confMatrix.at<float>( c_true, (int)est.at<uchar>(0,i) )++;
	    dtr += (int)est.at<uchar>(0,i) == c_true;
	}
    }
    cout << "Test GMM: Done\n" << endl;
    
    cout << endl << "Confusion matrix:" << endl;
    cout << confMatrix << endl;
    cout << endl << "No of correctly classified:\t" << dtr << " of " << n << "\t( " << (dtr/(double)n*100) << "% )" << endl;
	
	
	
	
	//some code for investigation of the diffrent pc-comp-combinations:
	//s = s + to_string(pc) + " " + to_string(comp) + " "+ to_string(dtr)+ " \n";
		
	//e}
	// }
	
	/*Write the current accuracy to disk:
	string path = "results.txt";
	
	FILE* pFile = fopen(path.c_str(), "w");
	if (pFile != NULL)
	{
		fputs(s.c_str(), pFile);
		fclose(pFile);
		cout << "SUCCESS!"<<endl;
	}
	else{
	cout << "there was a problem opening the file!"<<endl;
		}
		 */

}

// sets parameters and generates data for EM clustering
void Aia5::test(void){

	// dimensionality of the generated feature vectors
    int vectLen = 2;

    // number of components (clusters) to generate
    int actualComponentNum = 10;

    // maximal standard deviation of vectors in each cluster
    double actualDevMax = 0.3;

    // number of vectors in each cluster
    int trainingSize = 150;

    // initialise random component parameters (mean and standard deviation)
    Mat actualMean(actualComponentNum, vectLen, CV_32FC1);
    Mat actualSDev(actualComponentNum, vectLen, CV_32FC1); 
    randu(actualMean, 0, 1);
    randu(actualSDev, 0, actualDevMax);

    // print distribution parameters to screen
    cout << "true mean" << endl;
    cout << actualMean << endl;
    cout << "true sdev" << endl;
    cout << actualSDev << endl;

    // initialise random cluster vectors
    Mat trainingData = Mat::zeros(vectLen, trainingSize*actualComponentNum, CV_32FC1);
    int n=0;
    RNG rng;
    for(int c=0; c<actualComponentNum; c++){
		for(int s=0; s<trainingSize; s++){
			for(int d=0; d<vectLen; d++){
				trainingData.at<float>(d,n) = rng.gaussian( actualSDev.at<float>(c, d) ) + actualMean.at<float>(c, d);
			}
			n++;
		}
    }

    // train the corresponding mixture model using EM...
 	 vector<struct comp*> model;
    trainGMM(trainingData, actualComponentNum, model);
    
}

// Trains a Gaussian mixture model with a specified number of components on the basis of a given set of feature vectors.
/*
data:     		feature vectors, one vector per column
numberOfComponents:	the desired number of components in the trained model
model:			the model, that will be created and trained inside of this function
*/
void Aia5::trainGMM(const Mat& data, int numberOfComponents, vector<struct comp*>& model){

    // the number and dimensionality of feature vectors
    int featureNum = data.cols;
    int featureDim = data.rows;

    // initialize the model with one component and arbitrary parameters
    struct comp* fst = new struct comp();
    fst->weight = 1;
    fst->mean = Mat::zeros(featureDim, 1, CV_32FC1);
    fst->covar = Mat::eye(featureDim, featureDim, CV_32FC1);
    model.push_back(fst);

    // the data-log-likelihood
    double dataLogL[2] = {0,0};
        
    // iteratively add components to the mixture model
    for(int i=1; i<=numberOfComponents; i++){
      
		cout << "Current number of components: " << i << endl;
		
		// the current combined data log-likelihood p(X|Omega)
		Mat mixLogL = calcMixtureLogL(model, data);
		dataLogL[0] = sum(mixLogL).val[0];
		dataLogL[1] = 0.;
		
		// EM iteration while p(X|Omega) increases
		int it = 0;
		while( (dataLogL[0] > dataLogL[1]) or (it == 0) ){
		  
			printf("Iteration: %d\t:\t%f\r", it++, dataLogL[0]);

			// E-Step (computes posterior)
			Mat posterior = gmmEStep(model, data);

			// M-Step (updates model parameters)
			gmmMStep(model, data, posterior);
			
			// update the current p(X|Omega)
			dataLogL[1] = dataLogL[0];
			mixLogL = calcMixtureLogL(model, data);
			dataLogL[0] = sum(mixLogL).val[0];
		}
		cout << endl;

		// visualize the current model (with i components trained)
		if (featureDim >= 2){
		   plotGMM(model, data);
		}

		// add a new component if necessary
		if (i < numberOfComponents){
			initNewComponent(model, data);    
		}
    }
    
    cout << endl << "**********************************" << endl;
    cout << "Trained model: " << endl;
    for(int i=0; i<model.size(); i++){
		cout << "Component " << i << endl;
		cout << "\t>> weight: " << model.at(i)->weight << endl;
		cout << "\t>> mean: " << model.at(i)->mean << endl;
		cout << "\t>> std: [" << sqrt(model.at(i)->covar.at<float>(0,0)) << ", " << sqrt(model.at(i)->covar.at<float>(1,1)) << "]" << endl;
		cout << "\t>> covar: " << endl;
		cout << model.at(i)->covar << endl;
		cout << endl;
    }

}

// Adds a new component to the input mixture model by spliting one of the existing components in two parts.
/*
model:		Gaussian Mixture Model parameters, will be updated in-place
features:	feature vectors
*/
void Aia5::initNewComponent(vector<struct comp*>& model, const Mat& features){

    // number of components in current model
    int compNum = model.size();

    // number of features
    int featureNum = features.cols;

    // dimensionality of feature vectors (equals 3 in this exercise)
    int featureDim = features.rows;

    // the largest component is split (this is not a good criterion...)
    int splitComp = 0;
    for(int i=0; i<compNum; i++){
		if (model.at(splitComp)->weight < model.at(i)->weight){
			splitComp = i;
		}
    }

    // split component 'splitComp' along its major axis
    Mat eVec, eVal;
    eigen(model.at(splitComp)->covar, eVal, eVec);

    Mat devVec = 0.5 * sqrt( eVal.at<float>(0) ) * eVec.row(0).t();

    // create new model structure and compute new mean values, covariances, new component weights...
    struct comp* newModel = new struct comp;
    newModel->weight = 0.5 * model.at(splitComp)->weight;
    newModel->mean = model.at(splitComp)->mean - devVec;
    newModel->covar = 0.25 * model.at(splitComp)->covar;

    // modify the split component
    model.at(splitComp)->weight = 0.5*model.at(splitComp)->weight;
    model.at(splitComp)->mean += devVec;
    model.at(splitComp)->covar *= 0.25;

    // add it to old model
    model.push_back(newModel);

}

// Visualises the contents of a feature space and the associated mixture model.
/*
model: 		parameters of a Gaussian mixture model
features: 	feature vectors

Feature vectors are plotted as black points
Estimated means of components are indicated by blue circles
Estimated covariances are indicated by blue ellipses
If the feature space has more than 2 dimensions, only the first two dimensions are visualized.
*/
void Aia5::plotGMM(const vector<struct comp*>& model, const Mat& features){

    // size of the plot
    int imSize = 500;
  
    // get scaling factor to scale coordinates
    double max_x=0, max_y=0, min_x=0, min_y=0;
    for(int n=0; n<features.cols; n++){
		if (max_x < features.at<float>(0, n) )
			max_x = features.at<float>(0, n);
		if (min_x > features.at<float>(0, n) )
			min_x = features.at<float>(0, n);
		if (max_y < features.at<float>(1, n) )
			max_y = features.at<float>(1, n);
		if (min_y > features.at<float>(1, n) )
			min_y = features.at<float>(1, n);
    }  
    double scale = (imSize-1)/max((max_x - min_x), (max_y - min_y));
    // create plot
    Mat plot = Mat(imSize, imSize, CV_8UC3, Scalar(255,255,255) );

    // set feature points
    for(int n=0; n<features.cols; n++){
		plot.at<Vec3b>( ( features.at<float>(0, n) - min_x ) * scale, max( (features.at<float>(1,n)-min_y)*scale, 5.) ) = Vec3b(0,0,0);
    }
    // get ellipse of components
    Mat EVec, EVal;
    for(int i=0; i<model.size(); i++){

		eigen(model.at(i)->covar, EVal, EVec);
		double rAng = atan2(EVec.at<float>(0, 1), EVec.at<float>(0, 0));

		// draw components
		circle(plot,  Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), 3, Scalar(255,0,0), 2);
		ellipse(plot, Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), Size(sqrt(EVal.at<float>(1))*scale*2, sqrt(EVal.at<float>(0))*scale*2), rAng*180/CV_PI, 0, 360, Scalar(255,0,0), 2);

    }
    
    // show plot an abd wait for key
    imshow("Current model", plot);
    waitKey(0);

}

// constructs PCA-space based on all samples of each class
/*
imgDB		image data base; each matrix corresponds to one class; each row to one image
featSpace	the PCA-bases for each class
vectLen		number of principal components to be used
*/
void Aia5::genFeatureProjection(const vector<Mat>& imgDB, vector<PCA>& featSpace, int vectLen){

    int c = 0;
    for(vector<Mat>::const_iterator cat = imgDB.begin(); cat != imgDB.end(); cat++, c++){
	cout << " > Generate PC of category " << c << " :\t";
      	PCA compPCA = PCA(*cat, Mat(), CV_PCA_DATA_AS_ROW, vectLen);
		//cout << "Eigenvalues:"<<endl;
		//cout << compPCA.eigenvalues<< endl; //to delete
		//waitKey(0);
	featSpace.push_back(compPCA);
	cout << "Done" << endl;
    }
}

// reads image data base from disc
/*
dataPath	path to directory
db		each matrix in this vector corresponds to one class, each row of the matrix corresponds to one image 
*/
void Aia5::readImageDatabase(string dataPath, vector<Mat>& db){

    // directory delimiter. you might wanna use '\' on windows systems
    char delim = '/';

    char curDir[100];
    db.reserve(10);
    
    int numberOfImages = 0;
    for(int c=0; c<10; c++){

	list<Mat> imgList;
	sprintf(curDir, "%s%c%i%c", dataPath.c_str(), delim, c, delim);
 
	// read directory
	DIR* pDIR;
	struct dirent *entry;
	struct stat s;
	
	stat(curDir,&s);

	// if path is a directory
	if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
	    if( pDIR=opendir(curDir) ){
		// for all entries in directory
		while(entry = readdir(pDIR)){
		    // is current entry a file?
		    stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
		    if ( ( (s.st_mode & S_IFMT ) != S_IFDIR ) and ( (s.st_mode & S_IFMT ) == S_IFREG ) ){
			// if all conditions are fulfilled: load data
			Mat img = imread((curDir + string(entry->d_name)).c_str(), 0);
			img.convertTo(img, CV_32FC3);
			img /= 255.;
			imgList.push_back(img);
			numberOfImages++;
		    }
		}
		closedir(pDIR);
	    }else{
		cerr << "\nERROR: cant open data dir " << dataPath << endl;
		exit(-1);
	    }
	}else{
	    cerr << "\nERROR: provided path does not specify a directory: ( " << dataPath << " )" << endl;
	    exit(-1);
	}
	
	int numberOfImages = imgList.size();
	int numberOfPixPerImg = imgList.front().cols * imgList.front().rows;
	    
	Mat feature = Mat(numberOfImages, numberOfPixPerImg, CV_32FC1);
	
	int i = 0;
	for(list<Mat>::iterator img = imgList.begin(); img != imgList.end(); img++, i++){
	    for(int p = 0; p<numberOfPixPerImg; p++){
		feature.at<float>(i, p) = img->at<float>(p);
	    }
	}
	db.push_back(feature);
    }  
}
