#ifndef dlgp_h
#define dlgp_h

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

class dlgp
{
private:
	Eigen::MatrixXd X, Y, K, alpha, overlaps;
	//X: inputs Y: outputs K: covariance matrices alpha: prediction vectors 
	//overlaps: row 1 is cut dimension, row 2 is size of overlapping region

	Eigen::MatrixXi children; //children: row1 is left child, row2 is right child

	Eigen::RowVectorXi localCount, auxUbic;	//localCount: vector with amount of pts in each GP auxUbic: maps model to data

	Eigen::RowVectorXd medians;	//medians: vector of hyperplanes

	int amountL;	//if (ard) is xSize else is 1
	int count;		//amount of local GPs

public:
	int pts = 50;	//amount of pts in each local GP
	int N = 10;		//max. number of local GPs (leaves)
	int xSize = 6;	//dimensionality of input
	int divMethod = 3;	//division method 1: symmetric 2: mean 3: median
	int wo = 300;	//overlapping factor
	int outs = 2;	//amount of outputs
	
	Eigen::ArrayXd sigmaF, sigmaN; //hyperparameters
	Eigen::ArrayXd lengthS;			//hyperparameters

	dlgp(int xSize0, int outs0, int pts0, int N0, bool ard0);

	Eigen::RowVectorXd kernel(Eigen::MatrixXd Xi, Eigen::VectorXd Xj, int out); //construct matrices
	void updateParam(Eigen::VectorXd x, int model);	//update matrices with new data
	double mValue(int model, int cutD);	//computes hyperplane
	void addPoint(Eigen::VectorXd x, Eigen::VectorXd y, int model); //add point
	void divide(int model);	//divide model that has reached the limit
	double activation(Eigen::VectorXd x, int model);	//obtain prob. of belonging
	void update(Eigen::VectorXd x, Eigen::VectorXd y);	//update a new learned datum
	Eigen::VectorXd predict(Eigen::VectorXd x);
};

#endif 