#include "dlgp.h"
using Eigen::seq;

dlgp::dlgp(int xSize0, int outs0 ,int pts0, int N0, bool ard0){
	// initialize data
	//see parameter description in dlgp.h
	pts = pts0;
	N = N0;
	xSize = xSize0;
	outs = outs0;
	count = 0;

	X.resize(xSize, pts * N);	X.setZero();
	Y.resize(outs, pts * N);	Y.setZero();
	K.resize(pts * outs, pts * N);	K.setZero();
	//invK.resize(pts * outs, pts * N);	invK.setZero();
	alpha.resize(pts * outs, N);	alpha.setZero();
	
	localCount.resize( 2 * N - 1);	localCount.setZero();
	medians.resize( 2 * N - 1);
	auxUbic.resize( 2 * N - 1);
	auxUbic(0) = 0;
	auxUbic(2 * N - 2) = -1;

	children.resize(2, 2 * N - 1);	children.setConstant(-1);
	
	overlaps.resize(2, 2 * N - 1);

	if (ard0) {
		amountL = xSize;
	}
	else
	{
		amountL = 1; 
	}
}

Eigen::RowVectorXd dlgp::kernel(Eigen::MatrixXd Xi, Eigen::VectorXd Xj, int out) {
	// Compute the squared exponential kernel (aka RBF)
	Eigen::RowVectorXd kern(Xi.cols());
	for (int i = 0; i < Xi.cols(); i += 1) {
		kern(i) = std::pow(sigmaF(out), 2) * exp( -0.5 * ((Xi.col(i)-Xj).array()/lengthS( seq((out)*amountL,(out+1)*amountL-1)) ).pow(2).sum());
	}
	return kern;
}

double mValue(int model, int cutD);

void dlgp::updateParam(Eigen::VectorXd x, int model) {
	localCount(model) += 1;
	int pos = auxUbic(model);
	if (localCount(model) == 1) {	//set parameters for the first datum
		for (int p = 0; p < outs; p += 1) {
			double kVal = kernel(x, x, p)(0);
			K(p * pts, pos * pts) = kVal + std::pow(sigmaN(p), 2);
			alpha(p * pts, pos) = Y(p, pos * pts) / kVal;
		}
	}
	else {
		//update parameters for more data
		Eigen::MatrixXd auxX = X( Eigen::all, seq (pos*pts, pos*pts+localCount(model) - 2) ); // does not consider newest pt
		Eigen::MatrixXd auxY = Y( Eigen::all, seq(pos * pts, pos * pts + localCount(model) - 1) );
		for (int p = 0; p < outs; p += 1) {
			Eigen::MatrixXd b = kernel(auxX, x, p);
			double c = kernel(x, x, p)(0) + std::pow(sigmaN(p),2);
			int auxOut = p * pts;

			
			K( auxOut + localCount(model) - 1 , seq(pos * pts, pos * pts + localCount(model) - 2 ) ) = b;
			K( seq(auxOut, auxOut + localCount(model) - 2) , pos * pts + localCount(model) - 1 ) = b.transpose();
			K( auxOut + localCount(model) - 1 , pos * pts + localCount(model) - 1 ) = c;
			
			
			alpha( seq(auxOut,auxOut+localCount(model) - 1) , pos ) = K( seq(auxOut,auxOut+localCount(model)-1) , 
				seq(pos*pts,pos*pts+localCount(model)-1)  ).ldlt().solve( auxY( p , Eigen::all ).transpose() );
			
		}
	}
}

void dlgp::addPoint(Eigen::VectorXd x, Eigen::VectorXd y, int model) {
	if (localCount(model) < pts) { // if model is not full
		X(Eigen::all, auxUbic(model) * pts + localCount(model) ) = x;
		Y(Eigen::all, auxUbic(model) * pts + localCount(model) ) = y;
		updateParam(x, model);
	}
	if (localCount(model) == pts) {
		divide(model);
	}
}

void dlgp::divide(int model) {
	if (auxUbic(Eigen::last) != -1) {//(parent(Eigen::last) != 0 )
		std::cout << "no room for more divisions" << std::endl << std::endl;
		return;
	}
	//obtain cutting dimension
	Eigen::MatrixXd::Index cutD;
	double width = ( X(Eigen::all, seq(auxUbic(model) * pts, auxUbic(model) * pts + pts - 1)).rowwise().maxCoeff() - 
		X(Eigen::all, seq(auxUbic(model) * pts, auxUbic(model) * pts + pts - 1)).rowwise().minCoeff() ).maxCoeff(&cutD);
	//obtain hyperplane with (max + min) /2
	double mP = ( X(cutD, seq(auxUbic(model) * pts, auxUbic(model) * pts + pts - 1)).maxCoeff() +
		X(cutD, seq(auxUbic(model) * pts, auxUbic(model) * pts + pts - 1)).minCoeff() ) / 2;
	//set overlapping region and median
	double o = width / wo; 
	if (o == 0)
		o = 0.1;

	medians(model) = mP;
	overlaps(0, model) = cutD;
	overlaps(1, model) = o;
	//matrices for left/right models
	Eigen::MatrixXd xL(xSize,pts) , xR(xSize, pts);
	Eigen::MatrixXd yL(outs, pts) , yR(outs, pts);

	int lcount = 0;
	int rcount = 0;
	Eigen::VectorXi iL(pts), iR(pts); //vector with indeces

	for (int i = 0; i < pts; i += 1) { //sort the data
		double xD = X(cutD,  auxUbic(model) * pts + i); 
		if (xD < mP - 0.5 * o){ // if in left
			xL(Eigen::all, lcount) = X(Eigen::all, auxUbic(model) * pts + i);
			yL(Eigen::all, lcount) = Y(Eigen::all, auxUbic(model) * pts + i);
			iL(lcount) = i;
			lcount += 1;
		}
		else if (xD >= mP - 0.5*o && xD <= mP + 0.5*o ){ // if in overlapping
			double pL = 0.5 + (xD - mP) / o;
			if (pL >= (double)((rand() % 10000) + 1) / 10000) { //select left
				xL(Eigen::all, lcount) = X(Eigen::all, auxUbic(model) * pts + i);
				yL(Eigen::all, lcount) = Y(Eigen::all, auxUbic(model) * pts + i);
				iL(lcount) = i;
				lcount += 1;
			}
			else { //select right
				xR(Eigen::all, rcount) = X(Eigen::all, auxUbic(model) * pts + i);
				yR(Eigen::all, rcount) = Y(Eigen::all, auxUbic(model) * pts + i);
				iR(rcount) = i;
				rcount += 1;
			}
		}
		else if (xD > mP + 0.5*o) { //if in right
			xR(Eigen::all, rcount) = X(Eigen::all, auxUbic(model) * pts + i);
			yR(Eigen::all, rcount) = Y(Eigen::all, auxUbic(model) * pts + i);
			iR(rcount) = i;
			rcount += 1;
		}
	}
	localCount(model) = 0; //divided model is now empty
	if (count == 0)
		count += 1;
	else
		count += 2;
	children(0, model) = count; children(1, model) = count + 1; //assign children
	//set parameters of new models
	localCount(count) = lcount; localCount(count + 1) = rcount;
	auxUbic(count) = auxUbic(model); auxUbic(count + 1) = auxUbic.maxCoeff()+1;

	//values for K permutation
	Eigen::VectorXi order(pts);
	order << iL(seq(0, lcount - 1)), iR(seq(0, rcount - 1));
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(pts);
	perm.indices() = order.array();
	
	for (int p = 0; p < outs; p += 1) { //compute parameters of new model
		Eigen::MatrixXd newK = K(seq(p * pts, (p + 1) * pts -1), seq(auxUbic(model) * pts, auxUbic(model) * pts + pts -1));
		//permute K
		newK = perm.transpose() * newK *perm;
		//std::cout << newK << std::endl;
		//set child K
		K(seq(p * pts, p * pts + lcount - 1), seq(auxUbic(count) * pts, auxUbic(count) * pts + lcount - 1)) =
			newK(seq(0, lcount-1), seq(0, lcount-1));
		K(seq(p * pts, p * pts + rcount - 1), seq(auxUbic(count+1) * pts, auxUbic(count+1) * pts + rcount - 1)) =
			newK(seq(lcount, Eigen::last), seq(lcount, Eigen::last));
		//set child alphas
		alpha(seq(p * pts, p * pts + lcount - 1), auxUbic(count)) =
			newK(seq(0, lcount-1), seq(0, lcount-1)).ldlt().solve(yL(p, seq(0, lcount - 1)).transpose());
		alpha(seq(p * pts, p * pts + rcount - 1), auxUbic(count+1)) =
			newK(seq(lcount, Eigen::last), seq(lcount, Eigen::last)).ldlt().solve(yR(p, seq(0, rcount - 1)).transpose());
	}

	auxUbic(model) = -1; // parent set will not have more data
	//relocate X and Y
	X(Eigen::all, seq(auxUbic(count) * pts, auxUbic(count) * pts + pts - 1)) = xL;
	X(Eigen::all, seq(auxUbic(count+1) * pts, auxUbic(count+1) * pts + pts - 1)) = xR;
	Y(Eigen::all, seq(auxUbic(count) * pts, auxUbic(count) * pts + pts - 1)) = yL;
	Y(Eigen::all, seq(auxUbic(count + 1) * pts, auxUbic(count + 1) * pts + pts - 1)) = yR;
}

double dlgp::activation(Eigen::VectorXd x, int model) {
	if (children(1, model) == -1) //return zero when model is a leaf
		return 0;
	double mP = medians(model); //hyperplane value
	double xD = x((int)overlaps(0, model)); //value of x in cutting dimension
	double o = overlaps(1, model); //overlapping region
	if (xD < mP - 0.5 * o)
		return 1;
	else if (xD >= mP - 0.5 * o && xD <= mP + 0.5 * o) // if in overlapping
		return 0.5 + (xD - mP) / o;
	else
		return 0;
}

void dlgp::update(Eigen::VectorXd x, Eigen::VectorXd y) {
	int model = 0;
	if (localCount(3) == 48)
		int a = 0;
	while (children(1, model) != -1) { // if model is a parent
		// search for a leaf to asign the point
		double pL = activation(x, model);
		if (pL >= (double)((rand() % 10000) + 1) / 10000)
			model = children(0, model);// left child
		else
			model = children(1, model);
	}
		// add the model to the randomly selected model
	addPoint(x, y, model);
}

Eigen::VectorXd dlgp::predict(Eigen::VectorXd x) {
	Eigen::VectorXd out(outs);	out.setConstant(0);
	Eigen::VectorXi models(1000); // active models
	Eigen::VectorXd probs(1000); // global probs. 

	models(0) = 0; //start in root
	models(1) = 0;
	probs(0) = 1;
	probs(1) = 1;
	int mCount = 1; //amount of GPs used for prediction
	//while (children(0, models(mCount-1)) != -1 && children(0, models( mCount)) != -1) {
	while (children(0, models(seq(0,mCount - 1))).array().sum() != -1 * mCount)  {
		for (int j = 0; j < mCount; j += 1) {
			if (children(0, models(j)) != -1) { // go deeper in tree if node has children
				double pL = activation(x, models(j));
				if (pL == 1)
					models(j) = children(0, models(j));
				else if (pL == 0)
					models(j) = children(1, models(j));
				else if (1 > pL > 0) { //both children have non-zero probs.
					mCount += 1;

					models(mCount - 1) = children(1, models(j));
					probs(mCount - 1) = probs(1, j) * (1 - pL);

					models(j) = children(0, models(j));
					probs(j) = probs(j) * pL;
				}
			}
		}
	}
	//prediction weighting predictions
	Eigen::VectorXd pred(1);
	for (int p = 0; p < outs; p += 1) {
		for (int i = 0; i < mCount; i += 1) {
			int model = models(i);
			pred = kernel(X(Eigen::all, seq(auxUbic(model) * pts, auxUbic(model) * pts + localCount(model) - 1)), x, p)*
				alpha(seq(p*pts, p*pts+localCount(model)-1), auxUbic(model) );
			out(p) += pred(0) * probs(i);
		}
	}
	return out;

}
