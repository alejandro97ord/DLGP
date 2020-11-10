#include<iostream>
#include"dlgp.h"
#include<fstream>
#include<string>
#include<chrono>

Eigen::MatrixXd readFile(char* fileName) {
	std::ifstream f0;
	f0.open(fileName);
	if (f0.fail()) {
		std::cerr << "Error, file "<< fileName <<" not located" << std::endl;
		Eigen::MatrixXd err(1, 1); err.setZero();
		return err;
	}
	

	std::string line;
	int numC;
	int numR = 0;
	int aux = 0;
	
	static double sink[(int)1e7];

	while (!f0.eof()) {

		while (getline(f0, line))
		{
			
			std::stringstream stream(line);
			std::string value;
			
			for (int c = 0; getline(stream, value, ','); c += 1) {
				sink[aux] = std::stod(value);
				aux += 1;
				numC = c+1;
			}
			numR += 1;
		}
	}
	f0.close();
	aux = 0;
	Eigen::MatrixXd output(numR,numC); 
	for (int f = 0; f < numR; f += 1) {
		for (int c = 0; c < numC; c += 1) {
			output(f, c) = sink[aux];
			aux += 1;
		}
	}
	return output;
}

int main() {
	Eigen::MatrixXd X_train = readFile((char*) "../cpp/X_train.txt");
	Eigen::MatrixXd Y_train = readFile((char*)"../cpp/Y_train.txt");
	Eigen::MatrixXd X_test = readFile((char*)"../cpp/X_test.txt");
	Eigen::MatrixXd Y_test = readFile((char*)"../cpp/Y_test.txt");
	Eigen::MatrixXd sigmaF = readFile((char*)"../cpp/sigmaF.txt");
	Eigen::MatrixXd sigmaN = readFile((char*)"../cpp/sigmaN.txt");
	Eigen::MatrixXd ls = readFile((char*)"../cpp/ls.txt");

	dlgp gp01(21,7,50,5000,true); //(xSize, outs, pts, N, ard)
	gp01.sigmaF.resize(2, 1); gp01.sigmaF = sigmaF;
	gp01.sigmaN.resize(2, 1); gp01.sigmaN = sigmaN;
	gp01.lengthS.resize(12, 1); gp01.lengthS = ls;
	gp01.wo = 200000;

	for (int h = 0; h < 44484; h += 1) {
		//std::cout << h << std::endl;

		//auto start = std::chrono::high_resolution_clock::now();
		gp01.update(X_train(Eigen::all, h), Y_train(Eigen::all, h));
		//auto stop = std::chrono::high_resolution_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		//std::cout << duration.count() << std::endl;
	}

	Eigen::MatrixXd outs(Y_test.rows(), Y_test.cols());
	for (int h = 0; h < 4449; h += 1) {
		outs (Eigen::all, h) = gp01.predict(X_test(Eigen::all, h));
	}
	Eigen::VectorXd results(7);
	Eigen::VectorXd vars = pow(Y_test.array().colwise() - Y_test.rowwise().mean().array(), 2).rowwise().sum().array() / 4449;
	results = pow((outs - Y_test).array(), 2).rowwise().mean().array() / vars.array();
	std::cout << results << std::endl;
	std::cout << '\a';
	return 0;
}