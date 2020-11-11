#ifndef MATRIXMULMLMODEL_H
#define MATRIXMULMLMODEL_H
#include <Windows.h>
#include <iostream>
#include <sstream>

#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;


class MatrixMulMLModel {
public:
	RandomForest<GiniGain, RandomDimensionSelect> rf;
	arma::vec probabilities;
	size_t prediction;

	inline MatrixMulMLModel() { mlpack::data::Load("mymodel.xml", "model", rf); };
	~MatrixMulMLModel();

	int predict(int* params);
	inline mat intArrToMat(int* params) {
		stringstream s;
		s << "\"";
		for (int i = 1; i <= params[0]; i++)
			s << params[1] << ";";
		s << "\"";
		mat paramsMat(s.str());
		return paramsMat;
	}
};

#endif //MATRIXMULMLMODEL_H