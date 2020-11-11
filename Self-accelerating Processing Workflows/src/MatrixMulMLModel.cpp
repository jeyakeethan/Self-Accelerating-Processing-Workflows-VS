#include <Windows.h>
#include <iostream>
#include <sstream>
#include <MatrixMulMLModel.h>
using namespace std;

#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
// #include "mlpack/core/cv/k_fold_cv.hpp"
// #include "mlpack/core/cv/metrics/accuracy.hpp"
// #include "mlpack/core/cv/metrics/precision.hpp"
// #include "mlpack/core/cv/metrics/recall.hpp"
// #include "mlpack/core/cv/metrics/F1.hpp"
// using namespace mlpack::cv;

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;

MatrixMulMLModel::~MatrixMulMLModel(){}


int MatrixMulMLModel::predict(int* params) {
	stringstream s;
	s << "\"" << params[1] << ";" << params[2] << ";" << params[3] << ";\"";
	std::cout << s.str() << endl;
	mat attr(s.str());
	rf.Classify(attr, prediction, probabilities);
	return prediction;
}
