#include <Windows.h>
#include <iostream>
#include <sstream>
#include <MatrixMulMLModel.h>

#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
// #include "mlpack/core/cv/k_fold_cv.hpp"
// #include "mlpack/core/cv/metrics/accuracy.hpp"
// #include "mlpack/core/cv/metrics/precision.hpp"
// #include "mlpack/core/cv/metrics/recall.hpp"
// #include "mlpack/core/cv/metrics/F1.hpp"
// using namespace mlpack::cv;

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
    if (err != 0) {                                                         \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
      exit(1);                                                              \
    }                                                                       \
}

MatrixMulMLModel::~MatrixMulMLModel(){}


int MatrixMulMLModel::predict(int* params) {
	stringstream s;
	s << "\"" << params[1] << ";" << params[2] << ";" << params[3] << ";\"";
	std::cout << s.str() << endl;
	mat attr(s.str());
	return rf.Classify(attr);
}
