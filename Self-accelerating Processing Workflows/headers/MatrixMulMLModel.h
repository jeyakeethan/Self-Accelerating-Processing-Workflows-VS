#ifndef MATRIXMULMLMODEL_H
#define MATRIXMULMLMODEL_H
#include <Windows.h>
#include <iostream>
#include <sstream>

#include <Logger.h>
#include <ML_Configs.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <MatrixMulMLModel.h>

#include <time.h>
#include "config.h"
#include "pandas.h"
#include "xgboost.h"
#include "tree.h"
#include "utils.h"
#include "numpy.h"
#include <list>
#include "rapidjson/document.h"

using namespace rapidjson;
using namespace std;
using namespace xgboost;

class MatrixMulMLModel {
public:
	size_t prediction;
	XGBoost * xgboost;
	MatrixMulMLModel();
	inline ~MatrixMulMLModel() { };

	int predict(vector<float>* params);
	/*inline mat intArrToMat(int* params) {
		stringstream s;
		s << "\"";
		for (int i = 1; i <= params[0]; i++)
			s << params[1] << ";";
		s << "\"";
		mat paramsMat(s.str());
		return paramsMat;
	}*/
	void trainModel();
	static void trainModelStatic();
};

#endif //MATRIXMULMLMODEL_H