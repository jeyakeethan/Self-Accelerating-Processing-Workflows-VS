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
	inline MatrixMulMLModel() {
		Config mlConfig;
		mlConfig.n_estimators = 10;
		mlConfig.learning_rate = 0.1;
		mlConfig.max_depth = 6;
		mlConfig.min_samples_split = 10;
		mlConfig.min_data_in_leaf = 10;
		mlConfig.reg_gamma = 0.3;
		mlConfig.reg_lambda = 0.3;
		mlConfig.colsample_bytree = 0.8;
		mlConfig.min_child_weight = 5;
		mlConfig.max_bin = 100;
		xgboost = new XGBoost(mlConfig);
		//Document document;
		//document.parse(Logger::readFromFile("model.dump"));
		//xgboost->LoadModelFromJson(document);
	};
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
	static void trainModel();
};

#endif //MATRIXMULMLMODEL_H