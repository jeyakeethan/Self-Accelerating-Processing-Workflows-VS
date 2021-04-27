#ifndef MLMODEL_H
#define MLMODEL_H
#include <Windows.h>
#include <iostream>
#include <sstream>

#include <Logger.h>
#include <ML_Configs.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <vector>

#include <time.h>
#include "config.h"
#include "pandas.h"
#include "xgboost.h"
#include "tree.h"
#include "utils.h"
#include "numpy.h"
#include <list>
#include "rapidjson/document.h"

using namespace std;
using namespace numpy;
using namespace rapidjson;
using namespace xgboost;

class MLModel {
private:
	vector<float> *caching;

public:
	size_t prediction;
	XGBoost * xgboost;
	string model_name, model_path, dataset_path;

	MLModel(string name);
	~MLModel();

	int predict(vector<float>* params);
	bool predictGPU(vector<float>* params);
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
	void dumpModel();
	void loadModel();
};

#endif //MLMODEL_H