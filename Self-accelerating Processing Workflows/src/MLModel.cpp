#include <Windows.h>
#include <iostream>
#include <sstream>
#include <cmath>

#include <Logger.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <MLModel.h>
#include <ComputationalModel.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include "config.h"
#include "pandas.h"
#include "xgboost.h"
#include "tree.h"
#include "utils.h"
#include "numpy.h"
#include <list>
using namespace std;
using namespace xgboost;
using namespace pandas;
using namespace numpy;

using namespace std;


#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
    if (err != 0) {                                                         \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
      exit(1);                                                              \
    }                                                                       \
}


MLModel::MLModel(string name) {
	model_name = name;
	loadModel();
	//document.parse(Logger::readFromFile("model.dump"));
	//xgboost->LoadModelFromJson(document);
}

void MLModel::trainModel() {
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

	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/" + model_name + ".csv", ',', -1, 1000);
	xgboost->fit(dataset.features, dataset.labels);

	// print model
	// cout << xgboost.SaveModelToString() << endl << endl;

	// dump model for future use
	// Logger::writeToFile("model.dump", xgboost->SaveModelToString());
}

void MLModel::loadModel() {
	cout << "To do load model" << endl;
	// xgboost = "../ml-models/" + model_name + ".dump"
}

int MLModel::predict(vector<float>* params) {
	//vector<float> temp{ (*params)[1],(*params)[2],(*params)[3] };
	float prediction = xgboost->PredictProba({(*params)[1], (*params)[2], (*params)[3]})[0];

	return (int)round(prediction);
}
