#include <Windows.h>
#include <iostream>
#include <sstream>

#include <Logger.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <MatrixMulMLModel.h>

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

void MatrixMulMLModel::trainModel() {
	clock_t startTime, endTime;
	startTime = clock();

	Config config;
	config.n_estimators = 10;
	config.learning_rate = 0.1;
	config.max_depth = 6;
	config.min_samples_split = 10;
	config.min_data_in_leaf = 10;
	config.reg_gamma = 0.3;
	config.reg_lambda = 0.3;
	config.colsample_bytree = 0.8;
	config.min_child_weight = 5;
	config.max_bin = 100;

	pandas::Dataset dataset = pandas::ReadCSV("../source/submission.csv", ',', -1,1000);
	// pandas::Dataset dataset = pandas::ReadCSV("../source/credit_card.csv", ',', 5000, 50);
	XGBoost xgboost = XGBoost(config);
	xgboost.fit(dataset.features, dataset.labels);
	cout << xgboost.SaveModelToString() << endl << endl;
	Logger::writeToFile("model.dump", xgboost.SaveModelToString());
	/*
	for (int i = 0; i < dataset.features.size(); i++) {
		cout << i << ": " << xgboost.PredictProba(dataset.features[i])[0] << endl;
	}
	vector<float> pvalues;
	for (size_t i = 0; i < dataset.labels.size(); ++i) {
		pvalues.push_back(xgboost.PredictProba(dataset.features[i])[1]);
	}

	cout << "AUC: " << CalculateAUC(dataset.labels, pvalues) << endl;
	cout << "KS: " << CalculateKS(dataset.labels, pvalues) << endl;
	cout << "ACC: " << CalculateACC(dataset.labels, pvalues) << endl;

	endTime = clock();
	cout << "Totle Time : " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	system("pause");
	*/
}

