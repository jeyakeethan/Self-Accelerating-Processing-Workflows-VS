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

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/istreamwrapper.h""
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;
using namespace xgboost;
using namespace pandas;
using namespace numpy;

using namespace std;
using namespace rapidjson;


#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
    if (err != 0) {                                                         \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
      exit(1);                                                              \
    }                                                                       \
}

MLModel::MLModel(string name) {
	model_name = name;
	model_path = "../ml-models/" + model_name + ".json";
	dataset_path = "../ml-datasets/" + model_name + ".csv";

	/*if (fs::exists(model_path)) {
		loadModel();
	}
	else*/
		if(fs::exists(dataset_path)){
		trainModel();
		cout << "ML model is being trained! please wait for a moment..." << endl;
		dumpModel();
	}
	else {
		cout << "Neither ML model nor training dataset exists!" << endl;
		exit;
	}
}

MLModel::~MLModel() {
	delete xgboost;
}

void MLModel::trainModel() {
	Config mlConfig;
	mlConfig.n_estimators = 10;
	mlConfig.learning_rate = 0.1f;
	mlConfig.max_depth = 6;
	mlConfig.min_samples_split = 10;
	mlConfig.min_data_in_leaf = 10;
	mlConfig.reg_gamma = 0.3f;
	mlConfig.reg_lambda = 0.3f;
	mlConfig.colsample_bytree = 0.8f;
	mlConfig.min_child_weight = 5;
	mlConfig.max_bin = 100;

	xgboost = new XGBoost(mlConfig);

	pandas::Dataset dataset = pandas::ReadCSV(dataset_path, ',', -1, 1000);
	xgboost->fit(dataset.features, dataset.labels);
}

void MLModel::dumpModel() {
	ofstream ofs(model_path);
	ofs << xgboost->SaveModelToString().c_str();
	ofs.close();
}

void MLModel::loadModel() {
	ifstream ifs(model_path);
	IStreamWrapper isw(ifs);

	Document doc;
	doc.ParseStream(isw);

	ifs.close();

	Config mlConfig;

	if (doc.HasMember("Param")) {
		const Value& Param = doc["Param"];
		mlConfig.n_estimators = Param["n_estimators"].GetInt();
		mlConfig.learning_rate = Param["learning_rate"].GetFloat();
		mlConfig.max_depth = Param["max_depth"].GetInt();
		mlConfig.min_samples_split = Param["min_samples_split"].GetInt();
		mlConfig.min_data_in_leaf = Param["min_data_in_leaf"].GetInt();
		mlConfig.min_child_weight = Param["min_child_weight"].GetFloat();
		mlConfig.reg_gamma = Param["reg_gamma"].GetFloat();
		mlConfig.reg_lambda = Param["reg_lambda"].GetFloat();
		mlConfig.colsample_bytree = Param["colsample_bytree"].GetFloat();
		mlConfig.max_bin = Param["max_bin"].GetInt();
		
		/*cout << Param["n_estimators"].GetInt() << endl;
		cout << Param["learning_rate"].GetFloat() << endl;
		cout << Param["max_depth"].GetInt() << endl;
		cout << Param["min_samples_split"].GetInt() << endl;
		cout << Param["min_data_in_leaf"].GetInt() << endl;
		cout << Param["reg_gamma"].GetFloat() << endl;
		cout << Param["reg_lambda"].GetFloat() << endl;
		cout << Param["colsample_bytree"].GetFloat() << endl;
		cout << Param["min_child_weight"].GetFloat() << endl;
		cout << Param["max_bin"].GetInt() << endl;*/
	}

	xgboost = new XGBoost(mlConfig);

	if (doc.HasMember("Trees")) {
		const rapidjson::Value& trees = doc["Trees"];

		for (rapidjson::Value::ConstValueIterator itr = trees.Begin(); itr != trees.End(); ++itr)
		{
			// const rapidjson::Value& tree = *itr;
			xgboost->trees.push_back(xgboost->LoadTreeFromJson(*itr));
		}
	}
}

int MLModel::predict(vector<float>* params) {
	//vector<float> temp{ (*params)[1],(*params)[2],(*params)[3] };
	//float prediction = xgboost->PredictProba({(*params)[1], (*params)[2], (*params)[3]})[0];
	float prediction = xgboost->PredictProba(*params)[0];
	int pre = (int)round(prediction);
	//cout << pre;
	return pre;
}

bool MLModel::predictGPU(vector<float>* params) {
	//vector<float> temp{ (*params)[1],(*params)[2],(*params)[3] };
	//float prediction = xgboost->PredictProba({(*params)[1], (*params)[2], (*params)[3]})[0];
	float prediction = xgboost->PredictProbability(*params);
	return prediction <= 0.5;
}