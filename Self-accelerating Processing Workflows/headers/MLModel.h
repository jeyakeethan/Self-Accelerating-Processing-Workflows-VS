#ifndef MLMODEL_H
#define MLMODEL_H
#include <Windows.h>
#include <iostream>
#include <sstream>

#include <Constants.h>
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
#include "tree.h"
#include "utils.h"
#include "numpy.h"
#include <list>
#include "rapidjson/document.h"

using namespace std;
using namespace numpy;
using namespace rapidjson;

class MLModel {
private:
	vector<float> *caching;
	bool *caching_pred;

public:
	size_t prediction;
	ML_Algo* model;
	string model_name, model_path, dataset_path;

	MLModel(string name);
	~MLModel();

	int predict(vector<float>* params);
	bool predict_logic(vector<float> &params);
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

private:
	int m_no_features = 0;
	int m_no_tps = 0;
	int m_tp_last_i = 0;
	vector<float> m_leasts;
	vector<float> m_highests;
	vector<float> m_ranges;
	vector<float> m_upper_bound;
	vector<float> m_lower_bound;
	vector<float> m_center_bound;
	vector<vector<float>> m_turning_points;
	vector<double> error;
	void fit_data_local(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);
	float calculate_distance(vector<float>& a, vector<float>& b);
	void print_turning_points();
	void print_center_bound();
};

#endif //MLMODEL_H