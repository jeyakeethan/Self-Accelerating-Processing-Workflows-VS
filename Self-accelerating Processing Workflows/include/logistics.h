#pragma once
#include <vector>
#include "decision_tree.h"
#include "tree.h"
#include "config.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using namespace rapidjson;
using namespace std;


namespace logistics {
	class Logistics {
	public:
		Logistics(Config conf);
		~Logistics();
		void fit_data(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);
		int predict(const std::vector<double>& features);
		bool predict(const std::vector<float>& features);
		void print_weights();

		const Config config;


	private:
		short no_features;
		vector<double> weights;
		vector<double> error;
	};
}
