#pragma once
#include <vector>
#include "tree.h"
#include "config.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using namespace rapidjson;
using namespace std;

namespace turning_points {
	class TurningPoints {
	public:
		TurningPoints();
		~TurningPoints();
		void fit_data(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);
		bool predict(const std::vector<float>& features);
		void print_weights();

	private:
		int m_no_features = 0;
		int m_no_tps = 0;
		int m_tp_last_i = 0;
		vector<float> m_upper_bound;
		vector<float> m_lower_bound;
		vector<vector<float>> m_turning_points;
		vector<double> error;
	};
}
