#include <numeric>
#include <algorithm>
#include "turning_points.h"
#include <sstream>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include<vector>
#include <cmath>
#include <iostream>

using namespace std;
using namespace rapidjson;

namespace turning_points {
	TurningPoints::TurningPoints() {};
	TurningPoints::~TurningPoints() {};

	void TurningPoints::fit_data(const vector<vector<float>>& features, const vector<int>& labels) {
        m_no_features = features.at(0).size();
        m_turning_points.clear();

        int pre = 0;
        int curr_label;
        
        int training_data_size= labels.size() - 1;

        if (training_data_size == 0) {
            cout << "Training data insufficient!" << endl;
            return;
        }
        
        // initialize bounds
        m_upper_bound = features[0];
        m_lower_bound = features[training_data_size];

        for(size_t idx = 0; idx < training_data_size; idx++) {

            vector<float> features_x = features[idx];

            curr_label = labels[idx];
            if (pre == 0 &&  curr_label== 1) {
                m_turning_points.push_back(features_x);

                // update bounds
                for (int f_index = 0; f_index < m_no_features; f_index++) {
                    if (m_upper_bound[f_index] < features_x[f_index])
                        m_upper_bound[f_index] = features_x[f_index];

                    if (m_lower_bound[f_index] > features_x[f_index])
                        m_lower_bound[f_index] = features_x[f_index];
                }
            }
            pre = curr_label;
        }

        // remove bounds from turning points
        for (auto t_p = m_turning_points.begin(); t_p != m_turning_points.end(); ++t_p) {
            if (*t_p == m_lower_bound || *t_p == m_upper_bound) {
                m_turning_points.erase(t_p);
                t_p--;
            }
        }

        m_no_tps = m_turning_points.size();
        m_tp_last_i = m_no_tps - 1;
	}

	bool TurningPoints::predict(const vector<float>& features) {
        int f_index;
        // upper bound check
        for (f_index = 0; f_index < m_no_features; f_index++) {
            if (features[f_index] > m_upper_bound[f_index])
                return true;
        }

        // lower bounds check
        for (f_index = 0; f_index < m_no_features; f_index++) {
            if (features[f_index] > m_lower_bound[f_index])
                break;
        }
        if (++f_index == m_no_features)
            return false;

        // within bounds check
        for (size_t tp_i = 0; tp_i < m_no_features; tp_i++) {
            if (features > m_turning_points[tp_i])
                return true;
        }
        return true;
	}

    void TurningPoints::print_weights() {
        for (size_t tp_i = 0; tp_i < m_no_tps; tp_i++) {
            vector<float> turning_point = m_turning_points[tp_i];
            for (size_t f = 0; f < m_no_features; f++) {
                cout << turning_point[f] << ", ";
            }
            cout << endl;
        }
        cout << endl;
    }
}
