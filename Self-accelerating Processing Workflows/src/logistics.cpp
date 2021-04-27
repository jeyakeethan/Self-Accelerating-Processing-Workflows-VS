#include <numeric>
#include <algorithm>
#include "logistics.h"
#include <sstream>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include<vector>
#include <cmath>
#include <iostream>

using namespace std;
using namespace rapidjson;

bool custom_sort(double a, double b) /* this custom sort function is defined to                                  sort on basis of min absolute value or error*/
{
    double  a1 = abs(a - 0);
    double  b1 = abs(b - 0);
    return a1 < b1;
}

namespace logistics {
	Logistics::Logistics(Config conf) :config(conf) {};
	Logistics::~Logistics() {};
	void Logistics::fit_data(const vector<vector<float>>& features, const vector<int>& labels) {
        no_features = features.at(0).size();

        // fill weights
        weights.clear();
        for (int n = 0; n <= no_features; n++)
            weights.push_back(0);

        double err;

        for (int epoch = 0; epoch < config.number_of_epoch; epoch++) {
            for (size_t idx = 0; idx < labels.size(); ++idx) {

                vector<float> features_x = features[idx];
                features_x.push_back(-1);

                double p = weights[no_features];
                for (int w = 0; w < no_features; w++) {
                    p += weights[w] * features_x[w];
                }
                double pred = 1 / (1 + exp(-p)); //calculating final prediction applying sigmoid
                err = labels[idx] - pred;       //calculating the error

                for (int w = 0; w <= no_features; w++) {
                    weights[w] = weights[w] + config.alpha * err * pred * (1 - pred) * features_x[w];
                }
                error.push_back(err);
            }
        }
        sort(error.begin(), error.end(), custom_sort);      //custom sort based on absolute error difference
	}

	int Logistics::predict(const vector<double>& features) {

        double p = weights[no_features];

        for (int w = 0; w < no_features; w++) {
            p += weights[w] * features[w];
        }
        return p >= 0.5 ? 1 : 0;
	}

	bool Logistics::predict(const vector<float>& features) {

        double p = weights[no_features];

        for (int w = 0; w < no_features; w++) {
            p += weights[w] * features[w];
        }
        return p >= 0.5;
	}

    void Logistics::print_weights() {
        for (int w = 0; w <= no_features; w++) {
            cout << weights[w] << ", ";
        }
        cout << "." << endl;
    }
}
