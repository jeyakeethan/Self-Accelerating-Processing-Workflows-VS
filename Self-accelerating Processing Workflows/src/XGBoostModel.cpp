#include <Windows.h>
#include <iostream>
#include <thread>
#include <sstream>
#include <Constants.h>
using namespace std;

#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/F1.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;


class MatrixMulXGBoostModel {
	public:
		MatrixMulXGBoostModel() {}
		static int predict(int* params) {
			// TO DO decison of the model here

			RandomForest<GiniGain, RandomDimensionSelect> rf;
			Row<size_t> predictions;
			mlpack::data::Load("mymodel.xml", "model", rf);

			mat sample("800; 200; 900;"
				" 0; 0; 0");
			mat probabilities;
			rf.Classify(sample, predictions, probabilities);
			u64 result = predictions.at(0);
			cout << "\nClassification result: " << result << " , Probabilities: " <<
				probabilities.at(0) << "/" << probabilities.at(1);

			return 0;
		}
	private:

};