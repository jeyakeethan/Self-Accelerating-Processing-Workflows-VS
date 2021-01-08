#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <list>
#include "config.h"
#include "pandas.h"
#include "decision_tree.h"
#include "numpy.h"
using namespace std;


namespace xgboost {
	BaseDecisionTree::BaseDecisionTree(Config conf) :config(conf) {};
	//BaseDecisionTree::~BaseDecisionTree() {};

	Tree* BaseDecisionTree::fit(const vector<vector<float>>& features_in, const vector<int>& labels_in,
		const vector<float>& grad_in, const vector<float>& hess_in) {
		features = features_in;
		labels = labels_in;
		grad = grad_in;
		hess = hess_in;
		vector<int> sub_dataset;
		for (size_t i = 0; i < labels.size(); ++i) {
			sub_dataset.push_back(i);
		}
		decision_tree = _fit(sub_dataset, 1);
		return decision_tree;
	}

	Tree* BaseDecisionTree::_fit(vector<int>& sub_dataset, int depth) {
		float sub_hess = 0.0;
		for (int i : sub_dataset) {
			sub_hess += hess[i];
		}


		if (sub_dataset.size() <= config.min_samples_split || sub_hess <= config.min_child_weight) {
			Tree *tree = new Tree();
			tree->leaf_value = CalculateLeafValue(sub_dataset);
			return tree;
		}

		if (depth <= config.max_depth) {
			int dataset_index;
			for (size_t j = 0; j < sub_dataset.size(); ++j) {
				dataset_index = sub_dataset[j];
			}
			BestSplitInfo best_split_info = ChooseBestSplitFeature(sub_dataset);
			Tree *tree = new Tree();

			if (best_split_info.best_sub_dataset_left.size() < config.min_data_in_leaf ||
				best_split_info.best_sub_dataset_right.size() < config.min_data_in_leaf) {
				tree->leaf_value = CalculateLeafValue(sub_dataset);
				return tree;
			}
			else {
				tree->split_feature = best_split_info.best_split_feature;
				tree->split_value = best_split_info.best_split_value;
				tree->split_gain = best_split_info.best_split_gain;
				tree->internal_value = best_split_info.best_internal_value;
				tree->tree_left = _fit(best_split_info.best_sub_dataset_left, depth + 1);
				tree->tree_right = _fit(best_split_info.best_sub_dataset_right, depth + 1);
				return tree;
			}
		}
		else {
			Tree *tree = new Tree();
			tree->leaf_value = CalculateLeafValue(sub_dataset);
			return tree;
		}
	}

	BestSplitInfo BaseDecisionTree::ChooseBestSplitFeature(const vector<int>& sub_dataset) {
		BestSplitInfo best_split_info;

		float best_internal_value = CalculateLeafValue(sub_dataset);
		best_split_info.best_internal_value = best_internal_value;
		
		list<BestSplitInfo> best_split_info_list(features[0].size(), BestSplitInfo());

#pragma omp parallel for schedule(static, 1)
		for (int i = 0; i < features[0].size(); ++i) {
			best_split_info_list.push_back(ChooseBestSplitValue(sub_dataset, i));
		}

		list<BestSplitInfo>::iterator iter = best_split_info_list.begin();
		while (iter != best_split_info_list.end()) {
			if ((*iter).best_split_gain > best_split_info.best_split_gain) {
				best_split_info.best_split_gain = (*iter).best_split_gain;
				best_split_info.best_split_feature = (*iter).best_split_feature;
				best_split_info.best_split_value = (*iter).best_split_value;
				best_split_info.best_sub_dataset_left = (*iter).best_sub_dataset_left;
				best_split_info.best_sub_dataset_right = (*iter).best_sub_dataset_right;
			}
			++iter;
		}

		return best_split_info;
	}

	BestSplitInfo BaseDecisionTree::ChooseBestSplitValue(const vector<int>& sub_dataset, int feature_index) {

		vector<float> feature_values;
		vector<float> feature_values_unique;

		int dataset_index;
		for (size_t j = 0; j < sub_dataset.size(); ++j) {
			dataset_index = sub_dataset[j];
			feature_values.push_back(features[dataset_index][feature_index]);
			feature_values_unique.push_back(features[dataset_index][feature_index]);
		}

		vector<float> unique_values;
		sort(feature_values_unique.begin(), feature_values_unique.end());
		feature_values_unique.erase(unique(feature_values_unique.begin(), feature_values_unique.end()), feature_values_unique.end());
		if (feature_values_unique.size() <= config.max_bin) {
			unique_values = feature_values_unique;
		}
		else {
			vector<float> lins = numpy::Linspace(0, 100, config.max_bin);
			sort(feature_values.begin(), feature_values.end());
			for (size_t i = 0; i < lins.size(); ++i) {
				float p = lins[i];
				unique_values.push_back(numpy::Percentile(feature_values, p));
			}
			unique_values.erase(unique(unique_values.begin(), unique_values.end()), unique_values.end());
		}
		vector<int> sub_dataset_left;
		vector<int> sub_dataset_right;
		float left_grad_sum;
		float left_hess_sum;
		float right_grad_sum;
		float right_hess_sum;
		float split_gain;

		BestSplitInfo best_split_info;
		best_split_info.best_split_feature = feature_index;

		for (float split_value : unique_values) {
			sub_dataset_left.clear();
			sub_dataset_right.clear();
			left_grad_sum = 0;
			left_hess_sum = 0;
			right_grad_sum = 0;
			right_hess_sum = 0;


			for (int index : sub_dataset) {
				if (features[index][feature_index] <= split_value) {
					sub_dataset_left.push_back(index);
					left_grad_sum += grad[index];
					left_hess_sum += hess[index];
				}
				else {
					sub_dataset_right.push_back(index);
					right_grad_sum += grad[index];
					right_hess_sum += hess[index];
				}
			}
			split_gain = CalculateSplitGain(left_grad_sum, left_hess_sum, right_grad_sum, right_hess_sum);
			if (best_split_info.best_split_gain < split_gain) {
				best_split_info.best_split_gain = split_gain;
				best_split_info.best_split_feature = feature_index;
				best_split_info.best_split_value = split_value;
				best_split_info.best_sub_dataset_left = sub_dataset_left;
				best_split_info.best_sub_dataset_right = sub_dataset_right;
			}
		}
		return best_split_info;
	}


	float BaseDecisionTree::CalculateSplitGain(const float& left_grad_sum, const float& left_hess_sum,
		const float& right_grad_sum, const float& right_hess_sum) {
		float tmp1 = pow(left_grad_sum, 2) / (left_hess_sum + config.reg_lambda);
		float tmp2 = pow(right_grad_sum, 2) / (right_hess_sum + config.reg_lambda);
		float tmp3 = pow((left_grad_sum + right_grad_sum), 2) / (left_hess_sum + right_hess_sum + config.reg_lambda);
		return (tmp1 + tmp2 - tmp3) / 2 - config.reg_gamma;
	}


	float BaseDecisionTree::CalculateLeafValue(const vector<int>& sub_dataset) {
		float grad_sum = 0;
		float hess_sum = 0;
		for (int index : sub_dataset) {
			grad_sum += grad[index];
			hess_sum += hess[index];
		}
		return -grad_sum / (hess_sum + config.reg_lambda);
	}
}
