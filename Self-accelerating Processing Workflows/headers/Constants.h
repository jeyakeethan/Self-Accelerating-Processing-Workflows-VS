#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
typedef int numericalType1;

const bool LOGGER_MODE_ON = true;

// Test Data Generation
const int INPUT_NATURE = 1;
const int N = 10000;
const int EXPERIMENT_COUNT = 10000;
const int MAX_WIDTH_ALIGNED = 200;
const int MIN_WIDTH_ALIGNED = 10;

const int SMALL_ARRAY_MAX_LENGTH = 1300;
const int ARRAY_MAX_LENGTH = 10000;

// Computational Model Constant
const int REVISE_COUNT_MIN = 20;
const int REVISE_COUNT_MAX = 1000;
const int REVISE_COUNT_STEP = 10;
const int SAMPLE_COUNT = 5;
const int RANGE_OF_INT_VALUES = 32;


const int THREADS_PER_BLOCK = 1024;

const int DAY = 86400;
const int MONTH = 86400*30;
const int REVISE_PERIOD = 1800;					// in seconds
const int CPU_LOAD_REVISE_PERIOD = 3600;		// in seconds
const int ML_TRAIN_CHECK_PERIOD = DAY;			// in seconds
const int RESET_COUNT = 19;

const std::string LOG_FILE_NAME = "statistics.log";

struct myDim3 {
	short x, y, z;
	myDim3() :x(1), y(1), z(1) {}
	myDim3(short x_, short y_, short z_):x(x_), y(y_), z(z_) {}
};

// ML part
const short NUMBER_OF_PREDICTIONS_TO_BE_CACHED = 5;
const int MAX_OUTLIERS_LIMIT = 5;


#endif // CONSTANTS_H

