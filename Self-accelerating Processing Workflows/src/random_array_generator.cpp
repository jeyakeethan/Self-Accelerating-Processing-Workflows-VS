#include <Constants.h>

#ifndef FUNCTIONS
#define FUNCTIONS
numericalType1* generate_1d_array(int length) {
	numericalType1* arr = new numericalType1[length];
	for (int i = 0; i < length; i++) {
		arr[i] = rand() % RANGE_OF_INT_VALUES;
	}
	return arr;
}

numericalType1** generate_2d_array(int x, int y) {
	numericalType1** arr = new numericalType1*[x];
	for (int i = 0; i < x; i++) {
		arr[i] = new numericalType1[y];
		for (int j = 0; j < y; j++) {
			arr[i][j] = rand() % RANGE_OF_INT_VALUES;
		}
	}
	return arr;
}

numericalType1*** generate_3d_array(int x, int y, int z) {
	numericalType1*** arr = new numericalType1 ** [x];
	for (int i = 0; i < x; i++) {
		arr[i] = new numericalType1*[y];
		for (int j = 0; j < y; j++) {
			arr[i][j] = new numericalType1[y];
			for (int k = 0; k < z; k++) {
				arr[i][j][k] = rand() % RANGE_OF_INT_VALUES;
			}
		}
	}
	return arr;
}
#endif