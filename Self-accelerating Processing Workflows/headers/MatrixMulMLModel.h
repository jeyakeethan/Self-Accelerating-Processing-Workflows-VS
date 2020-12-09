#ifndef MATRIXMULMLMODEL_H
#define MATRIXMULMLMODEL_H
#include <Windows.h>
#include <iostream>
#include <sstream>

class MatrixMulMLModel {
public:
	size_t prediction;

	inline MatrixMulMLModel() { };
	inline ~MatrixMulMLModel() { };

	int predict(int* params);
	/*inline mat intArrToMat(int* params) {
		stringstream s;
		s << "\"";
		for (int i = 1; i <= params[0]; i++)
			s << params[1] << ";";
		s << "\"";
		mat paramsMat(s.str());
		return paramsMat;
	}*/
	static void trainModel();
};

#endif //MATRIXMULMLMODEL_H