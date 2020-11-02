#include <stdio.h>
#include <python.h>
#include <pyhelper.cpp>



static long predictMatMul(int* attr) {
	CPyInstance hInstance;

	CPyObject pName = PyUnicode_FromString("predictor");
	CPyObject pModule = PyImport_Import(pName);
	if (pModule) {
		CPyObject pFunc = PyObject_GetAttrString(pModule, "predictTime");
		PyObject* args = PyList_New(3);
		PyArg_ParseTuple(args, "0", attr);
		if (pFunc && PyCallable_Check(pFunc))
		{
			CPyObject pValue = PyObject_CallObject(pFunc, args);

			return PyLong_AsLong(pValue);
		}
	}
	else {

		return 0;
	}
	return 0;
}