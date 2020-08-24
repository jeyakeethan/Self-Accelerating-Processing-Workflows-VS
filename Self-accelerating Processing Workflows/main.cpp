#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <Constants.h>
#include <ComputationalModel.h>
#include <ArrayAdditionModel.h>
#include <random>

#include <fstream>
#include <string>
#include <sstream>


using namespace std;
int main()
{
    /* int inputA[N];
     int inputB[N];*/
     /*int output[N];*/

    ArrayAdditionModel arrayAdditionModel;

    // ---------------steam function -------------------------------
    fstream newfile;

    int* array1, siz;
    int* array2, size;
    int* output, sizes;

    newfile.open("tpoint.txt", ios::in); //open a file to perform read operation using file object
    if (newfile.is_open()) {   //checking whether the file is open

        string tp;

        while (getline(newfile, tp)) { //read data from file object and put it into string.
            //cout << tp << "\n";

            std::string s = tp;
            std::string delimiter = " ";

            size_t pos = 0;
            std::string token;
            while ((pos = s.find(delimiter)) != std::string::npos) {
                token = s.substr(0, pos);

                /*int* array1, siz;*/
                size_t n = std::count(token.begin(), token.end(), ',');
                //n >> siz;

                array1 = new int[n + 1];

                size_t array_pos = 0;

                std::string delimiter = ",";
                size_t po = 0;
                std::string toke;
                while ((po = token.find(delimiter)) != std::string::npos) {
                    toke = token.substr(0, po);

                    //std::cout << toke << std::endl;

                    // object from the class stringstream 
                    stringstream geek(toke);

                    // The object has the value 12345 and stream 
                    // it to the integer x 
                    int x = 0;
                    geek >> x;

                    // Now the variable x holds the value 12345 
                    //cout << "Value of x : " << x;
                    array1[array_pos++] = x;

                    token.erase(0, po + delimiter.length());

                }
                //std::cout << token << std::endl;

                stringstream geek(token);
                int x = 0;
                geek >> x;
                array1[array_pos++] = x;

                for (int i = 0; i < n + 1; i++)
                    cout << array1[i] << " ";
                cout << endl;
                //delete[]array1;

                s.erase(0, pos + delimiter.length());

            }
            //std::cout << s << std::endl;

            /*int* array2, size;*/
            size_t m = std::count(s.begin(), s.end(), ',');
            //size << m;
            array2 = new int[m + 1];
            size_t array_pos = 0;

            std::string delimite = ",";
            size_t pose = 0;
            std::string stoke;
            while ((pose = s.find(delimite)) != std::string::npos) {
                stoke = s.substr(0, pose);

                //std::cout << stoke << std::endl;
                // object from the class stringstream 
                stringstream geek(stoke);

                // The object has the value 12345 and stream 
                // it to the integer x 
                int x = 0;
                geek >> x;

                // Now the variable x holds the value 12345 
                //cout << "Value of x : " << x;
                array2[array_pos++] = x;

                s.erase(0, pose + delimite.length());
            }
            //std::cout << s << std::endl;

            stringstream geek(s);
            int x = 0;
            geek >> x;
            array2[array_pos++] = x;

            for (int i = 0; i < m + 1; i++)
                cout << array2[i] << " ";
            cout << endl;
            //delete[]array2;

            //N << m;
            output = new int[m];
            arrayAdditionModel.setData(array1, array2, output, m);
            arrayAdditionModel.execute();

        }
        newfile.close(); //close the file object.
    }

    //------------------------steam function end--------------------


    /*for (int k = 0; k < N; k++) {
        inputA[k] = rand() % RANGE_OF_INT_VALUES;
        inputB[k] = rand() % RANGE_OF_INT_VALUES;
    }

    arrayAdditionModel.setData(inputA, inputB, output, N);
    arrayAdditionModel.execute();*/

    /* for(int i=0; i<N; i++)
        cout << output[i] << ", ";
    */
    /*}
    return 0;*/
}
