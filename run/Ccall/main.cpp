#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
// #include "qlearning_call.cpp"
// #include "bayesian_call.cpp"
// #include "selection_call.cpp"
// #include "mixture_call.cpp"
// #include "fusion_call.cpp"

using namespace std;



int main () {	

	double fit [2] = {0.0, 0.0};
	int sari[100][8];

	int length = 0;
	
	ifstream is;
	is.open("../../data/data_bin/s.bin", ios::binary | ios::in);
	is.seekg(0, ios::end);
	length = is.tellg();
	is.seekg(0, ios::beg);

	std::cout << length << std::endl;
	std::cout << length / 16 << std::endl;
	
	// buffer = new unsigned int[length * sizeof(char) / sizeof(unsigned int)];	
	// is.read((char*)buffer, length);
	// is.close();
	
	// cout.write((char*)buffer, length);

	// std::cout << buffer << std::endl;

	// fit[0] = 0.0; fit[1] = 0.0;
 //  	sferes_call(fit, N, "data_meg/S3/", 0.000265578, 0.545996, 0, 0.899626, 0, 0.00186337, 0, 0.00347151);
 //  	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}