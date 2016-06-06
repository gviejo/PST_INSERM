#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
#include <math.h>
// #include "qlearning_call.cpp"
// #include "bayesian_call.cpp"
// #include "selection_call.cpp"
#include "mixture_call.cpp"
// #include "fusion_call.cpp"

using namespace std;



int main () {	


	std::map<char,int> map_monkeys_length_trial;
	map_monkeys_length_trial['g'] = 11399;
	map_monkeys_length_trial['m'] = 30715;
	map_monkeys_length_trial['p'] = 22347;
	map_monkeys_length_trial['r'] = 8745;
	map_monkeys_length_trial['s'] = 10668;


														

	double fit [2] = {0.0, 0.0};
	fit[0] = 0.0 ; fit[1] = 0.0;

	int N =  map_monkeys_length_trial['s'];

	sferes_call(fit, N, "../../data/data_txt_3_repeat/s", 0, 0.0969144, 0.291051, 0.111525, 1, 0, 0.20948, 1, 0.0907374);

	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}