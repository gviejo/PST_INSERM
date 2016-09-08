#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
// #include <math.h>
// #include "qlearning_call.cpp"
// #include "bayesian_call.cpp"
// #include "selection_call.cpp"
// #include "mixture_call.cpp"
// #include "fusion_call.cpp"
// #include "fhebbian_call.cpp"
// #include "metaf_call.cpp"
#include "sweeping_call.cpp"


using namespace std;



int main () {	


	std::map<char,int> map_monkeys_length_trial;
	map_monkeys_length_trial['g'] = 11399;
	map_monkeys_length_trial['m'] = 30715;
	map_monkeys_length_trial['p'] = 22347;
	map_monkeys_length_trial['r'] = 8745;
	map_monkeys_length_trial['s'] = 10668;



	float fit [2] = {0.0, 0.0};
	fit[0] = 0.0 ; fit[1] = 0.0;

	int N =  map_monkeys_length_trial['p'];

	sferes_call(fit, N, "../../data/data_txt_3_repeat/p", 0, 0.227429, 1, 0.203377, 0.00535952, 0.3135, 0.239221, 0.371858, 1, 0.278448);

	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}

