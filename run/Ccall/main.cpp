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

	sferes_call(fit, 2, "../../data/data_txt_3_repeat/p",0.348326,0.863463,0.19552,0.790971,0.0828114,0.164964,0.114382,0.4634,0.62559,0.440426);

	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}