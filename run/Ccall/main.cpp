#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
#include <math.h>
#include "qlearning_call.cpp"
// #include "bayesian_call.cpp"
// #include "selection_call.cpp"
// #include "mixture_call.cpp"
// #include "fusion_call.cpp"

using namespace std;



int main () {	


	std::map<char,int> map_monkeys_length_trial;
	map_monkeys_length_trial['g'] = 12701;
	map_monkeys_length_trial['m'] = 34752;
	map_monkeys_length_trial['p'] = 27692;
	map_monkeys_length_trial['r'] = 11634;
	map_monkeys_length_trial['s'] = 13348;


														

	double fit [2] = {0.0, 0.0};
	fit[0] = 0.0 ; fit[1] = 0.0;

	int N =  map_monkeys_length_trial['g'];

	sferes_call(fit, N, "../../data/data_txt/g.txt", 0.0, 0.0657192, 0.772174);
	
	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}