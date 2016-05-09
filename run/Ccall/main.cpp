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
	map_monkeys_length_trial['g'] = 11399;
	map_monkeys_length_trial['m'] = 30715;
	map_monkeys_length_trial['p'] = 22347;
	map_monkeys_length_trial['r'] = 8745;
	map_monkeys_length_trial['s'] = 10668;


														

	double fit [2] = {0.0, 0.0};
	fit[0] = 0.0 ; fit[1] = 0.0;

	int N =  map_monkeys_length_trial['g'];

	// sferes_call(fit, N, "../../data/data_txt/s", 0.671374, 0.12509, 0.170704, 0.551698, 0.722591, 0.138297, 0.271538, 0.659637, 0.539239, 0.686598 );
	// sferes_call(fit, N, "../../data/data_txt_3_repeat/g", 0.31487,0.362754,0.793149,0.889519,0.0241067,0.397067,0.745926,0.258521,0.203302);
	sferes_call(fit, N, "../../data/data_txt_3_repeat/g", 0.31487,0.362754,0.793149,0.889519,0.0241067);
	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}