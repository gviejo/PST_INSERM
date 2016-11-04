#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
// #include "call_5_3.cpp"
#include "call_6_1.cpp"

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

	int N =  map_monkeys_length_trial['p'];

	sferes_call(fit, N, "../../../data/data_txt_3_repeat/p", 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2);

	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}