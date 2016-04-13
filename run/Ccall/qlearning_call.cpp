#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>

using namespace std;

void alignToMedian(double *daArray, int iSize) {    
    double* dpSorted = new double[iSize];
    for (int i = 0; i < iSize; ++i) dpSorted[i] = daArray[i];
    for (int i = iSize - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (dpSorted[j] > dpSorted[j+1]) {
                double dTemp = dpSorted[j];
                dpSorted[j] = dpSorted[j+1];
                dpSorted[j+1] = dTemp;
            }
        }
    }
    double dMedian = dpSorted[(iSize/2)-1]+(dpSorted[iSize/2]-dpSorted[(iSize/2)-1])/2.0;    
    for (int i=0;i<iSize;i++) {daArray[i] = daArray[i]-dMedian;dpSorted[i] = dpSorted[i]-dMedian;}
    double dQ1 = dpSorted[(iSize/4)-1]+((dpSorted[(iSize/4)]-dpSorted[(iSize/4)-1])/2.0);
    double dQ3 = dpSorted[(iSize/4)*3-1]+((dpSorted[(iSize/4)*3+1]-dpSorted[(iSize/4)*3-1])/2.0);
    // std::cout << dpSorted[((iSize/4)*3)-2] << std::endl;
    // std::cout << dpSorted[((iSize/4)*3)-1] << std::endl;
    // // std::cout << dQ3 << std::endl;
    // std::cout << dpSorted[(iSize/4)*3] << std::endl;
    // std::cout << dpSorted[(iSize/4)*3+1] << std::endl;
    delete [] dpSorted;
    for (int i=0;i<iSize;i++) {
    	daArray[i] = daArray[i]/(dQ3-dQ1);    	
    }    
}
void softmax(double *p, double *v, double b) {
	double sum = 0.0;
	double tmp[4];
	
	for (int i=0;i<4;i++) {
		tmp[i] = exp(v[i]*b);
		sum+=tmp[i];		
	}		
	// for (int i=0;i<4;i++) {
	// 	if (isinf(tmp[i])) {
	// 		for (int j=0;j<4;j++) {
	// 			p[j] = 0.0000001;
	// 		}			
	// 		p[i] = 0.9999996;
	// 		return ;
	// 	}	
	// }	
	
	for (int i=0;i<4;i++) {
		p[i] = tmp[i]/sum;		
	}
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<4;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
// void sferes_call(double * fit, const char* data_dir, double alpha_, double beta_)
void sferes_call(double * fit, int N, const char* data_dir, double alpha_, double beta_, double sigma_=1.0) {
	
	///////////////////	
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0); //alpha +
	// double gamma=0.0+gamma_*(0.99-0.00); //alpha -
	double beta=0.0+beta_*(100.0-0.0);
	double sigma=0.0+(20.0-0.0)*sigma_;
	// double omega=0.0+omega_*(0.999999-0.0);
	// std::cout << alpha << " " << beta << " " << std::endl;
	int nb_trials = N;
	int n_state = 2;
	int n_action = 4;
	int n_r = 2;	
	///////////////////
	int sari [N][8];	
	double monkeys_rt_centered [N];	
	double values [N]; // log action probabilities according to monkeys
	double rt [N]; // rt du model	
	double p_a_mf [n_action];

	std::ifstream data_file(data_dir);
	string line;
	if (data_file.is_open())
	{ 
		for (int i=0;i<N;i++) 
		{  
			getline (data_file,line);			
			stringstream stream(line);
			std::vector<float> values_(
     			(std::istream_iterator<float>(stream)),
     			(std::istream_iterator<float>()));
			for (int j=0;j<8;j++)
			{
				sari[i][j] = (int)values_[j];				
			}			
			monkeys_rt_centered[i] = values_[8];
		}
	data_file.close();	
	}

	// double cumsum = 0.0;	

	for (int i=0;i<1;i++)	
	{		
		// START BLOC //
		double values_mf [n_action];	
		int s, a, r;		
		double Hf = 0.0;		
		for (int m=0;m<n_action;m++) {
				values_mf[m] = 0.0;			
		}		
		// START TRIAL //
		// for (int j=0;j<6;j++) 				
		for (int j=0;j<nb_trials;j++) 				
		{							
			// COMPUTE VALUE
			s = sari[j][4]-1; // s is the solution to the problem
			a = sari[j][5]-1; // a is the action performed by the monkey
			r = sari[j][3];	// r is the amout of reward						
			softmax(p_a_mf, values_mf, beta);
			double Hf = entropy(p_a_mf);
			
			values[j] = log(p_a_mf[a]);						
			rt[j] =  Hf;
			
			// MODEL FREE	
			double reward;
			if (r == 0) {
				reward = -1.0;
			} else if (r == 1) {
				reward = 1.0;
			}
			double delta = reward - values_mf[a];
			values_mf[a]+=(alpha*delta);
		}
	}
	
	// ALIGN TO MEDIAN
	alignToMedian(rt, N);	
	
	

	double error;
	for (int i=0;i<N;i++) {
		fit[0] += values[i];	
		fit[1] -= pow(monkeys_rt_centered[i] - rt[i], 2.0);
		// cumsum += pow(monkeys_rt_centered[i] - rt[i], 2.0);
		// std::cout << cumsum << std::endl;
	}

	if (isnan(fit[0]) || isinf(fit[0]) || isinf(fit[1]) || isnan(fit[1]) || fit[0]<-100000000.0 || fit[1]<-100000000.0) {
		fit[0]=-100000000.0;
		fit[1]=-100000000.0;
		return;
	}
	// else {
	// 	fit[0]+=2000.0;
	// 	fit[1]+=500.0;
	// 	return ;
	// }
}