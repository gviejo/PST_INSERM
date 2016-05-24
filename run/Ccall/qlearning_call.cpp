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
void sferes_call(double * fit, int N, const char* data_dir, double alpha_, double beta_, double sigma_, double kappa_, double shift_) {
	
	///////////////////	
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0); //alpha +
	// double gamma=0.0+gamma_*(0.99-0.00); //alpha -
	double beta=0.0+beta_*(100.0-0.0);
	double sigma=0.0+(20.0-0.0)*sigma_;
	double kappa=0.0+(1.0-0.0)*kappa_;
	double shift=0.0+(1.0-0.0)*shift_;
	// double omega=0.0+omega_*(0.999999-0.0);
	// std::cout << alpha << " " << beta << " " << std::endl;
	int nb_trials = N;
	int n_state = 2;
	int n_action = 4;
	int n_r = 2;	
	int problem;
	///////////////////
	int sari [N][5];
	double mean_rt[30][3];		
	double values [N]; // log action probabilities according to monkeys
	double rt [N]; // rt du model	
	double p_a_mf [n_action];
	double spatial_biases [n_action];

	const char* _data_dir = data_dir;
	std::string file1 = _data_dir;
	std::string file2 = _data_dir;
	file1.append(".txt");
	file2.append("_rt_reg.txt");	
	std::ifstream data_file1(file1.c_str());
	string line;
	if (data_file1.is_open())
	{ 
		for (int i=0;i<N;i++) 
		{  
			getline (data_file1,line);			
			stringstream stream(line);
			std::vector<float> values_(
     			(std::istream_iterator<float>(stream)),
     			(std::istream_iterator<float>()));

			sari[i][0] = (int)values_[3]; // reward
			sari[i][1] = (int)values_[4]; // problem
			sari[i][2] = (int)values_[5]; // action
			sari[i][3] = (int)values_[9]; // index
			sari[i][4] = (int)values_[2]; // phase

		}
	data_file1.close();	
	}	
	std::ifstream data_file2(file2.c_str());	
	if (data_file2.is_open())
	{
		for (int i=0;i<30;i++) 
		{  
			getline (data_file2,line);			
			stringstream stream(line);
			std::vector<float> values_(
				(std::istream_iterator<float>(stream)),
				(std::istream_iterator<float>()));
			for (int j=0;j<3;j++) {
				mean_rt[i][j] = values_[j];
			}
		}
	data_file2.close();	
	}		

	double values_mf [n_action];	
	int s, a, r;		
	problem = sari[0][1];
	double Hf = 0.0;
	for (int m=0;m<n_action;m++) {
		values_mf[m] = 0.0;
		spatial_biases[m] = 1./n_action;
	}	

	for (int i=0;i<nb_trials;i++) 	
	// for (int i=0;i<12;i++) 	
	{						
		if (sari[i][1] != problem) {
			if (sari[i][4]-sari[i-1][4] < 0.0) {				
				// START BLOC //
				problem = sari[i][1];				
				
				// RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
				double summ = 0.0;
				for (int m=0;m<n_action;m++) { // normalise spatial bias
					summ+=spatial_biases[m];
				}
				
				for (int m=0;m<n_action;m++) {					
					values_mf[m] = spatial_biases[m]/summ;					
					// std::cout << values_mf[m] << " ";
					
				}
				// std::cout << std::endl;
				// shift bias
				for (int m=0;m<n_action;m++) {
					if (m == sari[i-1][2]-1) {
						values_mf[m] *= (1.0-shift);		
					} else {
						values_mf[m] *= (shift/3.);
					}
				}
				// spatial biases update
				spatial_biases[sari[i][2]-1] += 1.0;
			}
		}		
		// START TRIAL //		
		// COMPUTE VALUE		
		a = sari[i][2]-1;
		// std::cout << "ACTION=" << a << std::endl;
		r = sari[i][0];						
		// COMPUTE VALUE

		softmax(p_a_mf, values_mf, beta);
		double Hf = entropy(p_a_mf);
		
		values[i] = log(p_a_mf[a]);						
		rt[i] =  sigma * Hf;
		// for (int j=0;j<4;j++) {
		// 	std::cout << values_mf[j] << " ";
		// }		
		// std::cout << std::endl;
		// std::cout << rt[i] << std::endl;
		
		// MODEL FREE	
		double reward;
		if (r == 0) {
			reward = -1.0;
		} else if (r == 1) {
			reward = 1.0;
		}
		// std::cout << " R= "<< reward << std::endl;
		// std::cout << values_mf[a] << std::endl;
		double delta = reward - values_mf[a];
		// std::cout << "delta =" << delta << std::endl;
		values_mf[a]+=(alpha*delta);
		// std::cout << "mf2 = ";
		// for (int j=0;j<4;j++) {
		// 	std::cout << values_mf[j] << " ";
		// }
		// std::cout << std::endl;	
		for (int m=0;m<n_action;m++) {
			if (m != a) {				
				values_mf[m] += (1.0-kappa)*(0.0-values_mf[m]);
			}
		}
		// std::cout << "mf3 = ";
		// for (int j=0;j<4;j++) {
		// 	std::cout << values_mf[j] << " ";
		// }
		// std::cout << std::endl;	

	}
	
	// ALIGN TO MEDIAN
	alignToMedian(rt, N);	
	
	// REARRANGE TO REPRESENTATIVE STEPS
	double mean_model [30];
	double sum_tmp [30];

	for (int i=0;i<30;i++) {
		mean_model[i] = 0;
		sum_tmp[i] = 0;
	}

	for (int i=0;i<N;i++) {
		if (sari[i][3] >= 0.0) {
			mean_model[sari[i][3]] += rt[i];
			sum_tmp[sari[i][3]] += 1.0;
		}
		fit[0] += values[i];
	}
	for (int i=0;i<30;i++) {
		// std::cout << mean_model[i] << " ";
		mean_model[i] = mean_model[i]/sum_tmp[i];
		// std::cout << mean_model[i] << std::endl;
		fit[1] -= pow(mean_model[i] - mean_rt[i][1], 2.0);
	}

	if (isnan(fit[0]) || isinf(fit[0]) || isinf(fit[1]) || isnan(fit[1]) || fit[0]<-100000000.0 || fit[1]<-100000000.0) {
		fit[0]=-100000000.0;
		fit[1]=-100000000.0;
		return;
	}
}
