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
    for (int i=0;i<iSize;i++) daArray[i] = daArray[i]/(dQ3-dQ1);    
}
void softmax(double *p, double *v, double b) {
	double sum = 0.0;
	double tmp[4];	
	for (int i=0;i<4;i++) {
		tmp[i] = exp(v[i]*b);
		sum+=tmp[i];		
	}			
	for (int i=0;i<4;i++) {
		p[i] = tmp[i]/sum;		
	}
	
	// for (int i=0;i<5;i++) {
	// 	if (p[i] == 0) {
	// 		sum = 0.0;
	// 		for (int i=0;i<5;i++) {
	// 			p[i]+=1e-4;
	// 			sum+=p[i];
	// 		}
	// 		for (int i=0;i<5;i++) {
	// 			p[i]/=sum;
	// 		}
	// 		return;
	// 	}
	// }	
}
double sigmoide(double Hb, double Hf, double n, double i, double t, double g) {		
	double x = 2.0 * -log2(0.25) - Hb - Hf;
	// std::cout << pow((n-i),t) <<  std::endl;
	double tmp = 1.0/(1.0+(pow((n-i),t)*exp(-x*g)));
	// std::cout << " n=" << n << " i=" << i << "Hb = "<< Hb << ", Hf = " << Hf << " x=" << x << " p(A)=" << tmp << " threshold = " << t << " gain = " << g << std::endl;
	// std::cout << tmp << std::endl;
	return tmp;
	// return 1.0/(1.0+((n-i)*t)*exp(-x*g));

}
void fusion(double *p_a, double *mb, double *mf, double beta) {
	double tmp[4];
	int tmp2[4];
	double sum = 0.0;
	double ninf = 0;	
	// std::cout << "fusion " << mf[0] << " " << mb[0] << std::endl;
	// std::cout << "tmp=";
	for (int i=0;i<4;i++) {				
		tmp[i] = exp((mb[i]+mf[i])*beta);
		// std::cout << tmp[i] << " ";
		if (isinf(tmp[i])) {
			tmp2[i] = 1;
			ninf += 1.0;
		} else {
			tmp2[i] = 0;
		}
		sum+=tmp[i];
	}
	// std::cout << std::endl;

	
	if (ninf > 0.0) {
		for (int i=0;i<4;i++) {
			if (tmp2[i] == 1) {				
				p_a[i] = (1.0 - 0.0000001 * (4.0 - ninf))/ninf;
			}
			else {
				p_a[i] = 0.0000001;
			}
		}
	}
	else {
		for (int i=0;i<4;i++) {				
			p_a[i] = tmp[i]/sum;		
		}		
	}		
	for (int i=0;i<4;i++) {
		if (p_a[i] == 0) {
			sum = 0.0;
			for (int i=0;i<4;i++) {
				p_a[i]+=1e-8;
				sum+=p_a[i];
			}
			for (int i=0;i<4;i++) {
				p_a[i]/=sum;
			}
			return;
		}
	}
	// std::cout << " p_afinal = ";
	// for (int i=0;i<4;i++) {
	// 	std::cout <<p_a[i] << " ";
	// }
	// std::cout << std::endl;
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<4;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
double sum_prod(double *a, double *b, int n) {
	double tmp = 0.0;
	for (int i=0;i<n;i++) {
		tmp+=(a[i]*b[i]);
	}
	return tmp;
}
// void sferes_call(double * fit, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double gain_, double threshold_, double gamma_)
void sferes_call(double * fit, const int N, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double gain_, double threshold_, double gamma_, double sigma_)
{

	///////////////////
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0);	
	double beta=0.0+beta_*(100.0-0.0);
	double noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;
	double gain=0.00001+(10000.0-0.00001)*gain_;
	double threshold=0.0+(20.0-0.0)*threshold_;
	double sigma=0.0+(20.0-0.0)*sigma_;	
	double gamma=0.0+(100.0-0.0)*gamma_;


	int nb_trials = N;
	int n_state = 2;
	int n_action = 4;
	int n_r = 2;
	int problem;
	double max_entropy = -log2(0.25);
	///////////////////
	int sari [N][5];	
	double mean_rt [30][3];	
	double values [N]; // action probabilities according to subject
	double rt [N]; // rt du model	
	double p_a_heb [n_action];
	double p_a_mb [n_action];

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
	problem = sari[0][1];
	double p_a [length][n_action];
	double p_r_a [length][n_action][n_r];				
	double p [n_action][2];		
	double values_heb [n_action*n_state][n_action];	
	double values_mb [n_action];
	double tmp [n_action][2];
	double p_a_r [n_action][2];
	double reward = 0.0;
	double p_r [2];
	int n_element = 0;
	int s, a, r;		
	double Hf = 0.0;
	for (int m=0;m<n_action*n_state;m++) {
		for (int n=0;n<n_action;n++) {
			values_heb[m][n] = 0.0;	
		}			
	}	
	int last_action = 0;

	for (int i=0;i<nb_trials;i++) 
	// for (int i=0;i<21;i++) 
	{						
		if (sari[i][1] != problem) {
			if (sari[i][4]-sari[i-1][4] < 0.0) {
				// START BLOC //
				problem = sari[i][1];
				n_element = 0;				
			}
		}
		// START TRIAL //
		// COMPUTE VALUE		
		a = sari[i][2]-1;		
		r = sari[i][0];
		s = sari[i][4];						
		double Hb = max_entropy;		
		for (int m=0;m<n_action;m++) {
			p[m][0] = 1./(n_action*n_r); 
			p[m][1] = 1./(n_action*n_r); 
		}					// fill with uniform
		softmax(p_a_heb, values_heb[last_action+s*n_action], gamma);
	
		double Hf = 0.0; 
		for (int n=0;n<n_action;n++){
			values_mb[n] = 1./n_action;
			Hf-=p_a_heb[n]*log2(p_a_heb[n]);
		}

		int nb_inferences = 0;
		double p_decision [n_element+1];
		double p_retrieval [n_element+1];
		double p_ak [n_element+1];

		double reaction [n_element+1];
		double values_net [n_action];
		double p_a_final [n_action];
		p_decision[0] = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);
		
		p_retrieval[0] = 1.0-p_decision[0];
		
		fusion(p_a_final, values_mb, values_heb[last_action+s*n_action], beta);
		
		p_ak[0] = p_a_final[a];
		// reaction[0] = entropy(p_a_final);
		reaction[0] = log2(1./n_action) + sigma*Hf;

		for (int k=0;k<n_element;k++) {
			// INFERENCE				
			double sum = 0.0;
			for (int m=0;m<n_action;m++) {
				for (int o=0;o<n_r;o++) {
					p[m][o] += (p_a[k][m] * p_r_a[k][m][o]);
					sum+=p[m][o];
				}
			}
							
			for (int m=0;m<n_action;m++) {
				for (int o=0;o<n_r;o++) {
					tmp[m][o] = (p[m][o]/sum);
				}
			}

			nb_inferences+=1;
			// // EVALUATION
			sum = 0.0;								
			for (int o=0;o<n_r;o++) {
				p_r[o] = 0.0;						
			}
			for (int m=0;m<n_action;m++) {
				for (int o=0;o<n_r;o++) {						
					p_r[o]+=tmp[m][o];						
				}
			}
			sum = 0.0;
			for (int m=0;m<n_action;m++) {
				for (int o=0;o<n_r;o++) {
					p_a_r[m][o] = tmp[m][o]/p_r[o];
				}
				values_mb[m] = p_a_r[m][1]/p_a_r[m][0];
				sum+=values_mb[m];
			}
			for(int m=0;m<n_action;m++) {
				p_a_mb[m] = values_mb[m]/sum;
			}			
			Hb = entropy(p_a_mb);
			// FUSION
			fusion(p_a_final, values_mb, values_heb[last_action+s*n_action], beta);
			
			p_ak[k+1] = p_a_final[a];
			double N = k+2.0;
			// reaction[k+1] = pow(log2(N), sigma) + entropy(p_a_final);
			reaction[k+1] = Hb + sigma * Hf;
		
			// SIGMOIDE
			double pA = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);				

			
			p_decision[k+1] = pA*p_retrieval[k];
			p_retrieval[k+1] = (1.0-pA)*p_retrieval[k];

		}		


		values[i] = log(sum_prod(p_ak, p_decision, n_element+1));

		double val = sum_prod(p_ak, p_decision, n_element+1);						
		
		rt[i] = sum_prod(reaction, p_decision, n_element+1);			
		
		
		// UPDATE MEMORY 						
		for (int k=length-1;k>0;k--) {						
			for (int m=0;m<n_action;m++) {
				p_a[k][m] = p_a[k-1][m]*(1.0-noise)+noise*(1.0/n_action);
				for (int o=0;o<n_r;o++) {
					p_r_a[k][m][o] = p_r_a[k-1][m][o]*(1.0-noise)+noise*0.5;				
				}
			}
		}

		if (n_element < length) n_element+=1;

		for (int m=0;m<n_action;m++) {
			p_a[0][m] = 1./n_action;
			for (int o=0;o<n_r;o++) {
				p_r_a[0][m][o] = 0.5;
			}
		}
		
		for (int m=0;m<n_action;m++) {
			p_a[0][m] = 0.0;
		}		
		p_a[0][a] = 1.0;		
		p_r_a[0][a][(r-1)*(r-1)] = 0.0;
		
		p_r_a[0][a][r] = 1.0;

		// HEBBIAN LEARNING
		values_heb[last_action+s*n_action][a] += alpha * (1.0 - values_heb[last_action+s*n_action][a]);
		for (int m=0;m<n_action;m++) {
			if (m != a) {
				values_heb[last_action+s*n_action][a] += alpha * (0.0 - values_heb[last_action+s*n_action][m]);						
			}
		}


		

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
		mean_model[i] = mean_model[i]/sum_tmp[i];
		// std::cout << mean_model[i] << " " << mean_rt[i][1] << std::endl;
		// if (isnan(mean_model[i])) {std::cout << i << " " << mean_model[i] << std::endl;}
		fit[1] -= pow(mean_model[i] - mean_rt[i][1], 2.0);
	}	
	// if (isnan(fit[0]) || isinf(fit[0]) || isinf(fit[1]) || isnan(fit[1]) || fit[0]<-100000000.0 || fit[1]<-100000000.0) {
	// 	fit[0]=-100000000.0;
	// 	fit[1]=-100000000.0;
	// 	return;
	// }
	// std::cout << fit[0] << " " << fit[1] << std::endl;

	if (isnan(fit[0]) || isinf(fit[0]) || isinf(fit[1]) || isnan(fit[1])) {
		fit[0]=-1e+15;
		fit[1]=-1e+15;
		return;
	}
}
