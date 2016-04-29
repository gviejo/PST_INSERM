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
	// for (int i=0;i<5;i++) std::cout << v[i] << " "; std::cout << std::endl;
	for (int i=0;i<4;i++) {
		tmp[i] = exp(v[i]*b);
		sum+=tmp[i];		
	}		
	// for (int i=0;i<5;i++) std::cout << tmp[i] << " "; std::cout << std::endl;
	for (int i=0;i<4;i++) {
		p[i] = tmp[i]/sum;		
	}
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<4;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
// void sferes_call(double * fit, const char* data_dir, double length_, double noise_, double threshold_)
void sferes_call(double * fit, const int N, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double weight_, double threshold_, double sigma_, double kappa_, double shift_)
{
	///////////////////
	double max_entropy = -log2(0.25);
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0);
	double beta=0.0+beta_*(100.0-0.0);
	double noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;	
	double threshold=0.01+(max_entropy-0.01)*threshold_;
	double sigma=0.0+(20.0-0.0)*sigma_;
	// double sigma = 1.0;
	double weight=0.0+(1.0-0.0)*weight_;	
	double kappa=0.0+(1.0-0.0)*kappa_;
	double shift=0.0+(1.0-0.0)*shift_;

	
	// std::cout << "alpha=" << alpha << " beta =" << beta << " weight=" << weight << " noise=" << noise << " length=" << length << " threshold=" << threshold << " sigma=" << sigma << std::endl;

	int nb_trials = N;
	int n_state = 1;
	int n_action = 4;
	int n_r = 2;	
	int problem; 
	///////////////////
	int sari [N][5];	
	double mean_rt [50][3];	
	double values [N]; // action probabilities according to subject
	double rt [N]; // rt du model	
	double p_a_mb [n_action];	
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

			sari[i][0] = (int)values_[3];
			sari[i][1] = (int)values_[4];
			sari[i][2] = (int)values_[5];
			sari[i][3] = (int)values_[9];
			sari[i][4] = (int)values_[2];
		}
	data_file1.close();	
	}	
	std::ifstream data_file2(file2.c_str());	
	if (data_file2.is_open())
	{
		for (int i=0;i<50;i++) 
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
	double values_mb [n_action];
	double values_mf [n_action];	
	double tmp [n_action][2];	
	double p_a_r [n_action][2];
	double p_r [2];	
	double p_a_final[n_action];
	int n_element = 0;
	int a, r;		
	double Hf = 0.0;
	double weigh = weight;
	for (int m=0;m<n_action;m++) {
		values_mf[m] = 0.0;
		spatial_biases[m] = 1./n_action;
	}	

	for (int i=0;i<nb_trials;i++) 	
	// for (int i=0;i<9;i++) 	
	{				
		if (sari[i][1] != problem) {
			if (sari[i][4]-sari[i-1][4] < 0.0) {
				// START BLOC //
				problem = sari[i][1];
				n_element = 0;			
				weigh = weight;
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
				values_mf[sari[i-1][2]-1] *= (1.0-shift);
				// spatial biases update
				spatial_biases[sari[i][2]-1] += 1.0;
			}
		}
		// START TRIAL //		
		// COMPUTE VALUE
		a = sari[i][2]-1; // a is the action performed by the monkey
		r = sari[i][0];	// r is the amout of reward						

		softmax(p_a_mf, values_mf, beta);
		double Hf = entropy(p_a_mf);

		for (int m=0;m<n_action;m++) {
			p[m][0] = 1./(n_action*n_r); 
			p[m][1] = 1./(n_action*n_r); 
		}					// fill with uniform
		double entrop = max_entropy;			
		for (int n=0;n<n_action;n++){
			p_a_mb[n] = 1./n_action;
			values_mb[n] = 1./n_action;								
		}						
		int nb_inferences = 0;												
		int k = 0;
		while ( entrop > threshold && nb_inferences < n_element) {						

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
			// EVALUATION
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
			entrop = entropy(p_a_mb);
			k+=1;				
		}	
		
		double sum = 0.0;		

		for (int n=0;n<n_action;n++) {
			p_a_final[n] = (1.0-weigh)*p_a_mf[n] + weigh*p_a_mb[n];
			sum+=p_a_final[n];
		}		
		int ind=-1;
		for (int n=0;n<n_action;n++) {
			if (isinf(p_a_final[n])) {
				ind = n;
				break;
			}
		}
		if (ind!=-1) {
			for (int n=0;n<n_action;n++) {
				p_a_final[n] = 0.0000001;
			}
			p_a_final[ind] = 0.9999997;
		} else {
			for (int n=0;n<n_action;n++) p_a_final[n]/=sum;	
		}

		double H = entropy(p_a_final);
		float N = nb_inferences+1.0;
		// if (isnan(H)) H = 0.005;
		values[i] = log(p_a_final[a]);			
		// std::cout << values[i] << std::endl;
		rt[i] =  pow(log2(N), sigma)+H;

		// UPDATE WEIGHT
		double p_wmc;
		double p_rl;
		if (r == 1) {
			p_wmc = p_a_mb[a];
			p_rl = p_a_mf[a];				
		} else {
			p_wmc = 1.0 - p_a_mb[a];
			p_rl = 1.0 - p_a_mf[a];
		}			
		weigh = (p_wmc*weigh)/(p_wmc*weigh+p_rl*(1.0-weigh));			

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
				values_mf[m] += (1.0-kappa)*((1.0/n_action)-values_mf[m]);
			}
		}

	}
	
	
	// ALIGN TO MEDIAN
	alignToMedian(rt, N);	

	// REARRANGE TO REPRESENTATIVE STEPS
	double mean_model [50];
	double sum_tmp [50];

	for (int i=0;i<50;i++) {
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
	for (int i=0;i<50;i++) {
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