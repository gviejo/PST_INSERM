#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
#include <cmath>
#include <iomanip>

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
	double max_de_sum = -10000.0;
	//summing mb + mf
	for (int i=0;i<4;i++) {
		if (v[i] > max_de_sum) {
			max_de_sum = v[i];
		}
	}

	for (int i=0;i<4;i++) {
		tmp[i] = exp((v[i]-max_de_sum)*b);
		sum+=tmp[i];		
	}			

	for (int i=0;i<4;i++) {
		p[i] = tmp[i]/sum;		
	}

	for (int i=0;i<4;i++) {
		if (p[i] == 0) {
			sum = 0.0;
			for (int i=0;i<4;i++) {
				p[i]+=1e-8;
				sum+=p[i];
			}
			for (int i=0;i<4;i++) {
				p[i]/=sum;
			}			
		}
	}
}
double sigmoide(double Hb, double Hf, double n, double i, double t, double g) {		
	double x = 2.0 * -log2(0.25) - Hb - Hf;
	// std::cout << pow((n-i),t) <<  std::endl;
	double tmp = 1.0/(1.0+(pow((n-i),t)*exp(-x*g)));
	// std::cout << " n=" << n << " i=" << i << "Hb = "<< Hb << ", Hf = " << Hf << " x=" << x << " p(A)=" << tmp << " threshold = " << t << " gain = " << g << std::endl;
	// std::cout << tmp << " ";
	return tmp;
	// return 1.0/(1.0+((n-i)*t)*exp(-x*g));

}
void fusion(double *p_a, double *mb, double *mf, double beta) {
	double tmp[4];
	int tmp2[4];
	double sum = 0.0;
	double ninf = 0;	
	double mbplusmf[4];
	double max_de_sum = -10000.0;
	//summing mb + mf
	for (int i=0;i<4;i++) {
		mbplusmf[i] = mb[i] + mf[i]; 
		if (mbplusmf[i] > max_de_sum) {
			max_de_sum = mbplusmf[i];
		}
	}	
	for (int i=0;i<4;i++) {				
		tmp[i] = exp((mbplusmf[i]-max_de_sum)*beta);
		sum+=tmp[i];
	}
	for (int i=0;i<4;i++) {				
		p_a[i] = tmp[i]/sum;		
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
		}
	}
	return;
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
void sferes_call(double * fit, const int N, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double gain_, double threshold_, double gamma_, double sigma_, double kappa_, double shift_)
{

	// /////////////////
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0);
	double beta=0.0+beta_*(100.0-0.0);
	double noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;
	double gain=0.00001+(10000.0-0.00001)*gain_;
	double threshold=0.0+(20.0-0.0)*threshold_;
	double sigma=0.0+(20.0-0.0)*sigma_;	
	double gamma=0.0+(100.0-0.0)*gamma_;
	double kappa=0.0+(1.0-0.0)*kappa_;
	double shift=0.0+(0.999999-0.0)*shift_;

	// std::cout << "alpha : " << alpha << std::endl;
	// std::cout << "beta : " << beta << std::endl;
	// std::cout << "noise : " << noise << std::endl;
	// std::cout << "length : " << length << std::endl;
	// std::cout << "gain : " << gain << std::endl;
	// std::cout << "threshold : " << threshold << std::endl;
	// std::cout << "sigma : " << sigma << std::endl;
	// std::cout << "gamma : " << gamma << std::endl;
	// std::cout << "kappa : " << kappa << std::endl;
	// std::cout << "shift : " << shift << std::endl;

	


	int nb_trials = N;
	int n_state = 1;
	int n_action = 4;
	int n_r = 2;
	int problem;
	double max_entropy = -log2(0.25);
	///////////////////
	int sari [N][5];	
	double mean_rt [30][3];	
	double values [N]; // action probabilities according to subject
	double rt [N]; // rt du model	
	double p_a_mf [n_action];
	double p_a_mb [n_action];
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
			std::vector<double> values_(
     			(std::istream_iterator<double>(stream)),
     			(std::istream_iterator<double>()));

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
			std::vector<double> values_(
				(std::istream_iterator<double>(stream)),
				(std::istream_iterator<double>()));
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
	double values_mf [n_action];	
	double values_mb [n_action];
	double tmp [n_action][2];
	double p_a_r [n_action][2];
	double reward = 0.0;
	double p_r [2];
	int n_element = 0;
	int s, a, r;		
	double Hf = max_entropy;
	double Hb = max_entropy;			
	for (int m=0;m<n_action;m++) {
		values_mf[m] = 0.0;
		spatial_biases[m] = 1./n_action;
	}	
	
	for (int i=0;i<nb_trials;i++) 	
	// for (int i=0;i<22;i++) 
	{				
		if (sari[i][4]-sari[i-1][4] < 0.0) {
			// START BLOC //
			problem = sari[i][1];
			n_element = 0;
		}		
		// START TRIAL //
		// COMPUTE VALUE		
		a = sari[i][2]-1;		
		r = sari[i][0];	
		// QLEARNING CALL
		softmax(p_a_mf, values_mf, gamma);
		double Hf = 0.0; 
		for (int n=0;n<n_action;n++){			
			Hf-=p_a_mf[n]*log2(p_a_mf[n]);
		}

		// BAYESIAN CALL
		int nb_inferences = 0;
		double p_decision [n_element+1];
		double p_retrieval [n_element+1];
		double p_ak [n_element+1];
		double reaction [n_element+1];
		double values_net [n_action];
		double p_a_final [n_action];

		// NO SWEEPING
		if ((sari[i][4] == 1) || (i == 0) || (sari[i][4]-sari[i-1][4] < 0.0)) {
			for (int m=0;m<n_action;m++) {
				p[m][0] = 1./(n_action*n_r); 
				p[m][1] = 1./(n_action*n_r); 
				values_mb[m] = 1./n_action;
				p_a_mb[m] = 1./n_action;
			}					// fill with uniform					
			Hb = max_entropy;		
			// std::cout << " NO SWEEPING" << std::endl;	
		} 

		// for (int x =0;x<4;x++) {
		// 	std::cout << values_mb[x] << " ";
		// }
		// std::cout<<std::endl;


		// START
		// std::cout << "Hb " << Hb << std::endl;
		// std::cout << "Hf " << Hf << std::endl;
		p_decision[0] = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);		
		// std::cout << p_decision[0] << std::endl;
		p_retrieval[0] = 1.0-p_decision[0];		
		fusion(p_a_final, values_mb, values_mf, beta);		
		p_ak[0] = p_a_final[a];
		reaction[0] = entropy(p_a_final);

		// std::cout << 0 << " p_ak=" << p_ak[0] << " p_decision=" << p_decision[0] << "p_retrieval=" << p_retrieval[0] << std::endl;
		// std::cout << "mf= ";
		// for (int x=0;x<4;x++) {
		// 	std::cout << values_mf[x] << " ";
		// }
		// std::cout << std::endl;

		// std::cout << "mb = ";
		// for (int x =0;x<4;x++) {
		// 	std::cout << values_mb[x] << " ";
		// }
		// std::cout<<std::endl;

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
			fusion(p_a_final, values_mb, values_mf, beta);
			
			// std::cout << "mb = ";
			// for (int x =0;x<4;x++) {
			// 	std::cout << values_mb[x] << " ";
			// }
			// std::cout<<std::endl;

			p_ak[k+1] = p_a_final[a];
			double N = k+2.0;
			reaction[k+1] = pow(log2(N), sigma) + entropy(p_a_final);

		
			// SIGMOIDE
			double pA = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);				

			// std::cout << pA << std::endl;

			p_decision[k+1] = pA*p_retrieval[k];
			p_retrieval[k+1] = (1.0-pA)*p_retrieval[k];


			// std::cout << k+1 << " p_ak=" << p_ak[k+1] << " p_decision=" << p_decision[k+1] << "p_retrieval=" << p_retrieval[k+1] << std::endl;
		}		

		values[i] = log(sum_prod(p_ak, p_decision, n_element+1));

		// for (int k=0;k<n_element+1;k++) {
		// 	std::cout << p_decision[k] << " ";
		// }
		// std::cout << std::endl;
		// for (int k=0;k<n_element+1;k++) {
		// 	std::cout << p_ak[k] << " ";
		// }
		// std::cout << std::endl;

		// std::cout << values[i] << std::endl;
		// std::cout << std::endl;
		double val = sum_prod(p_ak, p_decision, n_element+1);						
			
		rt[i] = sum_prod(reaction, p_decision, n_element+1);									
		// std::cout << rt[i] << " ";

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

		// // MODEL FREE			
		if (r == 0) {
			reward = -1.0;
		} else if (r == 1) {
			reward = 1.0;
		}
		
		double max_next = 0;
		for (int m=0;m<n_action;m++) {
			if (values_mf[m]>max_next) {
				max_next = values_mf[m];
			}
		}
		double delta = reward + shift*max_next - values_mf[a];
		values_mf[a]+=(alpha*delta);

		for (int m=0;m<n_action;m++) {
			if (m != a) {				
				values_mf[m] += (1.0-kappa)*(0.0-values_mf[m]);
			}
		}
		// SWEEPING
		if ((sari[i][4] == 0) && (r==0)) {
			for (int m=0;m<n_action;m++) {
				p[m][0] = 1./(n_action*n_r); 
				p[m][1] = 1./(n_action*n_r); 
				values_mb[m] = 1./n_action;
			}					// fill with uniform					
			Hb = max_entropy;	
			nb_inferences = 0;					
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
			}
			nb_inferences+=1;
			// // EVALUATION
			double sum = 0.0;								
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
			// std::cout << " INSIDE SWEE " << Hb << std::endl;
			// std::cout << "pmb apres sweep = ";
			// for (int x =0;x<4;x++) {
			// 	std::cout << values_mb[x] << " ";
			// }
			// std::cout<<std::endl;

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
		// std::cout << i << " " << mean_model[i] << " " << std::endl;
		fit[1] -= pow(mean_model[i] - mean_rt[i][1], 2.0);
	}		

	if (std::isnan(fit[0]) || std::isinf(fit[0]) || std::isinf(fit[1]) || std::isnan(fit[1]) || fit[0] < -1e+30 || fit[1] < -1e+30) {
		fit[0]=-1e+15;
		fit[1]=-1e+15;
		return;
	}
}
