#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
#include <cmath>


using namespace std;

bool myisnan(float var)
{
    volatile float temp1 = var;
    volatile float temp2 = var;
    return temp1 != temp2;
}

void alignToMedian(float *daArray, int iSize) {    
    float* dpSorted = new float[iSize];
    for (int i = 0; i < iSize; ++i) dpSorted[i] = daArray[i];
    for (int i = iSize - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (dpSorted[j] > dpSorted[j+1]) {
                float dTemp = dpSorted[j];
                dpSorted[j] = dpSorted[j+1];
                dpSorted[j+1] = dTemp;
            }
        }
    }
    float dMedian = dpSorted[(iSize/2)-1]+(dpSorted[iSize/2]-dpSorted[(iSize/2)-1])/2.0;    
    for (int i=0;i<iSize;i++) {daArray[i] = daArray[i]-dMedian;dpSorted[i] = dpSorted[i]-dMedian;}
    float dQ1 = dpSorted[(iSize/4)-1]+((dpSorted[(iSize/4)]-dpSorted[(iSize/4)-1])/2.0);
    float dQ3 = dpSorted[(iSize/4)*3-1]+((dpSorted[(iSize/4)*3+1]-dpSorted[(iSize/4)*3-1])/2.0);
    // std::cout << dpSorted[((iSize/4)*3)-2] << std::endl;
    // std::cout << dpSorted[((iSize/4)*3)-1] << std::endl;
    // // std::cout << dQ3 << std::endl;
    // std::cout << dpSorted[(iSize/4)*3] << std::endl;
    // std::cout << dpSorted[(iSize/4)*3+1] << std::endl;
    delete [] dpSorted;
    for (int i=0;i<iSize;i++) daArray[i] = daArray[i]/(dQ3-dQ1);    
}
void softmax(float *p, float *v, float b) {
	float sum = 0.0;
	float tmp[4];		
	float max_de_sum = -10000.0;
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
float sigmoide(float Hb, float Hf, float n, float i, float t, float g) {		
	float x = 2.0 * -log2(0.25) - Hb - Hf;
	// std::cout << pow((n-i),t) <<  std::endl;
	float tmp = 1.0/(1.0+(pow((n-i),t)*exp(-x*g)));
	// std::cout << " n=" << n << " i=" << i << "Hb = "<< Hb << ", Hf = " << Hf << " x=" << x << " p(A)=" << tmp << " threshold = " << t << " gain = " << g << std::endl;
	// std::cout << tmp << std::endl;
	return tmp;
	// return 1.0/(1.0+((n-i)*t)*exp(-x*g));

}
void fusion(float *p_a, float *mb, float *mf, float beta) {
	float tmp[4];
	int tmp2[4];
	float sum = 0.0;
	float ninf = 0;	
	float mbplusmf[4];
	float max_de_sum = -10000.0;
	//summing mb + mf
	for (int i=0;i<4;i++) {
		mbplusmf[i] = mb[i] + mf[i]; 
		if (mbplusmf[i] > max_de_sum) {
			max_de_sum = mbplusmf[i];
		}
	}
	// std::cout << "tmp=";
	for (int i=0;i<4;i++) {				
		tmp[i] = exp((mbplusmf[i]-max_de_sum)*beta);
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
	// std::cout << "ninf " << ninf << std::endl;
	
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
			// return;
		}
	}
	// std::cout << " p_afinal = ";
	// for (int i=0;i<4;i++) {
	// 	std::cout <<p_a[i] << " ";
	// }
	// std::cout << std::endl;
	return;
}
float entropy(float *p) {
	float tmp = 0.0;
	for (int i=0;i<4;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
float sum_prod(float *a, float *b, int n) {
	float tmp = 0.0;
	for (int i=0;i<n;i++) {
		tmp+=(a[i]*b[i]);
	}
	return tmp;
}
// void sferes_call(float * fit, const char* data_dir, float alpha_, float beta_, float noise_, float length_, float gain_, float threshold_, float gamma_)
void sferes_call(float * fit, const int N, const char* data_dir, float alpha_, float beta_, float noise_, float length_, float gain_, float threshold_, float gamma_, float sigma_, float kappa_, float shift_)
{

	///////////////////
	// parameters
	float alpha=0.0+alpha_*(1.0-0.0);
	float beta=0.0+beta_*(100.0-0.0);
	float noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;
	float gain=0.00001+(10000.0-0.00001)*gain_;
	float threshold=0.0+(20.0-0.0)*threshold_;
	float sigma=0.0+(20.0-0.0)*sigma_;	
	float gamma=0.0+(100.0-0.0)*gamma_;
	float kappa=0.0+(1.0-0.0)*kappa_;
	float shift=0.0+(1.0-0.0)*shift_;

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
	float max_entropy = -log2(0.25);
	///////////////////
	int sari [N][5];	
	float mean_rt [30][3];	
	float values [N]; // action probabilities according to subject
	float rt [N]; // rt du model	
	float p_a_mf [n_action];
	float p_a_mb [n_action];
	float spatial_biases [n_action];

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
	float p_a [length][n_action];
	float p_r_a [length][n_action][n_r];				
	float p [n_action][2];		
	float values_mf [n_action];	
	float values_mb [n_action];
	float tmp [n_action][2];
	float p_a_r [n_action][2];
	float reward = 0.0;
	float p_r [2];
	int n_element = 0;
	int s, a, r;		
	float Hf = max_entropy;
	float Hb = max_entropy;			
	for (int m=0;m<n_action;m++) {
		values_mf[m] = 0.0;
		spatial_biases[m] = 1./n_action;
	}	
	
	for (int i=0;i<nb_trials;i++) 	
	{								
		if (sari[i][4]-sari[i-1][4] < 0.0) {
			// START BLOC //
			problem = sari[i][1];
			n_element = 0;
			
			// RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
			float summ = 0.0;
			for (int m=0;m<n_action;m++) { // normalise spatial bias
				summ+=spatial_biases[m];
			}
			
			for (int m=0;m<n_action;m++) {					
				values_mf[m] = spatial_biases[m]/summ;					
				// std::cout << spatial_biases[m] << " " ;
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
		// START TRIAL //
		// COMPUTE VALUE		
		a = sari[i][2]-1;		
		r = sari[i][0];	
		// QLEARNING CALL
		softmax(p_a_mf, values_mf, gamma);
		float Hf = 0.0; 
		for (int n=0;n<n_action;n++){
			Hf-=p_a_mf[n]*log2(p_a_mf[n]);
		}
		// BAYESIAN CALL
		int nb_inferences = 0;
		float p_decision [n_element+1];
		float p_retrieval [n_element+1];
		float p_ak [n_element+1];
		float reaction [n_element+1];
		float values_net [n_action];
		float p_a_final [n_action];

		// NO SWEEPING
		if ((sari[i][4] == 1) || (i == 0)) {
			for (int m=0;m<n_action;m++) {
				p[m][0] = 1./(n_action*n_r); 
				p[m][1] = 1./(n_action*n_r); 
				values_mb[m] = 1./n_action;
			}					// fill with uniform					
			Hb = max_entropy;			
		} 

		// START
		p_decision[0] = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);		
		p_retrieval[0] = 1.0-p_decision[0];		
		fusion(p_a_final, values_mb, values_mf, beta);		
		p_ak[0] = p_a_final[a];
		reaction[0] = entropy(p_a_final);

		// std::cout << 0 << " p_ak=" << p_ak[0] << " p_decision=" << p_decision[0] << "p_retrieval=" << p_retrieval[0] << std::endl;

		for (int k=0;k<n_element;k++) {
			// INFERENCE				
			float sum = 0.0;
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
			// std::cout << "fusion " << values_mf[a] << " " << values_mb[a] <<  std::endl;
			p_ak[k+1] = p_a_final[a];
			float N = k+2.0;
			reaction[k+1] = pow(log2(N), sigma) + entropy(p_a_final);
		
			// SIGMOIDE
			float pA = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);				

			
			p_decision[k+1] = pA*p_retrieval[k];
			p_retrieval[k+1] = (1.0-pA)*p_retrieval[k];

			// std::cout << k+1 << " p_ak=" << p_ak[k+1] << " p_decision=" << p_decision[k+1] << "p_retrieval=" << p_retrieval[k+1] << std::endl;
		}		

		values[i] = log(sum_prod(p_ak, p_decision, n_element+1));

		float val = sum_prod(p_ak, p_decision, n_element+1);						
			
		rt[i] = sum_prod(reaction, p_decision, n_element+1);									
		// std::cout << "rt " << rt[i] << std::endl;

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
		float delta = reward - values_mf[a];
		values_mf[a]+=(alpha*delta);

		for (int m=0;m<n_action;m++) {
			if (m != a) {				
				values_mf[m] += (1.0-kappa)*(0.0-values_mf[m]);
			}
		}
		// SWEEPING
		if (sari[i][4] == 0) {
			for (int m=0;m<n_action;m++) {
				p[m][0] = 1./(n_action*n_r); 
				p[m][1] = 1./(n_action*n_r); 
				values_mb[m] = 1./n_action;
			}					// fill with uniform					
			float Hb = max_entropy;						
			for (int k=0;k<n_element;k++) {
				// INFERENCE				
				float sum = 0.0;
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
			float sum = 0.0;								
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
		}
	}
	
	// ALIGN TO MEDIAN
	alignToMedian(rt, N);	
	
	// REARRANGE TO REPRESENTATIVE STEPS
	float mean_model [30];
	float sum_tmp [30];
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

	// std::cout << fit[0] << " " << fit[1] << std::endl;

	if (std::isnan(fit[0]) || std::isinf(fit[0]) || std::isinf(fit[1]) || std::isnan(fit[1])) {
		fit[0]=-1e+10;
		fit[1]=-1e+10;
		return;
	}
}
