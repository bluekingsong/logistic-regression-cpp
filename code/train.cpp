#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include "logistic_regression.h"
using namespace std;

int main()
{
	LogisticRegression LR("conf/overall.conf");
	time_t t=time(0);
	cout<<"start process:"<<asctime(localtime(&t))<<endl;
//	LR.read_dataset("data/train.feature.filter","conf/overall.conf");
//	LR.init_data_buffer("data/train.feature.filter");
	LR.init_data_buffer("data/train.feature.lr_filter");
	string parameters;
	int status=LR.optimize(string("conf/lr_parameter"),parameters);
	cout<<"LBFGS status:"<<status<<endl;
	t=time(0);
	cout<<"end process:"<<asctime(localtime(&t))<<endl;
	ofstream fout("conf/lr_parameter.conf");
	fout<<parameters;
	fout.close();
	return 0;
}
