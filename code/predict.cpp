#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include "logistic_regression.h"
using namespace std;

int main(int argc,char **argv)
{
	if(argc!=4)
	{
		cout<<"usage:"<<argv[0]<<" parameter_file test_file output"<<endl;
		return -1;
	}
	char *parameter_file=argv[1];
	char *test_file=argv[2];
	char *output=argv[3];
	LogisticRegression LR("conf/overall.conf");
	time_t t=time(0);
	cout<<"start process:"<<asctime(localtime(&t))<<endl;
	double NLL=LR.predict(parameter_file,test_file,output);
	cout<<"NLL for ("<<test_file<<") is "<<NLL<<endl;
	t=time(0);
	cout<<"end process:"<<asctime(localtime(&t))<<endl;
	return 0;
}
