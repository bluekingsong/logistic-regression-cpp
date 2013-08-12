#ifndef __LR_H__
#define __LR_H__
#include <set>
#include <map>
#include <vector>
#include <string>
#include <sstream>
//#define LBFGS_FLOAT 32
#include <lbfgs.h>
#include "dummy_convertor.h"
#include "common_functions.h"
using namespace std;
class LogisticRegression
{
  private:
	DummyConvertor dummyConvertor;  // assistant tool for dummy variables
	// train dataset
	//vector<set<string> > dataset;
	//vector<int> labels;
	string data_buffer; // buffer for train file
//	istringstream data_stream;
	istringstream *p_data_stream; // stringstream for train file. * discard, because it copy data_buffer,double memory.
	unsigned int cursor; // buffer cursor for read
	lbfgsfloatval_t *parameters; // the parameters of logistic regression model we want to optimize
	map<string,string> conf_dict; // configures
	int lbfgs_iterations; // current lbfgs iterations
	int save_intermediate_peroid; // to save inter-mediate parameters, we set a peroid.
	string intermediate_file_prefix; // the prefix(include path) for intermediate result file.
	///////// method
	static void init_parameters(lbfgsfloatval_t *parameter, unsigned int n); // init
	static int convert_record(const vector<string>& record,set<string>& result,const vector<string>& field_vec); //assist
	static lbfgsfloatval_t _evalute(void *instance,const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,const lbfgsfloatval_t step)
	{
		return reinterpret_cast<LogisticRegression*>(instance)->evalute(w,g,n,step);
	}
	lbfgsfloatval_t evalute(const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,const lbfgsfloatval_t step); // evalute goal function and its gradients
  public:
	LogisticRegression(const char *configure);
	~LogisticRegression();
	int read_dataset(const char* filename,const char* configure); // discard, use huge memory when data file is large
	int init_data_buffer(const char* train_file); // read disk file to memory
	int prepare_read(); // init for read from data_buffer
	int finish_read(); // no use
//	istream& get_line(string &line);
	bool get_line(string &line); // get line from data_buffer
	int get_parameter_values(string &result)const;
	int optimize(const string& intermediate_file_prefix,string& final_results);
	int load_parameter_from_file(const char *filename);
	double predict(const vector<string>& record,const lbfgsfloatval_t *w)const;
	double predict(const char *parameter_file,const char *test_file,const char *output);
};

#endif
