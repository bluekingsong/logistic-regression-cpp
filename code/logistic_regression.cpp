#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include "logistic_regression.h"
using namespace std;

LogisticRegression::LogisticRegression(const char* configure)
{
	int status=CommonTool::load_configure(configure,this->conf_dict);
	if(status<0)
	{
		cout<<"load configure file:"<<configure<<" faild."<<endl;
		return;
	}
	//new (&dummyConvertor)DummyConvertor(conf_dict["dummy_conf"].c_str());
	dummyConvertor=DummyConvertor(conf_dict["dummy_conf"].c_str());
	this->p_data_stream=new istringstream();
	this->cursor=0;
	this->lbfgs_iterations=0;
	this->save_intermediate_peroid=10;
}
lbfgsfloatval_t LogisticRegression::evalute(const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,const lbfgsfloatval_t step)
{
	cout<<"evalute start"<<endl;
	if(g!=0)
		memset(g,0,sizeof(g)*n);
//	new (&(this->data_stream)) istringstream(this->data_buffer);
	this->prepare_read();
	vector<string> field_vec;
	CommonTool::split(this->conf_dict[string("filter:field_name_vec")],',',field_vec);
	string line;
	vector<string> record;
	set<string> fields;
	lbfgsfloatval_t NLL=0.0;
	int line_cnt=0;
	time_t t=time(0);
	while(get_line(line))
	{
		line_cnt++;
		if(line_cnt%1000000==0)
		{
			t=time(0);
			cout<<"progress:"<<line_cnt<<" time:"<<asctime(localtime(&t))<<endl;
		}
		CommonTool::split(line,'\t',record);
/*
		int yi=atoi(record.back().c_str());
		LogisticRegression::convert_record(record,fields,field_vec);
		lbfgsfloatval_t ti=0.0;
		for(set<string>::const_iterator iter=fields.begin();iter!=fields.end();iter++)
		{
			int i=this->dummyConvertor.convert(*iter);
//			cout<<"index="<<i<<" record:"<<line<<endl;
//			cout.flush();
			if(i>=n)
			{
				cout<<"Error index:"<<i<<" for item:"<<*iter<<endl;
			}
			ti+=w[i];
		}
		lbfgsfloatval_t ui=CommonTool::sigmod(ti);
		lbfgsfloatval_t gradient=ui-yi;
		for(set<string>::const_iterator iter=fields.begin();iter!=fields.end();iter++)
		{
			int i=this->dummyConvertor.convert(*iter);
			g[i]+=gradient;
		}
*/
		int yi=atoi(record.front().c_str());
		lbfgsfloatval_t ui=predict(record,w);
		lbfgsfloatval_t gradient=ui-yi;
		if(g!=0)
		{
			for(int j=1;j<record.size();j++)
			{
				int k=this->dummyConvertor.convert(record[j]);
				g[k]+=gradient;
			}
		}
		NLL+=-(yi*log(ui)+(1-yi)*log(1-ui));
	}
	this->finish_read();
	this->lbfgs_iterations++;
	if(this->lbfgs_iterations%this->save_intermediate_peroid==0 && this->intermediate_file_prefix.size()>0)
	{
		string parameter_values;
		this->get_parameter_values(parameter_values);
		ostringstream sout;
		sout<<this->intermediate_file_prefix<<"."<<this->lbfgs_iterations;
		ofstream fout(sout.str().c_str());
		fout<<parameter_values;
		fout.close();
	}
	cout<<"end evalute,NLL="<<NLL<<endl;
	return NLL;
}
int LogisticRegression::optimize(const string& intermediate_file_prefix,string& final_para_results)
{
	this->lbfgs_iterations=0;
	this->intermediate_file_prefix=intermediate_file_prefix;
	int n=dummyConvertor.get_length();
	this->parameters=lbfgs_malloc(n);
	if(this->parameters==0)
	{
		cout<<"malloc memery for parameters faild."<<endl;
		return -1;
	}
	double rmax=(double)RAND_MAX;
	for(int i=0;i<n;i++)
	{
		this->parameters[i]=(rand()/rmax)*2-1;
	}
	/// end of random initialize
	lbfgsfloatval_t NLL=-1.0;
	time_t t=time(0);
	cout<<"begin LBFGS, time:"<<asctime(localtime(&t))<<endl;
	int status=lbfgs(n,this->parameters,&NLL,_evalute,0,this,0);
	t=time(0);
	cout<<"end LBFGS. NLL="<<NLL<<" time:"<<asctime(localtime(&t))<<endl;
	get_parameter_values(final_para_results);
	lbfgs_free(this->parameters);
	return  status;
}
int LogisticRegression::get_parameter_values(string &result)const
{
	int n=this->dummyConvertor.get_length();
	ostringstream sout;
	for(int i=0;i<n;i++)
		sout<<this->parameters[i]<<endl;
	result.assign(sout.str());
	return n;
}
int LogisticRegression::load_parameter_from_file(const char* filename)
{
	ifstream fin(filename);
	if(!fin)
	{
		cout<<"Error, open "<<filename<<" failed."<<endl;
		return -1;
	}
	vector<string> lines;
	string line;
	while(getline(fin,line))
	{
		lines.push_back(line);
	}
	this->parameters=new lbfgsfloatval_t[lines.size()];
	if(this->parameters==0)
	{
		cout<<"Error, allocate memery faild."<<endl;
		return -1;
	}
	for(size_t i=0;i<lines.size();i++)
	{
		this->parameters[i]=atof(lines[i].c_str());
	}
	return lines.size();
}
double LogisticRegression::predict(const char *parameters_file,const char *test_file,const char *output)
{
	if(this->load_parameter_from_file(parameters_file)<0)
		return -1.0;
	ifstream fin(test_file);
	if(!fin)
	{
		cout<<"Error, open test file:"<<test_file<<" failed."<<endl;
		return -1.0;
	}
	ofstream fout(output);
	if(!fout)
	{
		cout<<"Error,open output file:"<<output<<" failed."<<endl;
		return -1.0;
	}
	string line;
	vector<string> record;
	double NLL=0.0;
	while(getline(fin,line))
	{
		CommonTool::split(line,'\t',record);
		double ui=this->predict(record,this->parameters);
		int yi=atoi(record.front().c_str());
		NLL+=-(yi*log(ui)+(1-yi)*log((1-ui)));
		fout<<ui<<","<<record.front()<<endl;
	}
	fin.close();
	fout.close();
	if(this->parameters!=0)
		delete this->parameters;
	return NLL;
}
double LogisticRegression::predict(const vector<string>& record, const lbfgsfloatval_t *w)const
{
		double ti=0.0;
		for(int j=1;j<record.size();j++)
		{
			int k=this->dummyConvertor.convert(record[j]);
			ti+=w[k];
		}
		double ui=CommonTool::sigmod(ti);
		return ui;
}
int LogisticRegression::init_data_buffer(const char* train_file)
{
	ifstream fin(train_file);
	if(!fin)
	{
		cout<<"open file:"<<train_file<<" faild."<<endl;
		return -1;
	}
	fin.seekg(0,fin.end);
	unsigned int file_size=fin.tellg();
	fin.seekg(0,fin.beg);
/*	char *buffer=new char[file_size+1];
	if(buffer==0)
	{
		cout<<"alloct memory faild, file size="<<file_size<<endl;
		return -1;
	}
*/
	this->data_buffer.resize(file_size);
	fin.read(&(this->data_buffer[0]),file_size);
	fin.close();
	cout<<"read file,size="<<this->data_buffer.size()<<endl;
	return 0;
}
int LogisticRegression::prepare_read()
{
//	this->data_stream.seekg(0);
//	this->p_data_stream=new (this->p_data_stream)istringstream(this->data_buffer);
	this->cursor=0;
	return 0;
}
int LogisticRegression::finish_read()
{
//	this->p_data_stream->~istringstream();
//	this->data_stream.close();
	return 0;
}
/*
istream& LogisticRegression::get_line(string &line)
{
	return getline(*(this->p_data_stream),line);
}
*/
bool LogisticRegression::get_line(string &line)
{
	if(this->cursor>=this->data_buffer.size())
		return false;
	line.clear();
	while(this->cursor<this->data_buffer.size() && this->data_buffer[this->cursor]!='\n')
	{
		line.push_back(this->data_buffer[this->cursor]);
		this->cursor++;
	}
	this->cursor++;
	return true;
}
LogisticRegression::~LogisticRegression()
{
	if(p_data_stream!=0)
		delete p_data_stream;
}
int LogisticRegression::convert_record(const vector<string>& record,set<string>& result,const vector<string>& field_vec)
{
	result.clear();
	if(record.size()!=field_vec.size())
		return -1;
	for(int i=0;i<field_vec.size()-1;i++)  // last field is label
	{
		if(field_vec[i].substr(0,3)==string("tag"))
		{
			if(record[i]=="1")
				result.insert(field_vec[i]);
		}else{
			result.insert(field_vec[i]+":"+record[i]);
		}
	}
	return 0;
}

/*
int LogisticRegression::read_dataset(const char* filename, const char* configure)
{
	map<string,string> conf_dict;
	int status=CommonTool::load_configure(configure,conf_dict);
	if(status<0)
	{
		cout<<"load configure file:"<<configure<<" faild."<<endl;
		return -1;
	}
	vector<string> field_vec;
	int n=CommonTool::split(conf_dict["filter:field_name_vec"],',',field_vec);
	string line;
	ifstream fin(filename);
	int cnt=0;
	while(getline(fin,line))
	{
		cnt++;
		vector<string> record;
		set<string> result;
		CommonTool::split(line,'\t',record);
		this->labels.push_back(atoi(record.back().c_str()));
		if(record.size()!=field_vec.size())
		{
			cout<<"error line:(fields="<<record.size()<<",required "<<field_vec.size()<<")"<<line<<endl;
			continue;
		}
		convert_record(record,result,field_vec);
		this->dataset.push_back(result);
		if(cnt%1000000==0)
		{
			time_t t=time(NULL);
			cout<<"progress:"<<cnt<<" time now:"<<asctime(localtime(&t))<<endl;
		}
	}
	cout<<"read records:"<<cnt<<endl;
}
*/
