#include "dummy_convertor.h"
#include <fstream>
#include <iostream>

DummyConvertor::DummyConvertor(const vector<string>& sequence)
{
	for(int i=0;i<sequence.size();i++)
	{
		dict.insert(map<string,int>::value_type(sequence[i],i));
	}
}

DummyConvertor::DummyConvertor(const char* filename)
{
	ifstream fin(filename);
	string line;
	vector<string> sequence;
	while(getline(fin,line))
	{
		sequence.push_back(line);
	}
	fin.close();
	new(this) DummyConvertor(sequence);
}
int DummyConvertor::convert(const string& item)const
{
	map<string,int>::const_iterator iter=dict.find(item);
	if(iter==dict.end())	return -1;
	return iter->second;
}
/*
int main()
{
	DummyConvertor convertor("conf/dummy.conf");
	cout<<convertor.convert(string("ad_pos:Edu_F_Upright"))<<endl;
	return 0;
}
*/
