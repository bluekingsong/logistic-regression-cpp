#ifndef __DUMMY_CONVERTOR_H__
#define __DUMMY_CONVERTOR_H__

#include <map>
#include <vector>
#include <string>
using namespace std;

class DummyConvertor
{
  private:
	map<string,int> dict;
  public:
	DummyConvertor(){};
	DummyConvertor(const vector<string>& sequence);
	DummyConvertor(const char* filename);
	int convert(const string& item)const;
	unsigned int get_length()const
	{
		return dict.size();
	}
};

#endif
