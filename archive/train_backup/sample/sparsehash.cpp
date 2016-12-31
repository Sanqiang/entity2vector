#include <iostream>
#include <sparsehash/dense_hash_map>

using google::dense_hash_map;      // namespace where class lives by default
using std::cout;
using std::endl;
using std::hash;  // or __gnu_cxx::hash, or maybe tr1::hash, depending on your OS

struct eqstr
{
    bool operator()(const char* s1, const char* s2) const
    {
        return (s1 == s2) || (s1 && s2 && strcmp(s1, s2) == 0);
    }
};

bool lookup(dense_hash_map<const char*, int, hash<const char*>, eqstr> months, char* word){
    dense_hash_map<const char*, int, hash<const char*>, eqstr>::const_iterator it = months.find(word);
    return (it != months.end() ? true : false);
}

int main()
{
    dense_hash_map<const char*, int, hash<const char*>, eqstr> months;

    months.set_empty_key(NULL);
    months["january"] = 31;
    months["february"] = 28;
    months["march"] = 31;
    months["april"] = 30;
    months["may"] = 31;
    months["june"] = 30;
    months["july"] = 31;
    months["august"] = 31;
    months["september"] = 30;
    months["october"] = 31;
    months["november"] = 30;
    months["december"] = 31;

    cout << "september -> " << months["september"] << endl;
    cout << "april     -> " << months["april"] << endl;
    cout << "june      -> " << months["june"] << endl;
    cout << "november  -> " << months["november"] << endl;

    cout << "size   -> " << months.size() << endl;

    cout << lookup(months, "may") << endl;
    cout << lookup(months, "mayx") << endl;
}