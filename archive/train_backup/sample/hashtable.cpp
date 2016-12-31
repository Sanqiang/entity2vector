#include <iostream>
#include <string>
#include <map>
#include <stdio.h>
#include <unordered_map>

using namespace std;

int main() {

    //ordered bst
    map<string, int> x;
    x.insert(make_pair("one",1));
    x.insert(make_pair("two",2));
    x.insert(make_pair("three",3));
    x.insert(make_pair("four",4));
    x.insert(make_pair("five",5));

    for(const auto& i : x) {
        cout << i.first << ":" << i.second << endl;
    }

    //unordered ht
    std::unordered_map<string, int> htmap;
    htmap["one"] = 1;
    htmap["two"] = 2;
    htmap["three"] = 3;
    htmap["four"] = 4;
    htmap["five"] = 5;

    for(const auto& i : htmap) {
        cout << i.first << ":" << i.second << endl;
    }

    return 0;
}