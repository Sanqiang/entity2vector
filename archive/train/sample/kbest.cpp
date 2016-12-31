#include <queue>
#include <vector>
#include <iostream>
// maxindices.cc
// compile with:
// g++ -std=c++11 maxindices.cc -o maxindices
int main()
{
    std::vector<double> test = {0.9, 0.1, 0.6, 1.5, 1.9, 3.1, 5.6, 0.1};

    if(1){
        std::priority_queue<std::pair<double, int>> q;
        for (int i = 0; i < test.size(); ++i) {
            q.push(std::pair<double, int>(test[i], i));
        }
        int k = 5; // number of indices we need
        for (int i = 0; i < k; ++i) {
            std::pair<double, int> pair = q.top();
            std::cout << "index[" << pair.second << "] = " << pair.first << std::endl;
            q.pop();
        }
    } else{
        std::nth_element(test.begin(), test.begin(), test.end(), std::greater<int>());
        std::cout << "The first largest element is " << test[0] << '\n';
        std::cout << "The second largest element is " << test[1] << '\n';
        std::cout << "The third largest element is " << test[2] << '\n';
    }
}