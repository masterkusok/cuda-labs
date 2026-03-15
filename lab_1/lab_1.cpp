#include <iostream>

int main() {
    int n;
    std::cin >> n;

    double temp;
    std::vector<double> a;
    for(int i = 0; i < n; i++) {
        std::cin >> temp;
        a.push_back(temp);
    }

    std::vector<double> b;
    for(int i = 0; i < n; i++) {
        std::cin >> temp;
        b.push_back(temp);
    }

    for(int i = 0; i < n; i++) {
        temp = a[i];
        if (a[i] > b[i]) {
            temp = b[i];
        }
        std::cout << temp << " ";
    }

    return 0;
}