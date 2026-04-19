#include <iostream>
#include <chrono>

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

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < n; i++) {
        temp = a[i];
        if (a[i] > b[i]) {
            temp = b[i];
        }
        std::cout << temp << " ";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    return 0;
}