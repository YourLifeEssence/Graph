#include "task19.h"
#include "task13.h"
#include "task14.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    const std::string path = "C:/Users/tutir/Graph tests/task14/list_of_adjacency_t14_006.txt";
    MatrixGraph g(path);
    task14(&g);

    return 0;
}