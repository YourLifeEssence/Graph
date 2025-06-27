#include "graph.h"
#include "task19.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    /*
    const std::string path = "C:/Users/tutir/Graph tests/task11/list_of_adjacency_t11_005.txt";
    MatrixGraph g(path);
    task11(&g);
    */

    /*
    Map map("C:/Users/tutir/Downloads/Graph tests/task6/maze_t6_001.txt");
    Point start{1, 5};
    Point end{3, 3};
    task6(map,start,end);
    */

    /* Map map("C:/Users/tutir/Graph tests/task12/map_001.txt");
    Point start{14,6}, end{14,13};
    task12(map,start,end,manhattan);
    */


    const std::string path = "C:/Users/tutir/Graph tests/task19/list_of_adjacency_t19_005.txt";
    MatrixGraph g(path);
    ACO solver(g, 10, 100, 1.0, 5.0, 0.5, 100.0);
    solver.run();

    return 0;
}