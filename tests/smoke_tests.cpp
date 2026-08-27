#include "graph.h"
#include "task12.h"
#include "task13.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    MatrixGraph graph("examples/example_graph.txt");
    assert(graph.size() == 4);
    assert(!graph.isDirected());
    assert(!graph.isWeighted());
    assert(graph.adjacencyMatrix()[0][1] == 1);
    assert(graph.adjacencyList()[0].size() == 2);

    ListGraph listGraph("examples/example_graph.txt");
    assert(listGraph.adjacencyMatrix() == graph.adjacencyMatrix());
    assert(!listGraph.isDirected());

    task1(&graph);
    task2(&graph);
    task3(&graph);
    task4(&graph);
    task7(&graph);
    task8(&graph);
    task12(&graph);

    MatrixGraph flow("examples/example_flow.txt");
    assert(flow.size() == 4);
    assert(flow.isDirected());
    assert(flow.isWeighted());
    task13(&flow);

    Map map("examples/example_map.txt");
    assert(map.size() == std::make_pair(5, 5));
    assert(map.valid({0, 0}));
    assert(!map.valid({-1, 0}));
    assert(!a_star(map, {-1, 0}, {0, 0}, manhattan).first.size());

    bool rejectedMissingFile = false;
    try {
        MatrixGraph missing("examples/missing.txt");
    } catch (const std::runtime_error&) {
        rejectedMissingFile = true;
    }
    assert(rejectedMissingFile);

    std::cout << "All smoke tests passed.\n";
    return 0;
}
