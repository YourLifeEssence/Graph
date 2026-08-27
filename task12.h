#ifndef TASK12_H
#define TASK12_H

#include "graph.h"

inline void task12(const Graph* graph) {
    if (graph == nullptr) {
        return;
    }
    if (graph->isDirected()) {
        std::cout << "Maximum matching requires an undirected graph.\n";
        return;
    }

    const int n = graph->size();
    const auto adjacency = graph->adjacencyList();
    std::vector<int> color(n, -1);
    bool bipartite = true;

    std::function<void(int, int)> colorDfs = [&](int u, int currentColor) {
        color[u] = currentColor;
        for (int v : adjacency[u]) {
            if (color[v] == -1) {
                colorDfs(v, 1 - currentColor);
            } else if (color[v] == currentColor) {
                bipartite = false;
            }
        }
    };

    for (int i = 0; i < n && bipartite; ++i) {
        if (color[i] == -1) {
            colorDfs(i, 0);
        }
    }

    if (!bipartite) {
        std::cout << "Graph is not bipartite.\n";
        return;
    }

    std::vector<int> matchToRight(n, -1);
    std::function<bool(int, std::vector<bool>&)> augment =
        [&](int u, std::vector<bool>& visited) {
            for (int v : adjacency[u]) {
                if (color[u] != 0 || visited[v]) {
                    continue;
                }
                visited[v] = true;
                if (matchToRight[v] == -1 || augment(matchToRight[v], visited)) {
                    matchToRight[v] = u;
                    return true;
                }
            }
            return false;
        };

    int matchingSize = 0;
    for (int u = 0; u < n; ++u) {
        if (color[u] == 0) {
            std::vector<bool> visited(n, false);
            if (augment(u, visited)) {
                ++matchingSize;
            }
        }
    }

    std::cout << "Size of maximum matching: " << matchingSize << ".\n";
    std::cout << "Maximum matching:\n{";
    for (int v = 0; v < n; ++v) {
        if (matchToRight[v] != -1) {
            std::cout << " (" << matchToRight[v] + 1 << ", " << v + 1 << ")";
        }
    }
    std::cout << " }\n";
}

#endif
