#ifndef TASK13_H
#define TASK13_H
#include "graph.h"

void task13(const Graph* graph) {
    int n = graph->size();
    std::vector<std::vector<int>> adj = graph->adjacencyList();

    std::vector<int> color(n, -1);
    bool isBipartite = true;

    std::function<void(int, int)> dfs_color = [&](int u, int c) {
        color[u] = c;
        for (int v : adj[u]) {
            if (color[v] == -1)
                dfs_color(v, 1 - c);
            else if (color[v] == color[u])
                isBipartite = false;
        }
    };

    for (int i = 0; i < n && isBipartite; ++i)
        if (color[i] == -1)
            dfs_color(i, 0);

    if (!isBipartite) {
        std::cout << "Graph is not bipartite.\n";
        return;
    }

    std::vector<int> matchTo(n, -1);
    std::function<bool(int, std::vector<bool>&)> kuhn = [&](int u, std::vector<bool>& visited) {
        for (int v : adj[u]) {
            if (color[u] != 0) continue; // только из левой доли
            if (visited[v]) continue;
            visited[v] = true;
            if (matchTo[v] == -1 || kuhn(matchTo[v], visited)) {
                matchTo[v] = u;
                return true;
            }
        }
        return false;
    };

    int matchCount = 0;
    for (int u = 0; u < n; ++u) {
        if (color[u] == 0) {
            std::vector<bool> visited(n, false);
            if (kuhn(u, visited)) ++matchCount;
        }
    }

    std::cout << "Size of maximum matching: " << matchCount << ".\n";
    std::cout << "Maximum matching:\n{";
    for (int v = 0; v < n; ++v) {
        if (matchTo[v] != -1) {
            std::cout << "(" << matchTo[v]+1 << ", " << v+1 << "), ";
        }
    }
    std::cout << "}\n";
}

#endif