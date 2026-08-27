#ifndef TASK13_H
#define TASK13_H

#include "graph.h"
#include <limits>

inline void task13(const Graph* graph) {
    if (graph == nullptr) {
        return;
    }
    if (!graph->isDirected()) {
        std::cout << "Maximum flow requires a directed graph.\n";
        return;
    }

    const int n = graph->size();
    const auto capacity = graph->adjacencyMatrix();
    for (const auto& row : capacity) {
        for (int value : row) {
            if (value < 0) {
                std::cout << "Capacities must be non-negative.\n";
                return;
            }
        }
    }

    int source = -1;
    int sink = -1;
    for (int i = 0; i < n; ++i) {
        bool hasIncoming = false;
        bool hasOutgoing = false;
        for (int j = 0; j < n; ++j) {
            hasIncoming = hasIncoming || capacity[j][i] > 0;
            hasOutgoing = hasOutgoing || capacity[i][j] > 0;
        }
        if (!hasIncoming && hasOutgoing) {
            if (source != -1) {
                std::cout << "Network must have exactly one source.\n";
                return;
            }
            source = i;
        }
        if (hasIncoming && !hasOutgoing) {
            if (sink != -1) {
                std::cout << "Network must have exactly one sink.\n";
                return;
            }
            sink = i;
        }
    }

    if (source == -1 || sink == -1 || source == sink) {
        std::cout << "Could not determine a unique source and sink.\n";
        return;
    }

    std::vector<std::vector<long long>> residual(
        n, std::vector<long long>(n, 0));
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            residual[u][v] = capacity[u][v];
        }
    }

    long long maxFlow = 0;
    std::vector<int> parent(n);
    while (true) {
        std::fill(parent.begin(), parent.end(), -1);
        parent[source] = source;
        std::queue<int> queue;
        queue.push(source);

        while (!queue.empty() && parent[sink] == -1) {
            const int u = queue.front();
            queue.pop();
            for (int v = 0; v < n; ++v) {
                if (parent[v] == -1 && residual[u][v] > 0) {
                    parent[v] = u;
                    queue.push(v);
                }
            }
        }
        if (parent[sink] == -1) {
            break;
        }

        long long pathFlow = std::numeric_limits<long long>::max();
        for (int v = sink; v != source; v = parent[v]) {
            pathFlow = std::min(pathFlow, residual[parent[v]][v]);
        }
        for (int v = sink; v != source; v = parent[v]) {
            const int u = parent[v];
            residual[u][v] -= pathFlow;
            residual[v][u] += pathFlow;
        }
        maxFlow += pathFlow;
    }

    std::cout << "Maximum flow value: " << maxFlow << ".\n"
              << "Source: " << source + 1
              << ", sink: " << sink + 1 << ".\n";
}

#endif
