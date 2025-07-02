#ifndef TASK14_H
#define TASK14_H
#include "graph.h"

void task14(const Graph* g) {
    int n = g->size();
    auto capacity = g->adjacencyMatrix();
    std::vector<std::vector<int>> flow(n, std::vector<int>(n, 0));

    // Определим source и sink
    int source = -1, sink = -1;
    for (int i = 0; i < n; ++i) {
        bool hasIn = false, hasOut = false;
        for (int j = 0; j < n; ++j) {
            if (capacity[j][i] > 0) hasIn = true;
            if (capacity[i][j] > 0) hasOut = true;
        }
        if (!hasIn && hasOut && source == -1) source = i;
        if (hasIn && !hasOut && sink == -1) sink = i;
    }

    if (source == -1 || sink == -1) {
        std::cout << "Source or sink not found automatically.\n";
        return;
    }

    std::vector<int> parent(n);
    auto bfs = [&](int& path_flow) {
        std::fill(parent.begin(), parent.end(), -1);
        parent[source] = source;
        std::queue<std::pair<int, int>> q;
        q.push({source, INT_MAX});

        while (!q.empty()) {
            int u = q.front().first;
            int curr_flow = q.front().second;
            q.pop();

            for (int v = 0; v < n; ++v) {
                if (parent[v] == -1 && capacity[u][v] - flow[u][v] > 0) {
                    parent[v] = u;
                    int new_flow = std::min(curr_flow, capacity[u][v] - flow[u][v]);
                    if (v == sink) {
                        path_flow = new_flow;
                        return true;
                    }
                    q.push({v, new_flow});
                }
            }
        }
        return false;
    };

    int max_flow = 0;
    int path_flow;

    while (bfs(path_flow)) {
        max_flow += path_flow;
        int v = sink;
        while (v != source) {
            int u = parent[v];
            flow[u][v] += path_flow;
            flow[v][u] -= path_flow;
            v = u;
        }
    }

    std::cout << "Maximum flow value: " << max_flow << ".\n";
    std::cout << "Source: " << source + 1 << ", sink: " << sink + 1 << ".\n";
    std::cout << "Flow:\n";
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            if (flow[u][v] > 0)
                std::cout << u + 1 << "-" << v + 1 << " : " << flow[u][v] << "\n";
        }
    }
}

#endif