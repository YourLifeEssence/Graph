#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <windows.h>
#include <set>
#include <chrono>
#include <stack>
#include <functional>
#include <limits>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <climits>

class Graph {
public:
    virtual ~Graph() = default;

    [[nodiscard]] virtual int size() const = 0;
    [[nodiscard]] virtual std::vector<std::vector<int>> adjacencyMatrix() const = 0;
    [[nodiscard]] virtual std::vector<std::vector<int>> adjacencyList() const = 0;
    [[nodiscard]] virtual std::vector<std::vector<std::pair<int, int>>> weightedAdjacencyList() const = 0;
    [[nodiscard]] virtual bool isDirected() const = 0;
    [[nodiscard]] virtual bool isWeighted() const = 0;

protected:
    int countVertex{};
    bool directed{};
    bool weighted{};
};

class MatrixGraph final : public Graph {
public:
    explicit MatrixGraph(const std::string& filePath) {
        std::ifstream in(filePath);
        if (!in) {
            throw std::runtime_error("Unable to open graph file: " + filePath);
        }

        std::string line;
        if (!std::getline(in, line) || line.empty()) {
            throw std::runtime_error("Graph file is missing the vertex count.");
        }
        try {
            countVertex = std::stoi(line);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid vertex count in graph file.");
        }
        if (countVertex <= 0) {
            throw std::runtime_error("Vertex count must be positive.");
        }

        std::vector<std::vector<std::string>> rawData;
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            std::vector<std::string> row;
            std::string token;
            while (iss >> token) {
                row.push_back(token);
            }
            rawData.push_back(row);
        }

        if (rawData.empty()) {
            throw std::runtime_error("Graph file contains no graph data.");
        }
        for (const auto& row : rawData) {
            if (row.empty()) {
                throw std::runtime_error("Graph data contains an empty row.");
            }
        }

        Matrix.resize(countVertex, std::vector<int>(countVertex, 0));
        weighted = false;

        if (rawData.size() == countVertex && !rawData.empty() &&
            !rawData[0].empty() && rawData[0][0].find(':') == std::string::npos &&
            rawData[0].size() == countVertex){
            for (int i = 0; i < countVertex; ++i) {
                for (int j = 0; j < countVertex; ++j) {
                    int val = std::stoi(rawData[i][j]);
                    Matrix[i][j] = val;
                    if (val != 0 && val != 1) weighted = true;
                }
            }
        } 
        else if (rawData.size() == countVertex) {
            for (int i = 0; i < countVertex; ++i) {
                for (const std::string& entry : rawData[i]) {
                    if (entry.find(':') != std::string::npos) {
                        size_t pos = entry.find(':');
                        int neighbor = std::stoi(entry.substr(0, pos)) - 1;
                        int weight = std::stoi(entry.substr(pos + 1));
                        Matrix[i][neighbor] = weight;
                        weighted = true;
                    } else {
                        int neighbor = std::stoi(entry) - 1;
                        Matrix[i][neighbor] = 1;
                    }
                }
            }
        } else {
            for (const auto& row : rawData) {
                if (row.size() >= 2) {
                    int from = std::stoi(row[0]) - 1;
                    int to = std::stoi(row[1]) - 1;
                    int weight = (row.size() == 3) ? std::stoi(row[2]) : 1;
                    Matrix[from][to] = weight;
                    if (weight != 1) weighted = true;
                }
            }
        }

        directed = false;
        for (int i = 0; i < countVertex; ++i) {
            for (int j = 0; j < countVertex; ++j) {
                if (Matrix[i][j] != Matrix[j][i]) {
                    directed = true;
                    i = countVertex;
                    break;
                }
            }
        }
    }

    [[nodiscard]] int size() const override {
        return countVertex;
    }

    [[nodiscard]] std::vector<std::vector<int>> adjacencyMatrix() const override {
        return Matrix;
    }

    [[nodiscard]] std::vector<std::vector<int>> adjacencyList() const override {
        std::vector<std::vector<int>> list(countVertex);
        for (int i = 0; i < countVertex; ++i) {
            for (int j = 0; j < countVertex; ++j) {
                if (Matrix[i][j] != 0) {
                    list[i].push_back(j);
                }
            }
        }
        return list;
    }

    [[nodiscard]] std::vector<std::vector<std::pair<int, int>>> weightedAdjacencyList() const override {
        std::vector<std::vector<std::pair<int,int>>> weightedList(countVertex);
        for(int i = 0; i < countVertex; ++i) {
            for(int j = 0; j < countVertex; ++j) {
                if(Matrix[i][j] != 0)
                    weightedList[i].emplace_back(j,Matrix[i][j]);
            }
        }
        return weightedList;
    }

    [[nodiscard]] bool isDirected() const override {
        return directed;
    }

    [[nodiscard]] bool isWeighted() const override {
        return weighted;
    }

private:
    std::vector<std::vector<int>> Matrix;
};

class ListGraph final : public Graph {
public:
    explicit ListGraph(const std::string& filePath) {
        std::ifstream in(filePath);
        if (!in) {
            throw std::runtime_error("Unable to open graph file: " + filePath);
        }

        std::string line;
        if (!std::getline(in, line) || line.empty()) {
            throw std::runtime_error("Graph file is missing the vertex count.");
        }
        try {
            countVertex = std::stoi(line);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid vertex count in graph file.");
        }
        if (countVertex <= 0) {
            throw std::runtime_error("Vertex count must be positive.");
        }

        std::vector<std::vector<std::string>> rawData;
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            std::vector<std::string> row;
            std::string token;
            while (iss >> token) {
                row.push_back(token);
            }
            rawData.push_back(row);
        }

        List.resize(countVertex);
        weightedList.resize(countVertex);
        weighted = false;

        if (rawData.size() == countVertex && !rawData[0].empty() && rawData[0][0].find(':') == std::string::npos && rawData[0].size() == countVertex) {
            for (int i = 0; i < countVertex; ++i) {
                for (int j = 0; j < countVertex; ++j) {
                    int val = std::stoi(rawData[i][j]);
                    if (val != 0) {
                        List[i].push_back(j);
                        weightedList[i].emplace_back(j,val);
                        if (val != 1) weighted = true;
                    }
                }
            }
        } else if (rawData.size() == countVertex) {
            for (int i = 0; i < countVertex; ++i) {
                for (const std::string& entry : rawData[i]) {
                    if (entry.find(':') != std::string::npos) {
                        size_t pos = entry.find(':');
                        int neighbor = std::stoi(entry.substr(0, pos)) - 1;
                        int weight = std::stoi(entry.substr(pos + 1));
                        List[i].push_back(neighbor);
                        weightedList[i].emplace_back(neighbor,weight);
                        weighted = true;
                    } else {
                        int neighbor = std::stoi(entry) - 1;
                        List[i].push_back(neighbor);
                        weightedList[i].emplace_back(neighbor,1);
                    }
                }
            }
        } else {
            for (const auto& row : rawData) {
                if (row.size() >= 2) {
                    int from = std::stoi(row[0]) - 1;
                    int to = std::stoi(row[1]) - 1;
                    int weight = (row.size() == 3) ? std::stoi(row[2]) : 1;
                    List[from].push_back(to);
                    weightedList[from].emplace_back(to, weight);
                    if (row.size() == 3) weighted = true;
                }
            }
        }

        directed = false;
        for (int u = 0; u < countVertex; ++u) {
            for (int v : List[u]) {
                if (std::find(List[v].begin(), List[v].end(), u) == List[v].end()) {
                    directed = true;
                    u = countVertex;
                    break;
                }
            }
        }
    }

    [[nodiscard]] int size() const override {
        return countVertex;
    }

    [[nodiscard]] std::vector<std::vector<int>> adjacencyMatrix() const override {
        std::vector<std::vector<int>> matrix(countVertex,std::vector(countVertex,0));
        for(int i = 0; i < countVertex; ++i) {
            for(auto &[neighbor,weight] : weightedList[i]) {
                matrix[i][neighbor] = weight;
            }
        }
        return matrix;
    }

    [[nodiscard]] std::vector<std::vector<int>> adjacencyList() const override {
        return List;
    }

    [[nodiscard]] std::vector<std::vector<std::pair<int, int>>> weightedAdjacencyList() const override {
        return weightedList;
    }

    [[nodiscard]] bool isDirected() const override {
        return directed;
    }

    [[nodiscard]] bool isWeighted() const override {
        return weighted;
    }

private:
    std::vector<std::vector<int>> List;
    std::vector<std::vector<std::pair<int, int>>> weightedList;
};

void dfs(int u, const std::vector<std::vector<int>>& adj, std::vector<bool>& visited, std::vector<int>& component) {
    visited[u] = true;
    component.push_back(u);
    for (int v : adj[u]) {
        if (!visited[v]) {
            dfs(v, adj, visited, component);
        }
    }
}

void task1(const Graph* graph) {
    std::vector<std::vector<int>> adj = graph->adjacencyList();

    if (graph->isDirected()) {
        std::vector<std::vector<int>> undirected(adj.size());
        for (int u = 0; u < adj.size(); ++u) {
            for (int v : adj[u]) {
                undirected[u].push_back(v);
                undirected[v].push_back(u);
            }
        }
        adj = std::move(undirected);
    }

    int n = graph->size();
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> components;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            std::vector<int> component;
            dfs(i, adj, visited, component);
            components.push_back(component);
        }
    }

    if(!graph->isDirected()) {
        if(components.size() == 1) std::cout << "Граф связный\n";
        else std::cout << "Граф не связный\n";
    }
    else {
        if(components.size() == 1) std::cout << "Орграф слабо связный\n";
        else std::cout << "Орграф не слабо связный\n";
    }
    std::cout << "Количество компонент связности: " << components.size() << "\n";

    for (size_t i = 0; i < components.size(); ++i) {
        std::sort(components[i].begin(), components[i].end());
        std::cout << "Компонента #" << i + 1 << ": ";
        for (int v : components[i]) {
            std::cout << (v + 1) << " ";
        }
        std::cout << "\n";
    }
}

void dfsTask8(int v,int parent, int& timer,
    const std::vector<std::vector<int>>& adj,
    std::vector<int>& tin, std::vector<int>& low,
    std::vector<bool>& visited, std::vector<bool>& isArticulation,
    std::vector<std::pair<int,int>>& bridges) {

    visited[v] = true;
    tin[v] = low[v] = timer++;
    int children = 0;

    for(int to : adj[v]) {
        if (to == parent) continue;
        if (visited[to]) {
            low[v] = std::min(low[v],tin[to]);
        }
        else {
            dfsTask8(to, v, timer, adj, tin, low, visited, isArticulation, bridges);
            low[v] = std::min(low[v], low[to]);
            if (low[to] > tin[v]) {
                bridges.emplace_back(v, to);
            }
            if (parent != -1 && low[to] >= tin[v]) {
                isArticulation[v] = true;
            }
            ++children;
        }
    }
    if (parent == -1 && children > 1) isArticulation[v] = true;
}

void task7(const Graph* graph) {
    if (graph == nullptr) {
        return;
    }
    if (graph->isDirected()) {
        std::cout << "Поиск мостов и точек сочленения поддерживается только для неориентированного графа.\n";
        return;
    }

    const int n = graph->size();
    int timer = 0;
    const auto adj = graph->adjacencyList();
    std::vector<int> tin(n, -1);
    std::vector<int> low(n, -1);
    std::vector<bool> visited(n, false);
    std::vector<bool> isArticulation(n, false);
    std::vector<std::pair<int, int>> bridges;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            dfsTask8(i, -1, timer, adj, tin, low, visited,
                     isArticulation, bridges);
        }
    }

    std::cout << "Мосты:\n";
    for (auto [from, to] : bridges) {
        if (from > to) {
            std::swap(from, to);
        }
        std::cout << from + 1 << " - " << to + 1 << "\n";
    }
    std::cout << "Точки сочленения:\n";
    for (int i = 0; i < n; ++i) {
        if (isArticulation[i]) {
            std::cout << i + 1 << "\n";
        }
    }
}

void dfsTask3(int v, std::vector<std::vector<int>>& adj, std::vector<bool>& visited, std::vector<std::pair<int, int>>& tree) {
    visited[v] = true;
    for(int u : adj[v]) {
        if(!visited[u]) {
            tree.push_back({v,u});
            dfsTask3(u,adj,visited,tree);
        }
    }
}

void task2(const Graph* graph) {
    int n = graph->size();
    std::vector<std::vector<int>> adj = graph->adjacencyList();
    std::vector<bool> visited(n,false);
    std::vector<std::pair<int, int>> tree;
    for(int i = 0; i < n; ++i) {
        if(!visited[i]) {
            dfsTask3(i,adj,visited,tree);
        }
    }
    std::cout << "Остовное дерево:\n";
    for(auto [from, to] : tree) {
        std::cout << from + 1 << " - " << to + 1 << std::endl;
    }
    std::cout << "Количество ребер: " << tree.size();
}

void algorithmPrim(const Graph* graph) {
    if (graph == nullptr || graph->isDirected()) {
        std::cout << "Prim requires a non-directed graph.\n";
        return;
    }

    const auto begin = std::chrono::high_resolution_clock::now();
    const int n = graph->size();
    const auto matrix = graph->adjacencyMatrix();
    const int infinity = std::numeric_limits<int>::max();
    std::vector<int> key(n, infinity);
    std::vector<int> parent(n, -1);
    std::vector<bool> inMst(n, false);
    std::priority_queue<std::pair<int, int>,
                        std::vector<std::pair<int, int>>,
                        std::greater<std::pair<int, int>>> queue;

    key[0] = 0;
    queue.push({0, 0});

    while (!queue.empty()) {
        const int u = queue.top().second;
        queue.pop();
        if (inMst[u]) {
            continue;
        }
        inMst[u] = true;

        for (int v = 0; v < n; ++v) {
            const int weight = matrix[u][v];
            if (weight != 0 && !inMst[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                queue.push({weight, v});
            }
        }
    }

    for (bool included : inMst) {
        if (!included) {
            std::cout << "Graph is not connected; MST was not built.\n";
            return;
        }
    }

    int totalWeight = 0;
    std::cout << "Остовное дерево (Прим):\n";
    for (int vertex = 1; vertex < n; ++vertex) {
        const int from = parent[vertex];
        std::cout << from + 1 << " - " << vertex + 1
                  << " (вес: " << matrix[from][vertex] << ")\n";
        totalWeight += matrix[from][vertex];
    }
    std::cout << "Суммарный вес: " << totalWeight << "\n";

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - begin;
    std::cout << "Время выполнения алгоритма Прима: " << elapsed.count()
              << " секунд\n";
}
void algorithmKruskal(const Graph* graph) {
    if (graph == nullptr || graph->isDirected()) {
        std::cout << "Kruskal requires a non-directed graph.\n";
        return;
    }

    const auto begin = std::chrono::high_resolution_clock::now();
    const int n = graph->size();
    const auto weightedList = graph->weightedAdjacencyList();
    std::vector<std::tuple<int, int, int>> edges;
    std::set<std::pair<int, int>> addedEdges;

    for (int from = 0; from < n; ++from) {
        for (const auto [to, weight] : weightedList[from]) {
            const int first = std::min(from, to);
            const int second = std::max(from, to);
            if (addedEdges.insert({first, second}).second) {
                edges.emplace_back(first, second, weight);
            }
        }
    }

    std::sort(edges.begin(), edges.end(), [](const auto& left, const auto& right) {
        return std::get<2>(left) < std::get<2>(right);
    });

    std::vector<int> parent(n);
    std::vector<int> rank(n, 0);
    for (int vertex = 0; vertex < n; ++vertex) {
        parent[vertex] = vertex;
    }

    std::function<int(int)> find = [&](int vertex) {
        if (parent[vertex] != vertex) {
            parent[vertex] = find(parent[vertex]);
        }
        return parent[vertex];
    };
    auto unite = [&](int left, int right) {
        left = find(left);
        right = find(right);
        if (left == right) {
            return false;
        }
        if (rank[left] < rank[right]) {
            std::swap(left, right);
        }
        parent[right] = left;
        if (rank[left] == rank[right]) {
            ++rank[left];
        }
        return true;
    };

    int totalWeight = 0;
    int edgeCount = 0;
    std::cout << "\nОстовное дерево (Краскал):\n";
    for (const auto [from, to, weight] : edges) {
        if (unite(from, to)) {
            std::cout << from + 1 << " - " << to + 1
                      << " (вес: " << weight << ")\n";
            totalWeight += weight;
            ++edgeCount;
        }
    }

    if (edgeCount != n - 1) {
        std::cout << "Graph is not connected; MST was not built.\n";
        return;
    }
    std::cout << "Суммарный вес: " << totalWeight << "\n";

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - begin;
    std::cout << "Время выполнения алгоритма Краскала: " << elapsed.count()
              << " секунд\n";
}

void task8(const Graph* graph) {
    if (graph == nullptr || graph->isDirected()) {
        std::cout << "MST requires a non-directed graph.\n";
        return;
    }

    const int n = graph->size();
    const auto adjacency = graph->adjacencyList();
    std::vector<bool> visited(n, false);
    std::queue<int> queue;
    queue.push(0);
    visited[0] = true;

    while (!queue.empty()) {
        const int vertex = queue.front();
        queue.pop();
        for (const int neighbor : adjacency[vertex]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    if (std::find(visited.begin(), visited.end(), false) != visited.end()) {
        std::cout << "Graph is not connected; MST was not built.\n";
        return;
    }

    algorithmPrim(graph);
    algorithmKruskal(graph);
}

void task3(const Graph* graph) {
    const int INF = 1e9;
    int n = graph->size();
    std::vector<std::vector<int>> matrix = graph->adjacencyMatrix();
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(i == j) matrix[i][j] = 0; //Расстояние до себя
            else if(matrix[i][j] == 0) matrix[i][j] = INF; //Нет ребра
        }
    }
    //Пункт а
    std::vector<int> degree(n,0);
    std::vector<int> eccentricities(n, 0);
    std::vector<int> peripheralVertices;
    std::vector<int> centralVertices;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if (matrix[i][j] != 0) degree[i]++;
        }
    }
    //Алг. Флойда
    for(int k = 0; k < n; ++k) {
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < n; ++j) {
                if(matrix[i][k] < INF && matrix[k][j] < INF) {
                    matrix[i][j] = std::min(matrix[i][j], matrix[i][k] + matrix[k][j]);
                }
            }
        }
    }
    for(int i = 0; i < n; ++i) {
        int maxDist = 0;
        for(int j = 0; j < n; ++j) {
            if (i != j && matrix[i][j] > maxDist)
                maxDist = matrix[i][j];
        }
        eccentricities[i] = maxDist;
    }
    int radius = *std::min_element(eccentricities.begin(), eccentricities.end());
    int diameter = *std::max_element(eccentricities.begin(), eccentricities.end());
    for (int i = 0; i < n; ++i) {
        if (eccentricities[i] == diameter)
            peripheralVertices.push_back(i);

        if (eccentricities[i] == radius) {
            centralVertices.push_back(i);
        }
    }
    //Вывод
    std::cout << "Vertices degrees:\n";
    for(int i = 0; i < n; ++i) {
        std::cout << degree[i] << " ";
    }

    std::cout << "\nEccentricity:\n";
    for(int i = 0; i < n; ++i) {
        if(eccentricities[i] == INF) std::cout << "+Infinity ";
        else std::cout << eccentricities[i] << " ";
    }

    if(eccentricities[0] == INF) std::cout << "\nRadius: +Infinity";
    else std::cout << "\nRadius: " << radius;

    std::cout << "\ncentralVertices:\n";
    for(int i = 0; i < centralVertices.size(); ++i) {
        std::cout << centralVertices[i] + 1<< " ";
    }

    if(eccentricities[0] == INF) std::cout << "\nDiameter: +Infinity";
    else std::cout << "\nDiameter: " << diameter;

    std::cout << "\nperipheralVertices:\n";
    for(int i = 0; i < peripheralVertices.size(); ++i) {
        std::cout << peripheralVertices[i] + 1<< " ";
    }
}

void task4(const Graph* graph) {
    int n = graph->size();
    std::vector<std::vector<int>> adj = graph->adjacencyList();
    std::vector<int> color(n, -1);

    bool isBipartite = true;
    std::queue<int> q;

    for (int start = 0; start < n && isBipartite; ++start) {
        if (color[start] == -1) {
            color[start] = 0;
            q.push(start);

            while (!q.empty() && isBipartite) {
                int u = q.front(); q.pop();
                for (int v : adj[u]) {
                    if (color[v] == -1) {
                        color[v] = 1 - color[u];
                        q.push(v);
                    } else if (color[v] == color[u]) {
                        isBipartite = false;
                        break;
                    }
                }
            }
        }
    }

    if (isBipartite) {
        std::cout << "Граф является двудольным.\n";
        std::vector<int> part1, part2;
        for (int i = 0; i < n; ++i) {
            if (color[i] == 0) part1.push_back(i + 1);
            else if (color[i] == 1) part2.push_back(i + 1);
        }

        std::cout << "Доля 1: ";
        for (int v : part1) std::cout << v << " ";
        std::cout << "\nДоля 2: ";
        for (int v : part2) std::cout << v << " ";
        std::cout << "\n";
    } else {
        std::cout << "Граф не является двудольным.\n";
    }
}

class Map {
public:
    Map(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Unable to open map file: " + filename);
        }

        std::string firstLine;
        if (!std::getline(in, firstLine)) {
            throw std::runtime_error("Map file is empty.");
        }

        std::istringstream iss(firstLine);
        std::vector<int> numbers;
        int number = 0;
        while (iss >> number) {
            numbers.push_back(number);
        }

        if (numbers.size() == 2) {
            rows = numbers[0];
            cols = numbers[1];
            if (rows <= 0 || cols <= 0) {
                throw std::runtime_error("Map dimensions must be positive.");
            }
            heights.resize(rows, std::vector<int>(cols));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if (!(in >> heights[i][j])) {
                        throw std::runtime_error("Map data has fewer cells than declared.");
                    }
                }
            }
        } else {
            if (numbers.empty()) {
                throw std::runtime_error("Map file has no data.");
            }
            heights.push_back(numbers);
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) {
                    continue;
                }
                auto row = parseLineToInts(line);
                if (row.empty()) {
                    throw std::runtime_error("Map contains an invalid row.");
                }
                heights.push_back(row);
            }

            rows = static_cast<int>(heights.size());
            cols = static_cast<int>(heights.front().size());
            if (cols == 0) {
                throw std::runtime_error("Map rows must not be empty.");
            }
            for (const auto& row : heights) {
                if (static_cast<int>(row.size()) != cols) {
                    throw std::runtime_error("Map rows must have equal lengths.");
                }
            }
        }
    }

    std::pair<int, int> size() const {
        return {rows, cols};
    }

    [[nodiscard]] bool valid(int i, int j) const {
        return i >= 0 && i < rows && j >= 0 && j < cols;
    }

    int operator()(int i, int j) const {
        return heights.at(i).at(j);
    }

    std::vector<std::pair<int,int>> neighbors(int i, int j) const {
        std::vector<std::pair<int,int>> result;
        const std::vector<std::pair<int,int>> dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        for (auto [di, dj] : dirs) {
            int ni = i + di;
            int nj = j + dj;
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && heights[ni][nj] > 0) {
                result.emplace_back(ni, nj);
            }
        }
        return result;
    }
private:
    std::vector<std::vector<int>> heights;
    int rows = 0, cols = 0;

    static std::vector<int> parseLineToInts(const std::string& line) {
        std::vector<int> result;
        std::istringstream iss(line);
        int x;
        while (iss >> x) {
            result.push_back(x);
        }
        return result;
    }

};

struct Point {
    int x,y;
    bool operator==(const Point &other) const { return x == other.x && y == other.y; }
    bool operator<(const Point& other) const {
        if (x != other.x)
            return x < other.x;
        return y < other.y;
    }
};

std::vector<Point> buildPath(
    const std::vector<std::vector<Point>>& parent,
    Point start,
    Point end
) {
    std::vector<Point> path;
    Point current = end;
    while (!(current == start)) {
        if (current.x < 0 || current.y < 0 ||
            current.x >= static_cast<int>(parent.size()) ||
            current.y >= static_cast<int>(parent[current.x].size())) {
            return {};
        }
        path.push_back(current);
        Point next = parent[current.x][current.y];
        if (next.x < 0 || next.y < 0) { // если родитель - (-1,-1), прерываем
            return {};
        }
        current = next;
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());
    return path;
}

void task5(const Map& map, Point start, Point end) {
    if (!map.valid(start.x, start.y) || !map.valid(end.x, end.y) ||
        map(start.x, start.y) <= 0 || map(end.x, end.y) <= 0) {
        std::cout << "Invalid or blocked start/end cell.\n";
        return;
    }

    const auto [rows, cols] = map.size();
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<Point>> parent(
        rows, std::vector<Point>(cols, {-1, -1}));

    std::queue<Point> queue;
    queue.push(start);
    visited[start.x][start.y] = true;

    bool found = false;
    while (!queue.empty()) {
        const Point current = queue.front();
        queue.pop();
        if (current == end) {
            found = true;
            break;
        }
        for (const auto [nextRow, nextColumn] : map.neighbors(current.x, current.y)) {
            if (!visited[nextRow][nextColumn]) {
                visited[nextRow][nextColumn] = true;
                parent[nextRow][nextColumn] = current;
                queue.push({nextRow, nextColumn});
            }
        }
    }

    if (!found) {
        std::cout << "No path found.\n";
        return;
    }

    const auto path = buildPath(parent, start, end);
    std::cout << "Length of path from (" << start.x << ", " << start.y
              << ") to (" << end.x << ", " << end.y << "): "
              << static_cast<int>(path.size()) - 1 << "\n";
    std::cout << "Path:\n[";
    for (size_t i = 0; i < path.size(); ++i) {
        std::cout << "(" << path[i].x << ", " << path[i].y << ")";
        if (i + 1 < path.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void dfsTask7(int v, const std::vector<std::vector<int>>& adj, std::vector<bool>& visited, std::stack<int>& order) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u]) {
            dfsTask7(u, adj, visited, order);
        }
    }
    order.push(v);
}

void dfsTask7_2(int v, const std::vector<std::vector<int>>& adjT, std::vector<bool>& visited, std::vector<int>& component) {
    visited[v] = true;
    component.push_back(v + 1);
    for (int u : adjT[v]) {
        if (!visited[u]) {
            dfsTask7_2(u, adjT, visited, component);
        }
    }
}

void task6(const Graph* graph) {
    if (!graph->isDirected()) {
        std::cout << "Graph must be directed\n";
        return;
    }

    auto adj = graph->adjacencyList();
    int n = graph->size();

    std::vector<bool> visited(n, false);
    std::stack<int> order;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            dfsTask7(i, adj, visited, order);
        }
    }
    std::vector<std::vector<int>> adjT(n);
    for (int v = 0; v < n; ++v) {
        for (int u : adj[v]) {
            adjT[u].push_back(v);
        }
    }

    visited.assign(n, false);
    std::vector<std::vector<int>> components;

    while (!order.empty()) {
        int v = order.top();
        order.pop();

        if (!visited[v]) {
            std::vector<int> component;
            dfsTask7_2(v, adjT, visited, component);
            std::sort(component.begin(), component.end());
            components.push_back(component);
        }
    }

    if (components.size() == 1) {
        std::cout << "Digraph is strongly connected\n\n";
    } else {
        std::cout << "Digraph is not strongly connected\n\n";
    }

    std::cout << "Strongly connected components:\n";
    for (const auto& comp : components) {
        std::cout << "[";
        for (size_t i = 0; i < comp.size(); ++i) {
            std::cout << comp[i];
            if (i + 1 < comp.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

void task9(const Graph* graph) {
    const auto& matrix = graph->adjacencyMatrix();
    int n = graph->size();
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 1e9));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                dist[i][j] = 0;
            else if (matrix[i][j] != 0)
                dist[i][j] = matrix[i][j];
        }
    }
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> components;

    std::function<void(int, std::vector<int>&)> dfs = [&](int v, std::vector<int>& comp) {
        visited[v] = true;
        comp.push_back(v);
        for (int u = 0; u < n; ++u) {
            if ((matrix[v][u] != 0 || matrix[u][v] != 0) && !visited[u])
                dfs(u, comp);
        }
    };

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            std::vector<int> comp;
            dfs(i, comp);
            components.push_back(comp);
        }
    }

    for (const auto& comp : components) {
        std::vector<int> degrees;
        std::vector<double> eccentricity;

        for (int v : comp) {
            int deg = 0;
            for (int u = 0; u < n; ++u) {
                if (matrix[v][u] != 0 || matrix[u][v] != 0)
                    deg++;
            }
            degrees.push_back(deg);

            double maxDist = 0;
            for (int u : comp) {
                if (dist[v][u] < 1e9)
                    maxDist = std::max(maxDist, dist[v][u]);
            }
            eccentricity.push_back(maxDist);
        }

        double R = *std::min_element(eccentricity.begin(), eccentricity.end());
        double D = *std::max_element(eccentricity.begin(), eccentricity.end());

        std::vector<int> central, peripheral;
        for (int i = 0; i < comp.size(); ++i) {
            if (eccentricity[i] == R)
                central.push_back(comp[i] + 1);
            if (eccentricity[i] == D)
                peripheral.push_back(comp[i] + 1);
        }

        // Вывод
        std::cout << "\nVertices list in component:\n[";
        for (size_t i = 0; i < comp.size(); ++i)
            std::cout << comp[i] + 1 << (i + 1 < comp.size() ? ", " : "");
        std::cout << "]\n";

        std::cout << "Vertices degrees:\n[";
        for (size_t i = 0; i < degrees.size(); ++i)
            std::cout << degrees[i] << (i + 1 < degrees.size() ? ", " : "");
        std::cout << "]\n";

        std::cout << "Eccentricity:\n[";
        for (size_t i = 0; i < eccentricity.size(); ++i)
            std::cout << eccentricity[i] << (i + 1 < eccentricity.size() ? ", " : "");
        std::cout << "]\n";

        std::cout << "R = " << R << "\n";
        std::cout << "Central vertices:\n[";
        for (size_t i = 0; i < central.size(); ++i)
            std::cout << central[i] << (i + 1 < central.size() ? ", " : "");
        std::cout << "]\n";

        std::cout << "D = " << D << "\n";
        std::cout << "Peripherial vertices:\n[";
        for (size_t i = 0; i < peripheral.size(); ++i)
            std::cout << peripheral[i] << (i + 1 < peripheral.size() ? ", " : "");
        std::cout << "]\n\n";
    }
}

void task10(const Graph* graph) {
    if (graph == nullptr) {
        return;
    }

    const int n = graph->size();
    const int infinity = std::numeric_limits<int>::max() / 4;
    const auto weightedList = graph->weightedAdjacencyList();
    std::vector<std::tuple<int, int, int>> edges;
    for (int from = 0; from < n; ++from) {
        for (const auto [to, weight] : weightedList[from]) {
            edges.emplace_back(from, to, weight);
        }
    }

    std::cout << "Enter the start vertex (1-" << n << "): ";
    int startVertex = 0;
    if (!(std::cin >> startVertex) || startVertex < 1 || startVertex > n) {
        std::cout << "Invalid start vertex.\n";
        return;
    }
    const int start = startVertex - 1;
    std::vector<int> distance(n, infinity);
    distance[start] = 0;

    for (int iteration = 0; iteration < n - 1; ++iteration) {
        bool changed = false;
        for (const auto [from, to, weight] : edges) {
            if (distance[from] != infinity &&
                distance[from] + weight < distance[to]) {
                distance[to] = distance[from] + weight;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    for (const auto [from, to, weight] : edges) {
        if (distance[from] != infinity &&
            distance[from] + weight < distance[to]) {
            std::cout << "A reachable negative cycle exists.\n";
            return;
        }
    }

    std::cout << "Shortest path lengths from " << startVertex << ":\n{";
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << i + 1 << ": ";
        if (distance[i] == infinity) {
            std::cout << "+Infinity";
        } else {
            std::cout << distance[i];
        }
    }
    std::cout << "}\n";
}

double manhattan(const Point& a, const Point& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}
double chebyshev(const Point& a, const Point& b) {
    return std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
}
double euclidean(const Point& a, const Point& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

std::vector<Point> reconstruct_path(
    const std::vector<std::vector<Point>>& came_from,
    Point current
) {
    std::vector<Point> path;
    while (current.x != -1 && current.y != -1) {
        if (current.x < 0 || current.x >= static_cast<int>(came_from.size()) ||
            current.y < 0 ||
            current.y >= static_cast<int>(came_from[current.x].size())) {
            return {};
        }
        path.push_back(current);
        current = came_from[current.x][current.y];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::pair<std::vector<Point>, double> a_star(
    const Map& map,
    Point start,
    Point end,
    std::function<double(const Point&, const Point&)> heuristic
) {
    const auto [rows, cols] = map.size();
    if (!map.valid(start.x, start.y) || !map.valid(end.x, end.y) ||
        map(start.x, start.y) <= 0 || map(end.x, end.y) <= 0) {
        return {};
    }

    constexpr double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> gScore(
        rows, std::vector<double>(cols, infinity));
    std::vector<std::vector<Point>> cameFrom(
        rows, std::vector<Point>(cols, {-1, -1}));
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

    struct ComparePriority {
        bool operator()(
            const std::pair<double, Point>& left,
            const std::pair<double, Point>& right
        ) const {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return right.second < left.second;
        }
    };

    using QueueElement = std::pair<double, Point>;
    std::priority_queue<QueueElement, std::vector<QueueElement>, ComparePriority> open;

    gScore[start.x][start.y] = 0.0;
    open.emplace(heuristic(start, end), start);

    while (!open.empty()) {
        const Point current = open.top().second;
        open.pop();

        if (visited[current.x][current.y]) {
            continue;
        }
        visited[current.x][current.y] = true;

        if (current == end) {
            return {reconstruct_path(cameFrom, current), gScore[current.x][current.y]};
        }

        for (const auto [nextRow, nextColumn] : map.neighbors(current.x, current.y)) {
            const Point neighbor{nextRow, nextColumn};
            const int movementCost =
                std::abs(nextRow - current.x) +
                std::abs(nextColumn - current.y) +
                std::abs(map(nextRow, nextColumn) - map(current.x, current.y));
            const double tentativeScore =
                gScore[current.x][current.y] + movementCost;

            if (tentativeScore < gScore[nextRow][nextColumn]) {
                gScore[nextRow][nextColumn] = tentativeScore;
                cameFrom[nextRow][nextColumn] = current;
                open.emplace(
                    tentativeScore + heuristic(neighbor, end), neighbor);
            }
        }
    }

    return {};
}

void task11(
    const Map& map,
    Point start,
    Point end,
    std::function<double(const Point&, const Point&)> heuristic
) {
    auto [path, cost] = a_star(map, start, end, heuristic);

    if (path.empty()) {
        std::cout << "Пути нет";
    } else {
        std::cout << cost << " - length of path between ("
                  << start.x << ", " << start.y << ") and ("
                  << end.x << ", " << end.y << ") points\nPath:\n[";
        for (int i = 0; i < path.size(); ++i) {
            std::cout << "(" << path[i].x << ", " << path[i].y << ")";
            if (i + 1 != path.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

#endif // GRAPH_H