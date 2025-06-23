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
#include <cmath>

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

class MatrixGraph final: public Graph {
public:
    explicit MatrixGraph(const std::string& filePath) {
        std::ifstream in(filePath);
        std::string line;
        std::getline(in, line);
        countVertex = std::stoi(line);

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

        Matrix.resize(countVertex, std::vector<int>(countVertex, 0));
        weighted = false;

        if (rawData.size() == countVertex && !rawData.empty() &&
            !rawData[0].empty() && rawData[0][0].find(':') == std::string::npos &&
            rawData[0].size() == countVertex)
            {
            for (int i = 0; i < countVertex; ++i) {
                for (int j = 0; j < countVertex; ++j) {
                    int val = std::stoi(rawData[i][j]);
                    Matrix[i][j] = val;
                    if (val != 0 && val != 1) weighted = true;
                }
            }
        } else if (rawData.size() == countVertex) {
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

    [[nodiscard]] std::vector<std::vector<std::pair<int, int>>> weightedAdjacencyList() const {
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

class ListGraph final: public Graph {
public:
    explicit ListGraph(const std::string& filePath) {
        std::ifstream in(filePath);
        std::string line;
        std::getline(in, line);
        countVertex = std::stoi(line);

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

        if (rawData.size() == countVertex && rawData[0][0].find(':') == std::string::npos && rawData[0].size() == countVertex) {
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

    [[nodiscard]] std::vector<std::vector<std::pair<int, int>>> weightedAdjacencyList() const {
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

void task8(const Graph* graph) {
    if(graph->isDirected()) std::cout << "Поиск мостов и шарниров для орграфа не реализован";

    int n = graph->size();
    int timer = 0;
    std::vector<std::vector<int>> adj = graph->adjacencyList();
    std::vector<int> tin(n,-1);
    std::vector<int> low(n,-1);
    std::vector<bool> visited(n,false), isArticulation(n,false);
    std::vector<std::pair<int,int>> bridges;

    for(int i = 0; i < n; ++i) {
        if(!visited[i]) {
            dfsTask8(i, -1, timer, adj, tin, low, visited, isArticulation, bridges);
        }
    }

    for (auto& bridge : bridges) {
        if (bridge.first > bridge.second) std::swap(bridge.first, bridge.second);
    }

    std::cout << "Мосты:\n";
    for (auto& u : bridges) {
        std::cout << (u.first + 1) << " - " << (u.second + 1) << "\n";
    }
    std::cout << "Точки сочленения:\n";
    for (int i = 0; i < n; ++i) {
        if (isArticulation[i]) {
            std::cout << (i + 1) << "\n";
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

void task3(const Graph* graph) {
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
    auto begin = std::chrono::high_resolution_clock::now();
    int n = graph->size();
    std::vector<std::vector<int>> matrix = graph->adjacencyMatrix();
    std::vector<int> key(n, INT_MAX); //Мин ребро для каждой вершины
    std::vector<int> parent(n,-1);
    std::vector<bool> inMST(n,false);
    int start = 0;
    key[start] = 0;
    parent[start] = -1;
    std::priority_queue<
        std::pair<int,int>,              //Тип элементов (что храним), вес - вершина
        std::vector<std::pair<int,int>>, //Контейнер (где храним)
        std::greater<std::pair<int,int>> //Компаратор
    > pq;
    pq.push({0, start});
    while(!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if(inMST[u]) continue;
        inMST[u] = true;

        for(int v = 0; v < n; ++v) {
            int weight = matrix[u][v];
            if(weight != 0 && !inMST[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                pq.push({key[v], v});
            }
        }
    }
    int totalWeight = 0;
    std::cout << "Остовное дерево (Прим):\n";
    for (int i = 1; i < n; ++i) {
        std::cout << parent[i] + 1 << " - " << i + 1 << " (вес: " << matrix[parent[i]][i] << ")\n";
        totalWeight += matrix[parent[i]][i];
    }
    std::cout << "Суммарный вес: " << totalWeight << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - begin;
    std::cout << "Время выполнения алгоритма Прима: " << elapsed.count() << " секунд\n";
}

void algorithmKruskal(const Graph* graph) {
    auto begin = std::chrono::high_resolution_clock::now();
    int n = graph->size();
    std::vector<std::vector<std::pair<int, int>>> list = graph->weightedAdjacencyList();
    std::vector<std::vector<int>> edgeList;
    std::set<std::pair<int, int>> addedEdges;// чтобы отлавливать дубликаты
    for (int from = 0; from < n; ++from) {
        for (auto [to, weight] : list[from]) {
            int u = std::min(from, to);
            int v = std::max(from, to);
            if (addedEdges.count({u, v}) == 0) {
                edgeList.push_back({from, to, weight});
                addedEdges.insert({u, v});
            }
        }
    }
    std::sort(edgeList.begin(),edgeList.end(),[](const std::vector<int>& a, const std::vector<int>& b){ return a[2] < b[2];});
    std::vector<int> parent(n), rank(n,0);
    for(int i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](int x) { //Уходим к корню множества
        while (x != parent[x])
            x = parent[x] = parent[parent[x]];
        return x;
    };
    auto unite = [&](int x, int y) { //Лямбда функция для объединения 2 множеств
        int rx = find(x), ry = find(y);
        if (rx == ry) return false;
        if (rank[rx] < rank[ry]) std::swap(rx, ry);
        parent[ry] = rx;
        if (rank[rx] == rank[ry]) ++rank[rx];
        return true;
    };
    int totalWeight = 0;
    std::cout << "\nОстовное дерево (Краскал):\n";
    for (const auto& edge : edgeList) {
        int u = edge[0], v = edge[1], w = edge[2];
        if (unite(u, v)) {
            std::cout << u + 1 << " - " << v + 1 << " (вес: " << w << ")\n";
            totalWeight += w;
        }
    }
    std::cout << "Суммарный вес: " << totalWeight << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - begin;
    std::cout << "Время выполнения алгоритма Краскала: " << elapsed.count() << " секунд\n";
}

void task9(const Graph* graph) {
    int n = graph->size();
    //Проверка связный ли граф
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> adj = graph->adjacencyList();
    std::vector<std::vector<int>> components;
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            std::vector<int> component;
            dfs(i, adj, visited, component);
            components.push_back(component);
        }
    }
    if(components.size() > 1) {
        std::cout << "Graph is not connected";
        return;
    }

    //Алгоритм Прима
    algorithmPrim(graph);

    //Алгоритм Краскала
    algorithmKruskal(graph);

}

void task4(const Graph* graph) {
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
            if(matrix[i][j] == 1)
                degree[i]++;
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

void task5(const Graph* graph) {
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
            std::cerr << "Ошибка при открытии файла\n";
            rows = cols = 0;
            return;
        }

        std::string firstLine;
        std::getline(in, firstLine);

        std::istringstream iss(firstLine);
        std::vector<int> numbers;
        int num;
        while (iss >> num) {
            numbers.push_back(num);
        }

        if (numbers.size() == 2) {
            // Формат с размерами
            rows = numbers[0];
            cols = numbers[1];
            heights.resize(rows, std::vector<int>(cols));

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    in >> heights[i][j];
                }
            }
        } else {
            // задание 12
            heights.push_back(numbers);

            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                heights.push_back(parseLineToInts(line));
            }

            rows = static_cast<int>(heights.size());
            cols = rows > 0 ? static_cast<int>(heights[0].size()) : 0;
        }
    }

    std::pair<int,int> size() const {
        return {rows, cols};
    }

    int operator()(int i, int j) const {
        return heights[i][j];
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

std::vector<Point> buildPath(const std::vector<std::vector<Point>>& parent, Point start, Point end) {
    std::vector<Point> path;
    Point cur = end;
    while(!(cur == start)) {
        path.push_back(cur);
        cur = parent[cur.x][cur.y];
    }
    path.push_back(start);
    std::reverse(path.begin(),path.end());
    return path;
}

void task6(const Map& map, Point start, Point end) {
    auto [rows, cols] = map.size();
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<Point>> parent(rows, std::vector<Point>(cols, {-1, -1}));

    std::queue<Point> q;
    q.push(start);
    visited[start.x][start.y] = true;

    bool found = false;
    while (!q.empty()) {
        Point cur = q.front(); q.pop();
        if (cur == end) {
            found = true;
            break;
        }
        for (auto [nx, ny] : map.neighbors(cur.x, cur.y)) {
            if (!visited[nx][ny]) {
                visited[nx][ny] = true;
                parent[nx][ny] = cur;
                q.push({nx, ny});
            }
        }
    }

    if (!found) {
        std::cout << "No path found.\n";
        return;
    }

    auto path = buildPath(parent, start, end);

    std::cout << "Length of path from (" << start.x << ", " << start.y << ") to (" << end.x << ", " << end.y << "): " << (int)path.size() - 1 << "\n";
    std::cout << "Path:\n[";
    for (size_t i = 0; i < path.size(); ++i) {
        std::cout << "(" << path[i].x << ", " << path[i].y << ")";
        if (i + 1 < path.size()) std::cout << ", ";
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

void task7(const Graph* graph) {
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

void task10(const Graph* graph) {
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

void task11(const Graph* graph) {
    const int INF = 1e9;
    int n = graph->size();
    std::vector<std::vector<std::pair<int, int>>> list = graph->weightedAdjacencyList();
    std::vector<std::vector<int>> edgeList;
    for (int from = 0; from < n; ++from) {
        for (auto [to, weight] : list[from]) {
            edgeList.push_back({from, to, weight});
        }
    }
    std::cout << "please enter the vertex: ";
    int start = 0;
    std::cin >> start;
    --start;

    std::vector<int> dist(n, INF);
    dist[start] = 0;

    for(int i = 0; i < n -1; ++i) {
        for(auto &edge:edgeList) {
            int u = edge[0], v = edge[1], w = edge[2];
            if(dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    bool hasNegativeCycle = false;
    for (const auto& edge : edgeList) {
        int u = edge[0], v = edge[1], w = edge[2];
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            hasNegativeCycle = true;
            break;
        }
    }
    std::cout << "Shotest paths lengths from " << start << ":\n{";
    for(int i = 0; i < n; ++i) {
        if(dist[i] == INF) std::cout << i+1 << ": " << "+Infinity";
        else std::cout << i+1 << ": " << dist[i];

        if(i != n -1)
            std::cout << ", ";
    }
    std::cout << "}";
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
)    {
    auto [rows, cols] = map.size();
    constexpr double INF = 1e9;

    std::vector<std::vector<double>> g_score(rows, std::vector<double>(cols,INF));  //стоимость пути от старта до точки point
    std::vector<std::vector<Point>> came_from(rows, std::vector<Point>(cols, {-1,-1})); //откуда мы пришли в point, чтобы восстановить путь
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

    using PQElement = std::pair<double, Point>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<>> open;

    g_score[start.x][start.y] = 0;
    open.emplace(heuristic(start,end),start);

    while(!open.empty()) {
        Point current = open.top().second;
        open.pop();

        if(visited[current.x][current.y]) continue;
        visited[current.x][current.y] = true;
        if(current == end) return {reconstruct_path(came_from, current), g_score[current.x][current.y]};

        for(auto [nx, ny] : map.neighbors(current.x, current.y)) {
            Point neighbor{nx,ny};
            int cost = std::abs(nx - current.x) + std::abs(ny - current.y)
                     + std::abs(map(nx, ny) - map(current.x, current.y));
            double tentative_g = g_score[current.x][current.y] + cost;

            if (tentative_g < g_score[nx][ny]) {
                g_score[nx][ny] = tentative_g;
                came_from[nx][ny] = current;
                double f = tentative_g + heuristic(neighbor, end);
                open.emplace(f, neighbor);
            }
        }
    }
    return {};
}

void task12(
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

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);


    // const std::string path = "C:/Users/tutir/Graph tests/task11/list_of_adjacency_t11_005.txt";
    // MatrixGraph g(path);
    // task11(&g);

    /*
    Map map("C:/Users/tutir/Downloads/Graph tests/task6/maze_t6_001.txt");
    Point start{1, 5};
    Point end{3, 3};
    task6(map,start,end);
    */

    // Map map("C:/Users/tutir/Graph tests/task12/map_001.txt");
    // Point start{14,6}, end{14,13};
    // task12(map,start,end,manhattan);

    return 0;
}