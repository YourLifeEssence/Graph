// #ifndef TASK19_H
// #define TASK19_H
//
// #include <limits>
// #include <random>
// #include <numeric>
// #include <thread>
// #include <mutex>
// #include "graph.h"
//
// constexpr int INF = std::numeric_limits<int>::max();
//
// class ACO {
// public:
//     ACO(const Graph& g, int ants, int iterations, double a, double b, double evap, double q);
//     void run();
// private:
//     const Graph& graph;
//     int n;
//     int numAnts;
//     int maxIterations;
//     double alpha, beta, evaporationRate, Q;
//
//     std::vector<std::vector<int>> distance;
//     std::vector<std::vector<double>> pheromone;
//
//     struct Ant {
//         std::vector<int> tour;
//         std::vector<bool> visited;
//         int tourLength = 0;
//     };
//
//     std::vector<Ant> ants;
//     std::vector<int> bestTour;
//     int bestTourLength;
//
//     std::mt19937 rng;
//     std::mutex mutex;
// };
//
// inline ACO::ACO(const Graph &g, int ants, int iterations, double a, double b, double evap, double q) :
//     graph(g), numAnts(ants), maxIterations(iterations), alpha(a), beta(b), evaporationRate(evap), Q(q), rng(std::random_device{}())
// {
//     n = graph.size();
//
//     pheromone.resize(n,std::vector<double>(n,1.0));
//
//     distance.resize(n, std::vector<int>(n, INF));
//     auto adj = graph.weightedAdjacencyList();
//     for (int i = 0; i < n; ++i) {
//         for (auto [j, w] : adj[i]) {
//             distance[i][j] = w;
//         }
//     }
// }
//
// void ACO::run () {
//     const size_t num_ants = std::thread::hardware_concurrency();
//     const size_t n = graph.size();
//     auto matrix = graph.adjacencyMatrix();
//
//     std::vector<std::vector<double>> pheromones(n, std::vector<double>(n, 1.0));
//
//     std::vector<size_t> best_path;
//     double best_path_length = std::numeric_limits<double>::max();
//     std::mutex best_path_mutex;
//
//     // Функция для одного муравья
//     auto ant_run = [&](size_t ant_id, size_t current_iter) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//
//         std::vector<size_t> path;
//         path.reserve(n);
//         std::uniform_int_distribution<size_t> start_dist(0, n-1);
//         size_t current = start_dist(gen);
//         path.push_back(current);
//
//         std::vector<bool> visited(n, false);
//         visited[current] = true;
//
//         for (size_t step = 1; step < n; ++step) {
//             std::vector<size_t> candidates;
//             std::vector<double> probabilities;
//
//             for (size_t next = 0; next < n; ++next) {
//                 if (!visited[next] && matrix[current][next] > 0) {
//                     candidates.push_back(next);
//                     double pheromone = pow(pheromones[current][next], alpha);
//                     double visibility = pow(1.0 / matrix[current][next], beta);
//                     probabilities.push_back(pheromone * visibility);
//                 }
//             }
//
//             if (candidates.empty()) break;
//
//             double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
//             if (sum > 0) {
//                 for (auto& p : probabilities) p /= sum;
//             }
//
//             std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
//             size_t next = candidates[dist(gen)];
//
//             path.push_back(next);
//             visited[next] = true;
//             current = next;
//         }
//
//         if (path.size() == n && matrix[path.back()][path.front()] > 0) {
//             double path_length = 0.0;
//             for (size_t i = 0; i < n; ++i) {
//                 path_length += matrix[path[i]][path[(i+1)%n]];
//             }
//
//             std::lock_guard<std::mutex> lock(best_path_mutex);
//             if (path_length < best_path_length) {
//                 best_path_length = path_length;
//                 best_path = path;
//                 std::cout << "Iteration " << current_iter << ": New best path = " << best_path_length << "\n";
//             }
//
//             for (size_t i = 0; i < n; ++i) {
//                 size_t from = path[i];
//                 size_t to = path[(i+1)%n];
//                 pheromones[from][to] += Q / path_length;
//                 if (!graph.isDirected()) {
//                     pheromones[to][from] += Q / path_length;
//                 }
//             }
//         }
//     };
//
//     for (size_t iter = 0; iter < maxIterations; ++iter) {
//         std::vector<std::thread> ants;
//         for (size_t ant = 0; ant < num_ants; ++ant) {
//             ants.emplace_back(ant_run, ant, iter + 1);
//         }
//
//         for (auto& ant : ants) ant.join();
//
//         for (auto& row : pheromones) {
//             for (auto& p : row) {
//                 p *= (1.0 - evaporationRate);
//             }
//         }
//     }
//
//     if (!best_path.empty()) {
//         std::cout << "\nBest path found (length = " << best_path_length << "):\n";
//         for (size_t v : best_path) std::cout << v + 1 << " ";
//         std::cout << best_path[0] + 1 << "\n";
//     } else {
//         std::cout << "\nNo valid Hamiltonian cycle found!\n";
//     }
// }
//
// #endif // TASK19_H
