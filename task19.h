#ifndef TASK19_H
#define TASK19_H

#include <limits>
#include <random>
#include <numeric>
#include "graph.h"
constexpr int INF = std::numeric_limits<int>::max();

class ACO {
public:
    ACO(const Graph& g, int ants, int iterations, double a, double b, double evap, double q);
    std::vector<int> run();
private:
    const Graph& graph;
    int n;
    int numAnts;
    int maxIterations;
    double alpha, beta, evaporationRate, Q; //параметры весов феромона, эвристики, испарение и интенсивность феромона

    std::vector<std::vector<int>> distance;
    std::vector<std::vector<double>> pheromone;

    struct Ant {
        std::vector<int> tour;
        std::vector<bool> visited;
        int tourLength = 0;
    };

    std::vector<Ant> ants;
    std::vector<int> bestTour;
    int bestTourLength;

    std::mt19937 rng;
};

inline ACO::ACO(const Graph &g, int ants, int iterations, double a, double b, double evap, double q) :
    graph(g), numAnts(ants), maxIterations(iterations), alpha(a), beta(b), evaporationRate(evap), Q(q), rng(std::random_device{}())
{
    n = graph.size();

    pheromone.resize(n,std::vector<double>(n,1.0));

    distance.resize(n, std::vector<int>(n, INF));
    auto adj = graph.weightedAdjacencyList();
    for (int i = 0; i < n; ++i) {
        for (auto [j, w] : adj[i]) {
            distance[i][j] = w;
        }
    }
}

inline std::vector<int> ACO::run() {
    bestTourLength = INF;
    bestTour.clear();

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        ants.clear();
        ants.resize(numAnts);

        for (int k = 0; k < numAnts; ++k) {
            int start = k % n;

            Ant ant;
            ant.visited.assign(n, false);
            ant.tour.push_back(start);
            ant.visited[start] = true;

            while (ant.tour.size() < n) {
                int from = ant.tour.back();
                std::vector<double> probabilities(n, 0.0);
                double sum = 0.0;

                for (int to = 0; to < n; ++to) {
                    if (!ant.visited[to] && distance[from][to] < INF) {
                        double tau = std::pow(pheromone[from][to], alpha);
                        double eta = std::pow(1.0 / distance[from][to], beta);
                        probabilities[to] = tau * eta;
                        sum += probabilities[to];
                    }
                }

                if (sum == 0.0) break;

                std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
                int next = dist(rng);
                if (ant.visited[next]) break; // защита от зацикливания

                ant.tour.push_back(next);
                ant.visited[next] = true;
                ant.tourLength += distance[from][next];
            }

            if (ant.tour.size() == static_cast<size_t>(n) && distance[ant.tour.back()][ant.tour.front()] < INF) {
                ant.tourLength += distance[ant.tour.back()][ant.tour.front()];
                ant.tour.push_back(ant.tour.front());
            }

            ants[k] = ant;

            if (ant.tourLength < bestTourLength && ant.tour.size() == static_cast<size_t>(n + 1)) {
                bestTourLength = ant.tourLength;
                bestTour = ant.tour;
            }
        }

        // Обновление феромона
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                pheromone[i][j] *= (1.0 - evaporationRate);

        for (const Ant& ant : ants) {
            if (ant.tour.size() != static_cast<size_t>(n + 1)) continue;
            for (int i = 0; i < n; ++i) {
                int from = ant.tour[i];
                int to = ant.tour[i + 1];
                pheromone[from][to] += Q / ant.tourLength;
                pheromone[to][from] += Q / ant.tourLength;
            }
        }
    }

    std::cout << "Length of shortest traveling salesman path is: " << bestTourLength << ".\n";
    std::cout << "Path:\n";

    for (size_t i = 0; i < bestTour.size() - 1; ++i) {
        int from = bestTour[i];
        int to = bestTour[i + 1];
        std::cout << from + 1 << "-" << to + 1 << " : " << distance[from][to] << "\n";
    }
    return bestTour;
}


#endif // TASK19_H