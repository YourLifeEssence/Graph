#include "task12.h"
#include "task13.h"
#include <windows.h>
#include <iostream>
#include <memory>
#include <string>

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::cout << "Graph algorithms\n"
              << "Choose a task (0 - exit, 1-13): ";

    int task = 0;
    if (!(std::cin >> task) || task == 0) {
        return 0;
    }
    if (task < 1 || task > 13) {
        std::cerr << "Unknown task.\n";
        return 1;
    }

    try {
        if (task == 5 || task == 11) {
            std::string path;
            int startRow = 0;
            int startColumn = 0;
            int endRow = 0;
            int endColumn = 0;

            std::cout << "Map file path: ";
            std::cin >> path;
            Map map(path);
            std::cout << "Start coordinates (row col): ";
            std::cin >> startRow >> startColumn;
            std::cout << "End coordinates (row col): ";
            std::cin >> endRow >> endColumn;

            if (task == 5) {
                task5(map, {startRow, startColumn}, {endRow, endColumn});
            } else {
                int heuristic = 1;
                std::cout << "Heuristic (1 Manhattan, 2 Chebyshev, 3 Euclidean): ";
                std::cin >> heuristic;
                if (heuristic == 1) {
                    task11(map, {startRow, startColumn}, {endRow, endColumn}, manhattan);
                } else if (heuristic == 2) {
                    task11(map, {startRow, startColumn}, {endRow, endColumn}, chebyshev);
                } else if (heuristic == 3) {
                    task11(map, {startRow, startColumn}, {endRow, endColumn}, euclidean);
                } else {
                    std::cerr << "Unknown heuristic.\n";
                    return 1;
                }
            }
            return 0;
        }

        std::string path;
        int format = 0;
        std::cout << "Input file path: ";
        std::cin >> path;
        std::cout << "Input format:\n"
                  << "  1 - adjacency matrix\n"
                  << "  2 - adjacency list\n"
                  << "  3 - edge list\n"
                  << "Format: ";
        std::cin >> format;
        if (format < 1 || format > 3) {
            std::cerr << "Unknown input format.\n";
            return 1;
        }

        std::unique_ptr<Graph> graph;
        if (format == 2) {
            graph = std::make_unique<ListGraph>(path);
        } else {
            // MatrixGraph accepts both an adjacency matrix and an edge list.
            graph = std::make_unique<MatrixGraph>(path);
        }

        switch (task) {
        case 1:
            task1(graph.get());
            break;
        case 2:
            task2(graph.get());
            break;
        case 3:
            task3(graph.get());
            break;
        case 4:
            task4(graph.get());
            break;
        case 6:
            task6(graph.get());
            break;
        case 7:
            task7(graph.get());
            break;
        case 8:
            task8(graph.get());
            break;
        case 9:
            task9(graph.get());
            break;
        case 10:
            task10(graph.get());
            break;
        case 12:
            task12(graph.get());
            break;
        case 13:
            task13(graph.get());
            break;
        default:
            break;
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
