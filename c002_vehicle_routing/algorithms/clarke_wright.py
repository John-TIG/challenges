from c002_vehicle_routing.challenge import Challenge


def solveChallenge(challenge: Challenge) -> list:
    D = challenge.distance_matrix
    C = challenge.max_capacity
    N = challenge.difficulty.num_nodes
    # Clarke-Wright heuristic for node pairs based on their distances to depot
    # vs distance between each other 
    scores = sorted(
        (
            (D[i, 0] + D[0, j] - D[i, j], i, j)
            for i in range(1, N)
            for j in range(i + 1, N)
        ),
        reverse=True
    )
    routes = []
    all_visited = set()
    while len(scores):
        route = list(scores.pop(0)[1:])
        visited = set(route)
        demands = challenge.demands[route[0]] + challenge.demands[route[1]]
        for s, i, j in scores:
            if s < 0 or (i in visited) == (j in visited):
                continue
            node, append = None, False
            if i == route[0]:
                node, append = j, False
            elif i == route[-1]:
                node, append = j, True
            elif j == route[0]:
                node, append = i, False
            elif j == route[-1]:
                node, append = i, True

            if node is not None and demands + challenge.demands[node] <= C:
                if append:
                    route.append(node)
                else:
                    route.insert(0, node)
                visited.add(node)
                demands += challenge.demands[node]
        routes.append([0] + route + [0])
        all_visited |= visited
        scores = [
            s for s in scores 
            if s[1] not in all_visited and s[2] not in all_visited
        ]
    for node in set(range(1, N)) - all_visited:
        routes.append([0, node, 0])
    return routes