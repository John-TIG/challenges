from c001_satisfiability.challenge import Challenge
import random

def solveChallenge(challenge: Challenge) -> list:
    # Schöning’s algorithm
    # algorithm must be deterministic given the same challenge
    random.seed(challenge.seed)
    variables = [random.random() < 0.5 for _ in range(challenge.difficulty.num_variables)]
    for i in range(challenge.difficulty.num_variables):
        # evaluate clauses and find any that are unsatisfied
        substituted = [
            [
                variables[lit - 1] if lit > 0 else not variables[-lit - 1]
                for lit in clause
            ] 
            for clause in challenge.clauses
        ]
        unsatisfied_clauses = [index for index, clause in enumerate(substituted) if not any(clause)]
        if len(unsatisfied_clauses) == 0:
            break
        # flip the value of a random variable from a random unsatisfied clause
        rand_unsatisfied_clause = random.choice(unsatisfied_clauses)
        rand_variable = abs(random.choice(challenge.clauses[rand_unsatisfied_clause]))
        variables[rand_variable - 1] = not variables[rand_variable - 1]
    return variables