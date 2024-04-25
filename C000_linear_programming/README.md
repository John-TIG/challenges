# Quadratic Programming Challenge

Quadratic Programming (QP) is a type of convex optimization problem where the objective function is quadratic, and the constraints are linear. The goal is to minimize a quadratic function subject to linear constraints. QP problems are widely used in various fields, such as finance, engineering, and machine learning.

The standard form of a QP problem is:

minimize    (1/2) * x^T * Q * x + c^T * x
subject to  A * x <= b
            x >= 0

where:
- x is the vector of variables to be optimized
- Q is a symmetric positive semidefinite matrix
- c is a vector of coefficients
- A is a matrix of coefficients for the linear inequality constraints
- b is a vector of constants for the linear inequality constraints

# Example

The following is an example of a Quadratic Programming challenge with configurable difficulty. Two parameters can be adjusted to vary the difficulty of the challenge instance:

- Parameter 1: `num_variables` is the number of variables in the QP problem. This determines the size of the problem.
- Parameter 2: `better_than_baseline` is the factor by which a solution must be better than the baseline value.

Consider an example `Challenge` instance with `num_variables=100` and `better_than_baseline=0.9`:

```python
Q = np.random.rand(100, 100)
Q = np.dot(Q, Q.T) # ensure Q is symmetric positive semidefinite
c = np.random.rand(100)
A = np.random.rand(50, 100)
b = np.random.rand(50) * 10
baseline_value = 1000.0
max_objective_value = baseline_value * better_than_baseline = 900.0
```

In this example, the problem has 100 variables, and the objective value of the solution must be less than or equal to 900.0 to be considered a valid solution.

The asymmetric property of this challenge lies in the fact that solving a QP problem of this size is computationally expensive, requiring advanced optimization algorithms. However, verifying a solution involves simple matrix-vector multiplications and checking linear constraints, which is comparatively less computationally intensive.

## Pseudo-random Instance Generation

To generate pseudo-random instances of the QP problem based on the difficulty parameters and a seed value, we can use the following method:

```python
def generateInstance(seed, num_variables, better_than_baseline):
    np.random.seed(seed)
    Q = np.random.rand(num_variables, num_variables)
    Q = np.dot(Q, Q.T) # ensure Q is symmetric positive semidefinite
    c = np.random.rand(num_variables)
    A = np.random.rand(num_variables // 2, num_variables)
    b = np.random.rand(num_variables // 2) * 10
    
    # Solve the QP problem using the Coordinate Descent algorithm to obtain the baseline value
    baseline_value = solveQP(Q, c, A, b, max_iterations=1000)
    max_objective_value = baseline_value * better_than_baseline
    
    return Q, c, A, b, max_objective_value
```

The `solveQP` function uses the Coordinate Descent algorithm to obtain the baseline value. The Coordinate Descent algorithm is a simple and intuitive optimization method that works by optimizing one variable at a time while keeping the others fixed. It iteratively cycles through the variables, updating each one to minimize the objective function with respect to that variable. This process continues until convergence or a maximum number of iterations is reached.

Here's the implementation of the Coordinate Descent algorithm:

```python
def coordinateDescent(Q, c, A, b, max_iterations=1000):
    num_variables = Q.shape[0]
    x = np.zeros(num_variables)
    
    for _ in range(max_iterations):
        for i in range(num_variables):
            # Compute the optimal value for variable i while keeping others fixed
            numerator = np.dot(Q[i, :i], x[:i]) + np.dot(Q[i, i+1:], x[i+1:]) + c[i]
            denominator = Q[i, i]
            x[i] = max(0, -numerator / denominator)
            
            # Project the solution back onto the feasible region
            if np.any(np.dot(A, x) > b):
                x[i] = 0
    
    return x
```

By using the Coordinate Descent algorithm instead of a standard solver, we obtain a baseline value that is likely to be less optimal than the one provided by an advanced solver. This gives participants more room to find better solutions while still maintaining the difficulty of the QP problem itself.

It's important to note that the introduction of the `max_objective_value` constraint may lead to instances with no solution. In such cases, the challenge is to prove that no solution exists that satisfies all the constraints, including the `max_objective_value` constraint.

## Our Challenge
In TIG, the baseline value is determined by solving the QP problem using the Coordinate Descent algorithm with a maximum number of iterations. The `better_than_baseline` parameter is then used to set the `max_objective_value` for the challenge. Participants must find a solution that achieves an objective value lower than this threshold or prove that no such solution exists.

The difficulty of the challenge can be adjusted by increasing the number of variables (`num_variables`) or by decreasing the `better_than_baseline` factor, requiring solutions to be closer to the optimal value.

Here's an example of how the challenge could be implemented in Python:

```python
import numpy as np

class QPChallenge:
    def __init__(self, seed, num_variables, better_than_baseline):
        self.Q, self.c, self.A, self.b, self.max_objective_value = self.generateInstance(seed, num_variables, better_than_baseline)
    
    def generateInstance(self, seed, num_variables, better_than_baseline):
        np.random.seed(seed)
        Q = np.random.rand(num_variables, num_variables)
        Q = np.dot(Q, Q.T) # ensure Q is symmetric positive semidefinite
        c = np.random.rand(num_variables)
        A = np.random.rand(num_variables // 2, num_variables)
        b = np.random.rand(num_variables // 2) * 10
        
        baseline_value = self.solveQP(Q, c, A, b, max_iterations=1000)
        max_objective_value = baseline_value * better_than_baseline
        
        return Q, c, A, b, max_objective_value
    
    def solveQP(self, Q, c, A, b, max_iterations=1000):
        x = coordinateDescent(Q, c, A, b, max_iterations)
        return self.objectiveValue(x)
    
    def verifySolution(self, x):
        if x is None:
            # Verify that no solution exists that satisfies all constraints
            return self.verifyNoSolution()
        else:
            # Verify that the solution satisfies all constraints and has an objective value less than max_objective_value
            return self.verifyFeasibleSolution(x) and self.objectiveValue(x) <= self.max_objective_value
    
    def verifyNoSolution(self):
        # Use a standard QP solver to check if a feasible solution exists
        # Return True if no feasible solution exists, False otherwise
        pass
    
    def verifyFeasibleSolution(self, x):
        # Check if the solution satisfies all constraints
        return np.all(x >= 0) and np.all(np.dot(self.A, x) <= self.b)
    
    def objectiveValue(self, x):
        # Compute the objective value of the solution
        return 0.5 * np.dot(x, np.dot(self.Q, x)) + np.dot(self.c, x)
```

In this implementation, the `QPChallenge` class encapsulates the QP problem instance and provides methods for generating pseudo-random instances, verifying solutions, and computing objective values. The `verifySolution` method checks if a submitted solution satisfies all constraints and has an objective value less than the `max_objective_value`, or if no solution exists that satisfies all constraints.
