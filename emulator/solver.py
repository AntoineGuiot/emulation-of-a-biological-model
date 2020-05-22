from scipy.integrate import odeint

class Solver:
    def __init__(self, t):
        self.t = t

    def solve(self, function, y_0, args):
        solution = odeint(function, y_0, self.t, args=args)
        return solution

