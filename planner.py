"""
motosim — MPC trajectory planner (v1.1).

Replaces the reactive target_lean_from_cones() with a receding-horizon
optimal-control problem solved every perception cycle.

Model: kinematic bicycle (standard low-speed AV planner model).
  State    : x (fwd), y (lateral), heading
  Control  : steer_angle
  Fixed    : forward speed = v_cruise (separate P controller holds this)

Horizon: 2.0 seconds, 20 steps (dt = 0.1s).

Cost function (minimize):
  w_goal * ||pos_terminal - goal||^2                 pull trajectory to goal
  w_lane * sum(y^2 over trajectory)                  stay near centerline
  w_steer * sum(steer^2)                             smooth steering
  w_rate * sum((steer[k+1] - steer[k])^2)            no jittery steer
  w_obs * sum(1 / (dist_to_obstacle^2 + eps))        soft obstacle repulsion

Constraints:
  dynamics (kinematic bicycle)                       equality per step
  steer_angle in [-MAX_STEER, +MAX_STEER]            box constraint on U

Why soft obstacle costs instead of hard constraints: ipopt is much more robust
to soft costs. Hard constraints can produce "infeasible" solver returns when
an obstacle gets very close; soft costs degrade gracefully (worse trajectory
but always a solution).

Dependencies:
  pip install casadi
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np


# -- Default problem parameters --
HORIZON_STEPS   = 20
DT              = 0.1       # seconds per step (horizon = 2.0 s total)
WHEELBASE       = 1.41      # CBR500R wheelbase
MAX_STEER       = 0.10      # rad — balance between maneuverability and
                            # stability. At 2 m/s this gives R ~= 14m but
                            # the slower speed means more arc traversed per
                            # meter forward, so lateral displacement adds up.

# Cost weights (tuned for CBR500R at 3 m/s with ~2m cone offsets)
W_GOAL          = 5.0       # terminal goal reach
W_LANE          = 3.0       # stronger pull to centerline (was 2.0)
W_HEADING_GOAL  = 3.0       # stronger terminal heading match (was 1.0)
W_STEER         = 2.0       # higher control cost = less aggressive steering
W_STEER_RATE    = 25.0      # much smoother steering transitions (was 5.0)
W_OBSTACLE_SOFT = 3.0       # background inverse-distance repulsion (1/d²)
W_OBSTACLE_HARD = 100.0     # exterior-penalty barrier inside SAFE_RADIUS
                            # Softened from 500 — too stiff caused solver to
                            # trade forward progress for distance from cone.
OBSTACLE_EPS    = 0.5       # (m^2) softening to avoid divide-by-zero
SAFE_RADIUS     = 1.5       # (m) clearance zone — strong penalty inside
STEER_RATE_MAX  = 0.025     # rad per step (0.25 rad/s at 10 Hz) — tighter
                            # to prevent planner from alternating max-left/max-right
                            # between consecutive re-plans.

MAX_OBSTACLES   = 8         # fixed slot count for the NLP parameters


@dataclass
class PlanResult:
    """Output of one MPC solve."""
    success: bool
    steer_cmd: float                      # first control to apply [rad]
    trajectory_xy: np.ndarray             # (N+1, 2) planned (x, y)
    trajectory_heading: np.ndarray        # (N+1,) planned heading
    solve_time_s: float
    cost: float


class MPCPlanner:
    """
    Receding-horizon planner for the motorcycle.

    Usage:
        planner = MPCPlanner()
        result = planner.solve(
            state=(bike_x, bike_y, bike_heading),
            goal=(goal_x, goal_y),
            obstacles=[(cx1, cy1), (cx2, cy2), ...],
            speed=5.0,
        )
        steer_cmd = result.steer_cmd
    """

    def __init__(self,
                 N: int = HORIZON_STEPS,
                 dt: float = DT,
                 wheelbase: float = WHEELBASE,
                 max_steer: float = MAX_STEER,
                 max_obstacles: int = MAX_OBSTACLES,
                 verbose: bool = False):
        self.N = N
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_obstacles = max_obstacles

        # Build the NLP once (symbolic). Reused for every solve with different params.
        self._build_nlp(verbose=verbose)

        # Warm start: previous solution to seed next solve
        self._last_solution: Optional[np.ndarray] = None

    def _build_nlp(self, verbose: bool) -> None:
        N = self.N

        # Decision variables (all time steps)
        X = ca.MX.sym("X", 3, N + 1)      # [x, y, heading] over N+1 steps
        U = ca.MX.sym("U", 1, N)          # [steer] over N steps

        # Parameters (set per-solve)
        x0 = ca.MX.sym("x0", 3)                        # initial [x, y, heading]
        goal = ca.MX.sym("goal", 3)                    # terminal [x, y, heading]
        obstacles = ca.MX.sym("obs", 2, self.max_obstacles)
        speed = ca.MX.sym("v")

        # Equality constraints: initial state + dynamics propagation
        g = [X[:, 0] - x0]
        for k in range(N):
            # Kinematic bicycle: midpoint integration
            heading_k = X[2, k]
            steer_k = U[0, k]
            dx = speed * ca.cos(heading_k)
            dy = speed * ca.sin(heading_k)
            dh = speed * ca.tan(steer_k) / self.wheelbase

            x_next = X[0, k] + dx * self.dt
            y_next = X[1, k] + dy * self.dt
            h_next = X[2, k] + dh * self.dt
            g.append(X[:, k + 1] - ca.vertcat(x_next, y_next, h_next))

        # Steer-rate inequality constraints (|U[k+1] - U[k]| <= STEER_RATE_MAX)
        # Split into two one-sided inequalities per step.
        g_rate_upper = []
        g_rate_lower = []
        for k in range(N - 1):
            g_rate_upper.append(U[0, k + 1] - U[0, k] - STEER_RATE_MAX)  # <= 0
            g_rate_lower.append(-(U[0, k + 1] - U[0, k]) - STEER_RATE_MAX)  # <= 0

        # Cost
        cost = ca.MX(0)

        # Terminal goal (position + heading)
        cost += W_GOAL * ((X[0, -1] - goal[0]) ** 2 + (X[1, -1] - goal[1]) ** 2)
        cost += W_HEADING_GOAL * (X[2, -1] - goal[2]) ** 2

        # Running costs
        for k in range(N):
            # Stay near centerline (lane keeping)
            cost += W_LANE * X[1, k] ** 2
            # Steer effort
            cost += W_STEER * U[0, k] ** 2
            # Steer rate (smoothness between consecutive steps)
            if k > 0:
                cost += W_STEER_RATE * (U[0, k] - U[0, k - 1]) ** 2
            # Obstacle cost: soft inverse-distance (always on, small) + hard
            # exterior barrier (only inside SAFE_RADIUS, very strong).
            # The hard barrier is what actually forces the bike wide of the
            # cone; the soft term keeps the planner from skimming too close
            # when it has room to spare.
            for i in range(self.max_obstacles):
                d2 = (X[0, k] - obstacles[0, i]) ** 2 + (X[1, k] - obstacles[1, i]) ** 2
                cost += W_OBSTACLE_SOFT / (d2 + OBSTACLE_EPS)
                penetration = ca.fmax(0, SAFE_RADIUS ** 2 - d2)
                cost += W_OBSTACLE_HARD * penetration ** 2

        # Flatten decision vars
        dec = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params = ca.vertcat(x0, goal, ca.reshape(obstacles, -1, 1), speed)

        # Variable bounds
        n_x = 3 * (N + 1)
        n_u = N
        lbx = [-ca.inf] * n_x + [-self.max_steer] * n_u
        ubx = [ ca.inf] * n_x + [ self.max_steer] * n_u

        # Constraint bounds: first 3*(N+1) are equality (dynamics),
        # then 2*(N-1) are one-sided inequality (steer rate <= max)
        n_g_eq = 3 * (N + 1)
        n_g_ineq = 2 * (N - 1)
        lbg = [0.0] * n_g_eq + [-ca.inf] * n_g_ineq
        ubg = [0.0] * n_g_eq + [0.0] * n_g_ineq

        nlp = {
            "x": dec,
            "p": params,
            "f": cost,
            "g": ca.vertcat(*g, *g_rate_upper, *g_rate_lower),
        }
        ipopt_opts = {
            "ipopt.print_level": 5 if verbose else 0,
            "ipopt.sb": "yes",
            "print_time": 1 if verbose else 0,
            "ipopt.max_iter": 50,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.linear_solver": "mumps",
        }
        self._solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = lbg
        self._ubg = ubg
        self._n_x = n_x
        self._n_u = n_u

    def solve(self,
              state: Tuple[float, float, float],
              goal: Tuple[float, float, float],
              obstacles: List[Tuple[float, float]],
              speed: float) -> PlanResult:
        """Run one MPC solve. Returns PlanResult with first steer command."""
        import time

        # Pad obstacle list to max_obstacles (fill with far-away virtual cones)
        obs_padded = list(obstacles[:self.max_obstacles])
        while len(obs_padded) < self.max_obstacles:
            obs_padded.append((1000.0, 1000.0))  # far enough that repulsion ~ 0
        obs_flat = np.array(obs_padded, dtype=float).T.reshape(-1, order="F")

        p = np.concatenate([
            np.array(state, dtype=float),
            np.array(goal, dtype=float),
            obs_flat,
            np.array([speed], dtype=float),
        ])

        # Warm start from previous solution, or zeros
        if self._last_solution is not None:
            x0_guess = self._last_solution
        else:
            x0_guess = np.zeros(self._n_x + self._n_u)

        t_start = time.time()
        try:
            sol = self._solver(x0=x0_guess, p=p, lbx=self._lbx, ubx=self._ubx,
                                lbg=self._lbg, ubg=self._ubg)
            dec = np.array(sol["x"]).flatten()
            self._last_solution = dec

            X_opt = dec[: self._n_x].reshape(3, self.N + 1, order="F")
            U_opt = dec[self._n_x: self._n_x + self._n_u]

            return PlanResult(
                success=True,
                steer_cmd=float(U_opt[0]),
                trajectory_xy=X_opt[:2, :].T,          # (N+1, 2)
                trajectory_heading=X_opt[2, :],        # (N+1,)
                solve_time_s=time.time() - t_start,
                cost=float(sol["f"]),
            )
        except Exception as e:
            # Solver failed: fall back to zero steer (go straight).
            return PlanResult(
                success=False,
                steer_cmd=0.0,
                trajectory_xy=np.zeros((self.N + 1, 2)),
                trajectory_heading=np.zeros(self.N + 1),
                solve_time_s=time.time() - t_start,
                cost=float("inf"),
            )


def yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw (heading about world z) from a chassis quat [w, x, y, z]."""
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz)))
