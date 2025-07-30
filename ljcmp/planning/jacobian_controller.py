import numpy as np
import time

from srmt.constraints.constraints import ConstraintBase


class TwoArmJacobianController():
    """
    Jacobian-based planner for dual-arm (leader-follower) motion planning.
    The leader arm is controlled to follow a desired end-effector trajectory,
    and the follower arm is controlled to maintain a relative pose or constraint.
    """

    def __init__(
        self,
        fk_func_leader,
        jacobian_func_leader,
        fk_func_follower,
        jacobian_func_follower,
        constraint: ConstraintBase = None,
        validity_fn=None,
        step_size=0.01,
        damping=1e-4,
        max_iters=100,
        tol=1e-4,
        follower_relative_pose=None,
    ):
        """
        Args:
            fk_func_leader: function(q_leader) -> x_leader, returns end-effector pose for leader arm
            jacobian_func_leader: function(q_leader) -> J_leader, returns Jacobian for leader arm
            fk_func_follower: function(q_follower) -> x_follower, returns end-effector pose for follower arm
            jacobian_func_follower: function(q_follower) -> J_follower, returns Jacobian for follower arm
            constraint: optional, constraint object for joint configurations
            validity_fn: optional, function(q_leader, q_follower) -> bool, returns True if configuration is valid
            step_size: float, step size for each iteration
            damping: float, damping factor for pseudo-inverse
            max_iters: int, maximum number of iterations
            tol: float, tolerance for convergence in task space
            follower_relative_pose: np.ndarray or None, desired relative pose of follower w.r.t. leader
        """
        self.fk_func_leader = fk_func_leader
        self.jacobian_func_leader = jacobian_func_leader
        self.fk_func_follower = fk_func_follower
        self.jacobian_func_follower = jacobian_func_follower
        self.step_size = step_size
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self.constraint = constraint
        self.validity_fn = validity_fn
        self.start_q_leader = None
        self.start_q_follower = None
        self.goal_pose_leader = None
        self.follower_relative_pose = follower_relative_pose  # desired follower pose w.r.t. leader

    def set_start(self, q_leader, q_follower):
        self.start_q_leader = np.copy(q_leader)
        self.start_q_follower = np.copy(q_follower)

    def set_goal(self, x_goal_leader, follower_relative_pose=None):
        self.goal_pose_leader = np.copy(x_goal_leader)
        if follower_relative_pose is not None:
            self.follower_relative_pose = np.copy(follower_relative_pose)

    def solve(self, max_time=None):
        """
        Plans a path for both leader and follower arms.
        The leader arm is controlled to reach the goal pose.
        The follower arm is controlled to maintain a relative pose w.r.t. the leader.

        Args:
            max_time: float or None, maximum allowed time in seconds for planning. If None, uses max_iters.

        Returns:
            q_path_leader: list of np.ndarray, sequence of leader joint configurations
            q_path_follower: list of np.ndarray, sequence of follower joint configurations
            success: bool, whether the goal was reached
        """
        if self.start_q_leader is None or self.start_q_follower is None:
            raise ValueError("Start configurations for both arms must be set")
        if self.goal_pose_leader is None:
            raise ValueError("Goal pose for leader arm must be set")

        q_leader = np.copy(self.start_q_leader)
        q_follower = np.copy(self.start_q_follower)
        q_path_leader = [np.copy(q_leader)]
        q_path_follower = [np.copy(q_follower)]

        start_time = time.time()
        iter_idx = 0
        while True:
            # Check time limit
            if max_time is not None:
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    break
            else:
                if iter_idx >= self.max_iters:
                    break

            # Leader arm control
            x_leader = self.fk_func_leader(q_leader)
            error_leader = self.goal_pose_leader - x_leader
            if np.linalg.norm(error_leader) < self.tol:
                return q_path_leader, q_path_follower, True

            J_leader = self.jacobian_func_leader(q_leader)
            JT_leader = J_leader.T
            JJ_leader = J_leader @ JT_leader
            lambda_I_leader = self.damping * np.eye(JJ_leader.shape[0])
            J_pinv_leader = JT_leader @ np.linalg.inv(JJ_leader + lambda_I_leader)
            dq_leader = self.step_size * (J_pinv_leader @ error_leader)
            q_leader_new = q_leader + dq_leader

            # Follower arm control: maintain relative pose w.r.t. leader
            if self.follower_relative_pose is not None:
                x_leader_new = self.fk_func_leader(q_leader_new)
                x_follower = self.fk_func_follower(q_follower)
                desired_x_follower = x_leader_new + self.follower_relative_pose
                error_follower = desired_x_follower - x_follower
                J_follower = self.jacobian_func_follower(q_follower)
                JT_follower = J_follower.T
                JJ_follower = J_follower @ JT_follower
                lambda_I_follower = self.damping * np.eye(JJ_follower.shape[0])
                J_pinv_follower = JT_follower @ np.linalg.inv(JJ_follower + lambda_I_follower)
                dq_follower = self.step_size * (J_pinv_follower @ error_follower)
                q_follower_new = q_follower + dq_follower
            else:
                q_follower_new = np.copy(q_follower)

            # Optionally project to constraints
            if self.constraint is not None:
                q_new = np.concatenate([q_leader_new, q_follower_new])
                if not self.constraint.is_valid(q_new):
                    break

            if self.validity_fn is not None:
                q_new = np.concatenate([q_leader_new, q_follower_new])
                if not self.validity_fn(q_new):
                    break


            q_leader = q_leader_new
            q_follower = q_follower_new
            q_path_leader.append(np.copy(q_leader))
            q_path_follower.append(np.copy(q_follower))

            iter_idx += 1

        return q_path_leader, q_path_follower, False


