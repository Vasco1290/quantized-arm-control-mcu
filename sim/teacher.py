"""The teacher: a classical controller that the student network will later copy.

Three parts, run 50 times a second:
  1. Inverse kinematics: target position -> the joint angles that put the hand there.
  2. PD: a spring pulls each joint toward its goal angle, a brake slows it down.
  3. Gravity compensation: add exactly the torque gravity is taking away.
"""
import numpy as np

from exo_arm_env import L1, L2

# Spring stiffness (N·m per radian of error) and brake strength (N·m per rad/s).
# Chosen by a gain scan (see NOTES.md): 100% success, reached in ~0.37 s.
# A brake twice this strong fails, because the command is held for 0.02 s at a time.
KP = np.array([60.0, 20.0])  # shoulder, elbow
KD = np.array([8.0, 2.0])


def inverse_kinematics(x, z):
    """Hand position (x, z) relative to the shoulder -> joint angles (q1, q2) in radians.

    Uses the elbow-forward solution (q2 >= 0), the only one our elbow can reach.
    """
    # Step 1, elbow: the further away the target, the straighter the elbow.
    # Law of cosines: links L1, L2 and the shoulder-to-target distance form a triangle.
    cos_q2 = (x * x + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = np.arccos(np.clip(cos_q2, -1.0, 1.0))  # clip guards against rounding just past +-1

    # Step 2, shoulder: point the upper arm at the target, then lean back by the extra
    # angle the bent elbow adds on top.
    direction = np.arctan2(x, -z)  # angle of the shoulder->target line, measured from straight down
    bend_effect = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = direction - bend_effect

    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi  # wrap into -180..180 deg (avoids the -230 deg bug)
    return np.array([q1, q2])


def teacher_policy(obs, env):
    """PD + gravity compensation. Allowed to read the true simulator state from env."""
    target = obs[4:6]                      # target (x, z), from the observation
    q_goal = inverse_kinematics(*target)

    q = env.data.qpos                      # current joint angles (true state)
    qdot = env.data.qvel                   # current joint speeds (true state)
    gravity = env.data.qfrc_bias           # torque needed to cancel gravity (+ small speed effects)

    torque = KP * (q_goal - q) - KD * qdot + gravity
    gear = env.model.actuator_gear[:, 0]   # 30 shoulder, 15 elbow
    # torque -> motor command, limited to what the motor can do. The student copies
    # these numbers, so they must be the commands that were really applied.
    return np.clip(torque / gear, -1.0, 1.0)
