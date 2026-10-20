"""Scripted demonstrations in the sim, recorded through the real stack (teleop, recorder), and policies
scored closed loop through the same stack.

A task (task.Task) says where things go, what the arm does and what counts as success; motion streams
the arm's policy targets, navigator presses the recorder's buttons, and generate runs the episodes.
evaluate runs a policy (the policy runtime) on the generator's seeded layouts, outcome scores each attempt.
"""
