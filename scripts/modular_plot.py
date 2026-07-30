"""Thin orchestrator for the modular scripts/plotting/ package.

Same interactive behavior as the original plot_joint_data.py (folder/
experiment selection, wbc_only prompt), same output tree under images/ —
just built from plotting.loaders + plotting.sections instead of one
monolithic file. Kept as a separate script for now (not replacing
plot_joint_data.py) so the two can be compared before switching over.
"""

from plotting import loaders
from plotting.sections import (
    com_zmp,
    contact_forces,
    ekf,
    execution_times,
    feedback,
    feet,
    joint_limits,
    joint_tracking,
    task_orientation,
    wbc_solutions,
)

if __name__ == '__main__':
    expNumber = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if expNumber == '0':
        folder = '/tmp/robot_logs'
    else:
        folder = 'experiments/experiment_' + expNumber + '/robot_logs'

    wbc_only = input("Enter 1 to plot only from WBC activation onward, "
                      "or press Enter to plot all data: ").strip() == '1'

    ctx = loaders.load_context(folder, wbc_only)

    joint_tracking.run(ctx)
    wbc_solutions.run(ctx)
    transitions_by_foot = contact_forces.run(ctx)
    joint_limits.run(ctx, transitions_by_foot)
    com_zmp.run(ctx)
    feet.run(ctx)
    ekf.run(ctx)
    task_orientation.run(ctx)
    feedback.run(ctx)
    execution_times.run(ctx)

    print("Done. Plots written under images/.")