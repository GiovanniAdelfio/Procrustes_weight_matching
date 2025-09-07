import matplotlib.pyplot as plt

def plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever, train_acc_interp_proc, test_acc_interp_proc):

""" 
Plots 3 different interpolation train accuracies and 3 test accuracies on the same plot
"""

  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.plot(lambdas,
          train_acc_interp_naive,
          linestyle="dashed",
          color="tab:blue",
          alpha=0.5,
          linewidth=2,
          label="Train, naïve interp.")
  ax.plot(lambdas,
          test_acc_interp_naive,
          linestyle="dashed",
          color="tab:orange",
          alpha=0.5,
          linewidth=2,
          label="Test, naïve interp.")
  ax.plot(lambdas,
          train_acc_interp_clever,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, permutation weight matching")
  ax.plot(lambdas,
          test_acc_interp_clever,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, permutation weight matching")
  ax.plot(lambdas,
          train_acc_interp_proc,
          linestyle="dotted",
          color="tab:blue",
          linewidth=2,
          label="Train, weight matching via Procrustes")
  ax.plot(lambdas,
          test_acc_interp_proc,
          linestyle="dotted",
          color="tab:orange",
          linewidth=2,
          label="Test, weight matching via Procrustes")

  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Accuracy")
  # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2

  ax.set_title(f"Accuracy between the two models")
  ax.legend(loc="lower right", framealpha=0.5)
  fig.tight_layout()
  return fig


def plot_interp_acc_2(lambdas, train_acc_interp_naive, test_acc_interp_naive, train_acc_interp_proc, test_acc_interp_proc):
"""
Same as the previous one, just with only 2 train and test accuracy.
"""
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(lambdas,
          train_acc_interp_naive,
          linestyle="dashed",
          color="tab:blue",
          alpha=0.5,
          linewidth=2,
          label="Train, naïve interp.")
  ax.plot(lambdas,
          test_acc_interp_naive,
          linestyle="dashed",
          color="tab:orange",
          alpha=0.5,
          linewidth=2,
          label="Test, naïve interp.")
  ax.plot(lambdas,
          train_acc_interp_proc,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, Procrustes interp.")
  ax.plot(lambdas,
          test_acc_interp_proc,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, Procrustes interp.")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Accuracy")
  # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
  ax.set_title(f"Accuracy between the two models")
  ax.legend(loc="lower right", framealpha=0.5)
  fig.tight_layout()
  return fig
