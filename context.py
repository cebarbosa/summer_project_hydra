import getpass

import matplotlib.pyplot as plt
import seaborn as sns


project_name = "summer_project_hydra"
home_dir= []
if getpass.getuser() == "kadu":
    home_dir = f"/home/kadu/Dropbox/{project_name}"
elif getpass.getuser() == "natalie":
    home_dir = f"/home/natalie/Desktop/files/uni/ESO/{project_name}"
else:
    raise ValueError("Computer not set up!")

# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
sns.set()
sns.set_style(rc={"font.family": "serif"})
