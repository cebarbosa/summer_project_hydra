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

fields = ["fieldA", "fieldB", "fieldC", "fieldD"]
seeing = [0.92, 1.4, 1.3, 1.8]
PS = 0.262 # arcsec / pixel for MUSE
fig_width = 3.54 # inches - A&A template

# Matplotlib settings
sns.set_style("ticks")
# plt.style.context("seaborn-paper")
# plt.rcParams["text.usetex"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

