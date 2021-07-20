import os
import getpass

project_name = "summer_project_hydra"
home_dir= []
if getpass.getuser() == "kadu":
    home_dir = f"/home/kadu/Dropbox/{project_name}"
elif getpass.getuser() == "natalie":
    home_dir = f"/home/natalie/Desktop/files/uni/ESO/{project_name}"
else:
    raise ValueError("Computer not set up!")
