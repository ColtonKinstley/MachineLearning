import datetime

import matplotlib as mpl
from cartopy import crs as ccrs
from matplotlib import pyplot as plt


def plot(close=True, filename=False, save=True):
    if filename and save:
        plt.savefig(filename)
    elif save:
        title = plt.axes().get_title()
        plt.savefig("./graphs/" +
                    datetime.datetime.utcnow().strftime("%Y-%m-%d+%H:%M:%S") +
                    "_" + title)

    with plt.style.context(("dark_background")):
        if close:
            plt.show()
            plt.close()
        else:
            plt.show()


def uk_axes():
    ax = plt.axes(projection=ccrs.OSGB())
    ax.coastlines(resolution="10m")
    return ax
