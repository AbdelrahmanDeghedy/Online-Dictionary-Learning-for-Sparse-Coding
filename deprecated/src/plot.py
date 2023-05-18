import matplotlib.pyplot as plt


def plot_many(x, y_list, x_label, y_label, title=None, filename=None, to_save=False):
    fig = plt.figure(figsize=(10, 5))
    plt_axes = plt.axes()

    if title is not None:
        plt_axes.set_title(title)

    for i in range(len(y_list)) :
        plt.plot(x, y_list[i])

    plt_axes.set_xlabel(x_label)
    plt_axes.set_ylabel(y_label)

    if to_save:
        fig.savefig("../logs/" + filename + ".png")
    else:
        plt.show()