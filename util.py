import numpy as np
import matplotlib.pyplot as plt


def demo_results(nodes_coord, nodal_value, opt='euler', save='results'):
    pos_x = nodes_coord[:, 0]
    pos_y = nodes_coord[:, 1]
    if opt == 'euler':
        pos_x += nodal_value[:, 0]
        pos_y += nodal_value[:, 1]
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.scatter(pos_x, pos_y, c=nodal_value[:, 0], cmap='jet')  # 你可以指定一个颜色映射
    plt.colorbar()
    plt.subplot(122)
    plt.scatter(pos_x, pos_y, c=nodal_value[:, 1], cmap='jet')  # 你可以指定一个颜色映射
    plt.colorbar()
    if save:
        plt.savefig(save)
    plt.show()

