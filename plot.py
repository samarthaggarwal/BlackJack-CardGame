import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator

def plot(val, name=None):
    nx, ny = 10, 31
    x = range(nx)
    for i in range(4):
        y = range(-30,ny)
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        plt.xticks(x, range(1,nx+1))
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        ha.set_zlabel('Value Function')
        # ha.yaxis.set_major_locator(ticker.MultipleLocator(5))
        X, Y = np.meshgrid(x, y) 
        # ha.plot_wireframe(X, Y, val[i,0:ny+10*i,:], color='black')
        # ha.plot_surface(X, Y, val[i,0:ny+10*i,:],cmap=cm.coolwarm)
        Z = val[:,i,:]
        ha.plot_wireframe(X, Y, Z, color='black')
        ha.plot_surface(X, Y, Z,cmap=cm.coolwarm)
        ha.set_title('Surface plot for Trump cards = ' + str(i));
        ha.view_init(elev=25, azim=-7)
        if name:
            hf.savefig(name+'-'+str(i)+'.png')
    if not name:
        plt.show()

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Make data.
# X = np.arange(-30, 31, 1)
# Y = np.arange(0, 10, 1)
# Y, X = np.meshgrid(Y, X)
# # R = np.sqrt(X**2 + Y**2)
# # Z = np.sin(R)
# Z = v[:,3,:]

# # Plot the surface.
# # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
# # ax.view_init(elev=25, azim=-7)

# # Customize the z axis.
# ax.set_zlim(-1, 1)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()