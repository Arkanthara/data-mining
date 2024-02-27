from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def get_density(x, y):
    """
    Fits a Kernel Density Estimation model to the provided x and y data.

    Parameters:
    x (np.array): A 1D NumPy array containing the x-coordinates of the data points.
    y (np.array): A 1D NumPy array containing the y-coordinates of the data points.

    Returns:
    KernelDensity: A KernelDensity object fitted to the combined (x, y) data.

    This function creates a 2D dataset from the input 1D arrays `x` and `y` by
    stacking them vertically and then transposing the result to get an array of [x, y] pairs.
    It then fits a Kernel Density Estimation (KDE) model to this dataset, which can be used
    to estimate the density of the data at any point in the space. The `bandwidth` parameter
    of the KDE controls the smoothness of the resulting density estimate.
    """
    # Combine x and y into a single 2D array of shape (n_samples, 2).
    xy_train = np.vstack([x, y]).T

    # Initialize the Kernel Density model with a specified bandwidth.
    density = KernelDensity(bandwidth=0.2)

    # Fit the model to the data.
    return density.fit(xy_train)


def get_grid(x_min, x_max, y_min, y_max, n_points):
    """
    Generates a grid of points within specified bounds on the x and y axes.

    The function creates a uniform grid that spans the specified ranges on the x and y axes,
    with an equal number of points along each axis. This grid is particularly useful for
    evaluating functions over a specified area or for visualizing data that requires a
    systematic arrangement of points, such as in contour plots or density maps.

    Parameters:
    x_min (float): The minimum value on the x-axis.
    x_max (float): The maximum value on the x-axis.
    y_min (float): The minimum value on the y-axis.
    y_max (float): The maximum value on the y-axis.
    n_points (int): The number of points along each axis, leading to a grid of
                    n_points by n_points.

    Returns:
    tuple: A tuple containing two 2D numpy arrays. The first array corresponds to the
           x-coordinates of the grid points, and the second to the y-coordinates.

    Example:
    To generate a 10x10 grid for the area defined by x in [0, 5] and y in [0, 5]:
        x_grid, y_grid = get_grid(0, 5, 0, 5, 10)

    This will produce two arrays where `x_grid` contains the x-coordinates of each
    point in the grid and `y_grid` contains the corresponding y-coordinates.
    """
    # Generate linearly spaced values between the min and max values for each axis.
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)

    # Create a meshgrid, which generates two 2D arrays from the 1D arrays.
    # `x` becomes an array where each row is a copy of x values, and
    # `y` is an array where each column is a copy of y values.
    x, y = np.meshgrid(x, y)

    # Return the generated grid.
    return x, y


def density_eval2d(density, x_values, y_values):
    """
    Computes the density estimates for a 2D grid using a fitted density model.

    Parameters:
    density: A fitted density model object that supports the `score_samples` method.
             This could be, for example, an instance of sklearn's KernelDensity.
    x_values (np.array): A 2D NumPy array containing the x-coordinates of the grid points.
    y_values (np.array): A 2D NumPy array containing the y-coordinates of the grid points.

    Returns:
    np.array: A 2D NumPy array of the same shape as `x_values` and `y_values` containing
              the density estimates at each grid point.

    The function takes a density estimator (like a Kernel Density Estimator from sklearn)
    and evaluates the density at each point in a 2D grid defined by `x_values` and `y_values`.
    """
    # Reshape the grid points into a two-column array where each row is a point (x, y).
    grid = np.vstack([x_values.ravel(), y_values.ravel()]).T

    # Compute the log density model's score at each point and exponentiate to get the density.
    # The score_samples function returns the log of the density, so we use np.exp to obtain the actual density.
    return np.exp(density.score_samples(grid)).reshape(x_values.shape)

def plot_density(x, y, z):
    """
    Displays a color density map from the input NumPy arrays.

    Parameters:
    x (np.array): A 1D or 2D NumPy array containing the x-coordinates of the grid points.
    y (np.array): A 1D or 2D NumPy array containing the y-coordinates of the grid points.
    z (np.array): A 2D NumPy array containing the density or intensity values to be displayed.
                  The dimensions of z must match the grid defined by x and y.

    This function uses `plt.pcolormesh` to create a grid visualization where each cell
    is colored according to the corresponding density value in `z`. This approach is particularly
    useful for visualizing scientific or technical data such as density fields, heat maps, etc.

    Note:
    - `x` and `y` can either be grids (2D arrays) corresponding to each point in `z`,
      or vectors (1D arrays) specifying the coordinates of the grid cells' edges.
    - The function `plt.show()` is called at the end to display the generated plot.
    """
    # Create and display the density map using pcolormesh.
    # Colors represent the density/intensity values in 'z'.
    plt.pcolormesh(x, y, z)

    # Display the generated plot.
    # Useful in environments like Jupyter notebooks or Python scripts.
    plt.show()

def plot_density1d(x, y, color_map='viridis', title=None, xlabel=None, ylabel=None):
    """
    Remplit l'espace sous la courbe y avec des couleurs qui changent continuellement
    en fonction de la valeur de y, en utilisant une colormap.

    :param x: Les valeurs de l'axe des abscisses.
    :param y: Les valeurs de l'axe des ordonnées.
    :param color_map: Le nom de la colormap à utiliser.
    """
    # Créer une figure et un axe
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculer le minimum et le maximum de y pour normaliser les valeurs de y à [0, 1]
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Créer une colormap
    cmap = plt.get_cmap(color_map)

    # Créer des segments et les colorer en fonction de la valeur de y
    segments = []
    colors = []
    for i in range(len(x) - 1):
        seg = [(x[i], 0), (x[i], y[i]), (x[i+1], y[i+1]), (x[i+1], 0)]
        segments.append(seg)
        color = cmap(y_norm[i])
        colors.append(color)

    # Créer une collection de polygones
    poly = PolyCollection(segments, facecolors=colors, edgecolors='none')
    ax.add_collection(poly)

    # Dessiner la ligne de la courbe
    ax.plot(x, y, color='black')

    # Ajouter une barre de couleur pour montrer l'échelle
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(y), vmax=np.max(y)))
    sm.set_array([])  # Vous devez définir l'array même si vous ne l'utilisez pas
    plt.colorbar(sm, ax=ax, label='Density')

    # Ajouter des détails au graphique
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

def get_delta(min_, max_, n_points):
    corr = (max_- min_)/n_points
    delta = (max_- min_ + corr) / n_points
    return delta
 
def density_eval1d(density, points):
    log_prob = density.score_samples(points.T)   
    return np.exp(log_prob).reshape(points.shape[1])
