import numpy as np
import scipy.stats as stats # used for inverse Gaussian
import scipy.spatial.distance as distance # distance used for kernel
from sklearn.gaussian_process.kernels import Matern as skMatern

#Function generating standard normals using the inverse of the univariate Gaussian CDF:
def normals_inv(u):
    # avoid origin
    u[u == 0] = np.nextafter(0, 1)
    # create standard normal samples
    z = stats.norm.ppf(u, loc=0, scale=1)
    return z


def boxmuller(unif1, unif2):
    u1 = np.sqrt(-2 * np.log(unif1)) * np.cos(2 * np.pi * unif2)
    u2 = np.sqrt(-2 * np.log(unif1)) * np.sin(2 * np.pi * unif2)
    return np.transpose(np.vstack([u1, u2]))


def normals(n, d, unif, sv=False):
    # avoid origin
    unif[unif == 0] = np.nextafter(0, 1)

    # if d is odd, add one dimension
    if d % 2 != 0:
        dim = d + 1
    else:
        dim = d

    # expand dimensions for SV model
    if sv == True:
        dim = 2 + 2 * d

    # create standard normal samples
    u = np.zeros((n, dim))
    for i in np.arange(0, dim, 2):
        u[:, i:(i + 2)] = boxmuller(unif[:, i], unif[:, (i + 1)])

    # if d is odd, drop one dimension
    if d % 2 != 0 or sv == True:
        u = np.delete(u, -1, 1)

    return u


# generator for g-and-k distribution
def sample_gandk(n, theta, method="MC"):
    a = theta[0]
    b = theta[1]
    g = theta[2]
    k = np.exp(theta[3])

    if method == "MC":
        z_unif = np.random.rand(n, 1)
    elif method == "QMC":
        sampler = stats.qmc.Sobol(d=1, scramble=True)
        z_unif = sampler.random(n)

    z = normals_inv(z_unif)
    x = np.zeros(shape=(n, 1))

    for i in range(n):
        x[i] = a + b * (1 + 0.8 * ((1 - np.exp(-g * z[i])) / (1 + np.exp(-g * z[i])))) * ((1 + z[i] ** 2) ** (k)) * z[i]
    return z_unif, z, x

# Function estimate the MMD-squared using the V-statistic
def MMD_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    z = np.concatenate((x, y), axis=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * np.sum(kxx) - (2 / (m * n)) * np.sum(kxy) + (1 / n ** 2) * np.sum(kyy)

# Function to set the lengthscale of the kernel using median heuristic
def median_heuristic(y):
    a = distance.cdist(y, y, 'sqeuclidean')
    return np.sqrt(np.median(a / 2))


# Function to compute the kernel Gram matrix
def kernel_matrix(x, y, l):
    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    return np.exp(-(1 / (2 * l ** 2)) * distance.cdist(x, y, 'sqeuclidean'))

# Function to compute the kernel mean embedding wrt Lebesgue measure (U is uniformly distributed) with c as the SE kernel
def embedding_unif(u):

    # Compute lengthscale for kernel c
    l = median_heuristic(u)

    dim = u.shape[1]
    z = np.zeros(shape=u.shape)

    for i in range(dim):
        z[:, i] = np.sqrt(2 * np.pi) * l * (stats.norm.cdf(1, loc=u[:, i], scale=l) -
                                            stats.norm.cdf(0, loc=u[:, i], scale=l))
    if dim == 1:
        return z
    else:
        return np.prod(z, axis=1)

# Function to compute the kernel mean embedding wrt Gaussian measure with c as the SE kernel
def embedding_Gaussian(u):
    # Function to compute the embedding when U is Gaussian distributed

    # Compute lengthscale for kernel c
    l = median_heuristic(u)

    dim = u.shape[1]
    m = u.shape[0]
    z = np.zeros(shape=(m, 1))

    mu = 0
    sigma = 1

    for i in range(m):
        z[i] = (l ** 2 / (l ** 2 + sigma ** 2)) ** (dim / 2) * np.exp(
            - np.linalg.norm(u[i, :]) ** 2 / (2 * (l ** 2 + sigma ** 2)))

    return z

# Function to compute the kernel mean embedding wrt Lebesgue measure (U is uniformly distributed) with c as the Matern kernel
def embedding_Matern(x, nu = 1.5):
    l = median_heuristic(x)
    r = 1 # For domain of Lebesgue measure as (0,1)
    if nu==1.5:
        term1 = 4 * l / np.sqrt(3) - 1 / 3 * np.exp(np.sqrt(3) * (x - 1) / l) * (3*1 + 2 * np.sqrt(3) * l - 3*x)
        term2 = 1 / 3 * np.exp(np.sqrt(3) * (0-x) / l) * ( 3 * x + 2 * np.sqrt(3) * l)
        K = 1 / r * (term1 - term2)
    elif nu == 2.5:
        term1 = 16 * l / (3 * np.sqrt(5))
        term2 = 1 / (15*l) * np.exp(np.sqrt(5) * (x-1) / l) * (8 * np.sqrt(5) * l**2 + 25*l*(1-x) + 5*np.sqrt(5)*(1-x)**2)
        term3 = 1 / (15*l) * np.exp(np.sqrt(5) * (0-x) / l) * (8 * np.sqrt(5) * l**2 + 25*l*(x) + 5*np.sqrt(5)*(0-x)**2)
        K = 1 / r * (term1 - term2 - term3)
    return K

# Function estimate the MMD-squared using our optimally-weighted (OW) estimator
def MMD_weighted(x, y, w, lengthscale, kernel="Gaussian", nu=1.5):
    #     """ Optimally weighted squared MMD estimate between samples x_i ~ P and y_i ~ Q
    #     """

    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()
        w = np.array(w, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    xy = np.concatenate((x, y), axis=0)

    if kernel == "Gaussian":
        K = kernel_matrix(xy, xy, lengthscale)
    elif kernel == "Matern":
        km = skMatern(length_scale=lengthscale, nu=nu)
        K = km(xy, xy)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    # first sum
    sum1 = np.matmul(np.matmul(w.transpose(), kxx), w)

    # second sum
    sum2 = np.sum(np.matmul(w.transpose(), kxy))

    # third sum
    sum3 = (1 / n ** 2) * np.sum(kyy)

    return sum1 - (2 / (n)) * sum2 + sum3

# Function to compute the optimal weights given random variables u_1, ..., u_m ~ U
def computeWeights(u, z, kernel="Gaussian", nu=2.5):
    m = u.shape[0]

    # Compute lengthscale for kernel c
    l = median_heuristic(u)

    # Compute Gram-matrix C
    delta = 1e-8
    if kernel == "Gaussian":
        C = kernel_matrix(u, u, l) + delta * np.identity(m)
    elif kernel == "Matern":
        km = skMatern(length_scale=l, nu=nu)
        C = km(u, u) + delta * np.identity(m)

    C_inv = np.linalg.inv(C)

    return np.matmul(C_inv, z)

# generator for multivariate g-and-k distribution. Outputs uniform rvs u, Gaussian rvs z_standard and data x
def sample_mvgandk(input_values, n, dim):
    # Parameters
    A = input_values[0]
    B = input_values[1]
    g = input_values[2]
    k = np.exp(input_values[3])
    rho = input_values[4]
    c = 0.8

    cov = np.eye(dim) + rho * np.eye(dim, k=1) + rho * np.eye(dim, k=-1)
    L = np.linalg.cholesky(cov)

    u = np.random.rand(n, dim)
    z_standard = np.zeros(shape=u.shape)

    for i in range(dim):
        z_standard[:, i] = normals_inv(u[:, i])

    z = np.matmul(L, z_standard.transpose())
    z = z.transpose()

    z = z_standard

    x = np.zeros(shape=(n, dim))

    for i in range(n):
        x[i, :] = A + B * (1 + c * ((1 - np.exp(-g * z[i, :])) / (1 + np.exp(-g * z[i, :])))) * ((1 + z[i, :] ** 2) **k) * z[i, :]

    return u, z_standard, x