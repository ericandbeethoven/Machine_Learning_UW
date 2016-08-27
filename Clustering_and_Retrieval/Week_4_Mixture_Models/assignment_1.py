import colorsys
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

PLOT_FLAG = False


def generate_MoG_data(num_data, means, covariances, weights):
    """
    Generate a list of data points from given mixture of Gaussians model.
    :param num_data: number of data points
    :param means: means of Gaussians
    :param covariances: covariance of Gaussians
    :param weights: weights of each model
    :return: a list of data points
    """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data


def log_sum_exp(Z):
    """
    Compute log(\sum_i exp(Z_i)) for some list Z
    :param Z: list
    :return: log(\sum_i exp(Z_i))
    """
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def loglikelihood(data, weights, means, covs):
    """
    Compute the log-likelihood of the data for a Gaussian mixture model with the given parameters.
    :param data: dataset
    :param weights: list of weights of models
    :param means: list of means of models
    :param covs: list of covariance matrix of models
    :return: log-likelihood of the data for a mixture model
    """
    num_clusters = len(means)
    num_dim = len(data[0])

    ll = 0
    for d in data:
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

            # Compute log-likelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1 / 2. * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)

        # Increment log-likelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)

    return ll


def run_em(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4, report=True):
    """
    Run EM algorithm on data to assign models.
    :param data: whole dataset
    :param init_means: initial list of means
    :param init_covariances: initial list of covariance matrices
    :param init_weights: initial list of weights of models
    :param maxiter: maximum number of iterations
    :param thresh: threshold on log-likelihood to stop algorithm
    :param report: print progress by every 5 iterations if True
    :return: {'weights': weights, 'means': means, 'covs': covariances,
              'loglik': log-likelihood trace, 'resp': responsibilities matrix}
    """
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]

    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)

    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]

    if report:
        print("Start running EM...")
    for i in range(maxiter):
        if i % 5 == 0 and report:
            print("Iteration %s" % i)

        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        # Hint: To compute likelihood of seeing data point j given cluster k, use multivariate_normal.pdf.
        for j in range(num_data):
            for k in range(num_clusters):
                resp[j, k] = weights[k] * multivariate_normal.pdf(data[j], mean=means[k], cov=covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums  # normalize over all possible cluster assignments

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = np.sum(resp, axis=0)
        for k in range(num_clusters):

            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            weights[k] = counts[k] / num_data

            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            weighted_sum = 0
            for j in range(num_data):
                weighted_sum += data[j] * resp[j, k]
            means[k] = weighted_sum / counts[k]

            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                # Hint: Use np.outer on the data[j] and this cluster's mean
                weighted_sum += resp[j, k] * np.outer(data[j] - means[k], data[j] - means[k])
            covariances[k] = weighted_sum / counts[k]

        # Compute the log-likelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)

        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest

    if report:
        if i % 5 != 0:
            print("Iteration %s" % i)
        print()

    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}
    return out


def plot_contours(data, means, covs, title):
    """Make contour plots of given data points and given Gaussian components"""
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors=col[i])
        plt.title(title)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


def plot_responsibilities_in_RB(img, resp, title):
    """Plot soft assignment responsibilities only on R and B dimensions."""
    N, K = resp.shape

    HSV_tuples = [(x * 1.0 / K, 0.5, 0.9) for x in range(K)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    R = img['red']
    B = img['blue']
    resp_by_img_int = [[resp[n][k] for k in range(K)] for n in range(N)]
    cols = [tuple(np.dot(resp_by_img_int[n], np.array(RGB_tuples))) for n in range(N)]

    plt.figure()
    for n in range(len(R)):
        plt.plot(R[n], B[n], 'o', c=cols[n], markeredgecolor='black', markeredgewidth=0.5)
    plt.title(title)
    plt.xlabel('R value')
    plt.ylabel('B value')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


def get_top_images(assignments, cluster, k=5):
    """
    Get top k images' paths that assigned to given cluster index.
    :param assignments: dataset contains cluster assignments and paths
    :param cluster: index of cluster
    :param k: number of images
    :return: list of image paths
    """
    images_in_cluster = assignments[assignments['assignments'] == cluster]
    top_images = images_in_cluster.sort_values('probs', ascending=False).head(k)
    return top_images['image_path'].values


def display_images(image_paths, title=''):
    """Display all images in image paths list"""
    for i in range(len(image_paths)):
        plt.figure()
        plt.axis("off")
        plt.title("{:s} Image {:d}".format(title, i))
        plt.imshow(mpimg.imread(image_paths[i]))


# A simple mixture of models:
print("Generate data points from a simple mixture of models:")
# Model parameters
init_means = [
    [5, 0],  # mean of cluster 1
    [1, 1],  # mean of cluster 2
    [0, 5]   # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]],         # covariance of cluster 1
    [[.92, .38], [.38, .91]],    # covariance of cluster 2
    [[.5, 0.], [0, .5]]          # covariance of cluster 3
]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

# Report
print("    Initial means: " + str(init_means))
print("    Initial covariances: " + str(init_covariances))
print("    Initial Weights: " + str(init_weights))
print()

# Plot
if PLOT_FLAG:
    plt.figure()
    d = np.vstack(data)
    plt.plot(d[:, 0], d[:, 1], 'ko')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Test EM implementation
print("Testing EM implementation. Should end at Iteration 22:")

# Initialization of parameters
np.random.seed(4)
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

# Run EM
results_1 = run_em(data, initial_means, initial_covs, initial_weights)

# Plot
# Parameters after initialization
plot_contours(data, initial_means, initial_covs, 'Initial clusters')

# Parameters after 12 iterations
results_plot_12 = run_em(data, init_means, initial_covs, init_weights, maxiter=12, report=False)
plot_contours(data, results_plot_12['means'], results_plot_12['covs'], 'Clusters after 12 iterations')

# Parameters after running EM to convergence
plot_contours(data, results_1['means'], results_1['covs'], 'Final clusters')

# Run EM Algorithm on images
print("Run EM algorithm on images:")

# Load data
images = pd.read_csv('../Data/images.csv')
images['rgb'] = list(images[['red', 'green', 'blue']].values)
images['path'] = images['path'].map(lambda s: "../Data/images/" + s[len("/data/coursera/images/"):])

# Initial 4 randomly chosen images as initial clusters.
# Initialize the covariance matrix of each cluster to be diagonal
# with each element equal to the sample variance from the full data.
np.random.seed(1)
init_means = [images['rgb'][x] for x in np.random.choice(len(images), 4, replace=False)]
cov = np.diag([images['red'].var(), images['green'].var(), images['blue'].var()])
init_covariances = [cov, cov, cov, cov]
init_weights = [1/4., 1/4., 1/4., 1/4.]

# Convert rgb data to numpy arrays
img_data = [np.array(i) for i in images['rgb']]

# Run our EM algorithm on the image data using the above initializations.
# This should converge in about 125 iterations
out = run_em(img_data, init_means, init_covariances, init_weights)

# Compute the likelihood (score) of the first image under each Gaussian component.
score = []
for k in range(4):
    score.append(multivariate_normal.pdf(images['rgb'][0], mean=out['means'][k], cov=out['covs'][k]))

# Plot for convergence
if PLOT_FLAG:
    # Log-likelihood at each iteration
    ll = out['loglik']
    plt.plot(range(len(ll)), ll, linewidth=4)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    # Log-likelihood at first 3 iterations
    plt.figure()
    plt.plot(range(3, len(ll)), ll[3:], linewidth=4)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Plot for cluster assignment
N, K = out['resp'].shape

# Random, after 1 iteration, after 20 iterations
random_resp = np.random.dirichlet(np.ones(K), N)
out_after_1 = run_em(img_data, init_means, init_covariances, init_weights, maxiter=1, report=False)
out_after_20 = run_em(img_data, init_means, init_covariances, init_weights, maxiter=20, report=False)

# Plot
if PLOT_FLAG:
    plot_responsibilities_in_RB(images, random_resp, 'Random responsibilities')
    plot_responsibilities_in_RB(images, out_after_1['resp'], 'After 1 iteration')
    plot_responsibilities_in_RB(images, out_after_20['resp'], 'After 20 iterations')
    plot_responsibilities_in_RB(images, out['resp'], 'Final result')

# Calculate cluster assignments for the entire image dataset using the result of running EM for 20 iterations above:
means_after_20 = out_after_20['means']
cov_after_20 = out_after_20['covs']
resp_after_20 = out_after_20['resp']
rgb = images['rgb']
N = len(images)
K = 4

assignments = [0] * N
probs = [0] * N

for i in range(N):
    # Compute the score of data point i under each Gaussian component:
    p = np.zeros(K)
    for k in range(K):
        # use multivariate_normal.pdf and rgb[i]
        p[k] = multivariate_normal.pdf(rgb[i], mean=means_after_20[k], cov=cov_after_20[k])

    # Compute assignments of each data point to a given cluster based on the above scores:
    assignments[i] = np.argmax(p)

    # For data point i, store the corresponding score under this cluster assignment:
    probs[i] = np.max(p)

assignments = pd.DataFrame({
    'assignments': pd.Series(assignments),
    'probs': pd.Series(probs),
    'image_path': images['path']
})

# Report
print("Cluster Assignments after 20 iterations (show top 20):")
print(assignments[['assignments', 'probs']].head(20).to_string(index=False))
print()
print("Top 5 images with highest likelihood in each component:")
for component_id in range(4):
    images = get_top_images(assignments, component_id)
    print("Component {:d}:".format(component_id))
    for image in images:
        print("\t{:s}".format(image[len("../Data/images/"):]))
    if PLOT_FLAG:
        display_images(images, title='Component {:d}:'.format(component_id))
print()

# QUIZ QUESTIONS:
print("Quiz Questions:")
# 1. What is the weight that EM assigns to the first component after running the above code block?
print("1. The weight that EM assigns to the first component after running the above code block is {:f}.\n"
      .format(results_1['weights'][0]))
# 2. Obtain the mean that EM assigns the second component. What is the mean in the first dimension?
print("2. The mean of the second component in the first dimension is {:f}.\n"
      .format(results_1['means'][1][0]))
# 3. Obtain the covariance that EM assigns the third component. What is the variance in the first dimension?
print("3. The variance of the third component in the first dimension is {:f}.\n"
      .format(results_1['covs'][2][0, 0]))
# 4. Calculate the likelihood (score) of the first image in our data set (images[0])
#    under each Gaussian component through a call to multivariate_normal.pdf.
#    Given these values, what cluster assignment should we make for this image?
print("4. The likelihood of the first image in our data set is: ")
print("   " + str(score))
print("   The first image should be assigned to Cluster {:d}.\n".format(np.argmax(score)))
# 5. Which of the following images are not in the list of top 5 images in the first cluster?
print("5. Image 1, 2, 6, 7 are not in the top 5 images in the first cluster. "
      "(See assignment description on course website for images.)")

if PLOT_FLAG:
    plt.show()
