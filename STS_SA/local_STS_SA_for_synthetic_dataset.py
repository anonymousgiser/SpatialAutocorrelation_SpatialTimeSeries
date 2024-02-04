import ERP
import numpy as np
import scipy.stats as sta

# Generate spatial weighted matrix
weight = np.zeros([21 * 28, 21 * 28], dtype=int)
for i in range(21):
    for j in range(28):
        if 0 <= i - 1 < 21 and 0 <= j < 28:
            weight[28 * i + j, 28 * (i - 1) + j] = 1
        if 0 <= i + 1 < 21 and 0 <= j < 28:
            weight[28 * i + j, 28 * (i + 1) + j] = 1
        if 0 <= i < 21 and 0 <= j - 1 < 28:
            weight[28 * i + j, 28 * i + j - 1] = 1
        if 0 <= i < 21 and 0 <= j + 1 < 28:
            weight[28 * i + j, 28 * i + j + 1] = 1

# Calculate ERP between STSs
file_path = "datasets\synthetic_STS_Data.csv"
sts = np.loadtxt(file_path, delimiter=",")
g = np.array(0, dtype=np.float64)
dist = np.zeros([28 * 21, 28 * 21])
for i, data_i in enumerate(sts):
    tsi = np.array(data_i, dtype=np.float64)
    for j, data_j in enumerate(sts):
        tsj = np.array(data_j, dtype=np.float64)
        if i < j:
            dist[i, j] = ERP.erp_similarity(tsi, tsj, g)
            dist[j, i] = dist[i, j]

# Measure spatial autocorrelation of STS
si = np.zeros([21, 28])
for i in range(21):
    for j in range(28):
        index = 28 * i + j
        dist_row = dist[index, :]
        msk = weight[index, :]
        sum_all = np.sum(dist_row)
        sum_neighbor = np.sum(dist_row[np.where(msk)])
        si[i, j] = sum_neighbor / sum_all

# Calculate Z-scores and P-values of results
z = np.zeros([21, 28])
p = np.zeros([21, 28])
n = 21*28
for i in range(21):
    for j in range(28):
        index = 28 * i + j
        wi = np.sum(weight[index, :])
        yi1 = np.sum(dist[index, :]) / (n - 1)
        yi2 = np.dot(dist[index, :], np.transpose(dist[index, :])) / (n - 1) - yi1**2
        E = wi / (n - 1)
        D = wi * (n - 1 - wi) * yi2 / ((n - 1)**2 * (n - 2) * yi1**2)
        z[i, j] = (si[i, j] - E) / np.sqrt(D)
        p[i, j] = sta.norm.sf(np.abs(z[i, j]))
        
# Save results into files
path1 = "results\local_SA_for_synthetic_STS.csv"
np.savetxt(path1, si, delimiter=",", fmt='%f')
path2 = "results\local_SA_for_synthetic_STS_z_scores.csv"
np.savetxt(path2, z, delimiter=",", fmt='%f')
path3 = "results\local_SA_for_synthetic_STS_p_values.csv"
np.savetxt(path3, p, delimiter=",", fmt='%f')