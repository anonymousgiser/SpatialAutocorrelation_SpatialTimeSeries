import ERP
import numpy as np
import scipy.stats as sta

# generate spatial weighted matrix
def weight_matrix(row, col):
    weight = np.zeros([row*col, row*col], dtype=int)
    for i in range(row):
        for j in range(col):
            if 0 <= i - 1 < row and 0 <= j < col:
                weight[col * i + j, col * (i - 1) + j] = 1
            if 0 <= i + 1 < row and 0 <= j < col:
                weight[col * i + j, col * (i + 1) + j] = 1
            if 0 <= i < row and 0 <= j - 1 < col:
                weight[col * i + j, col * i + j - 1] = 1
            if 0 <= i < row and 0 <= j + 1 < col:
                weight[col * i + j, col * i + j + 1] = 1
    return weight
weight = weight_matrix(7, 7)
N = 49 * 48

# load STS data and divide the data into 12 blocks in 3 rows and 4 columns
file_path = "datasets\synthetic_STS_Data.csv"
data = np.loadtxt(file_path, delimiter=",")
sts_data = data.reshape(21, 28, 10)
blocks = []
for i in range(0, 21, 7):
    for j in range(0, 28, 7):
        blocks.append(sts_data[i:i+7, j:j+7])

# calculate the global S of STS_SA and test the significance of the results
global_s = []
z_score = []
p_value = []
for block in blocks:
    dist = np.zeros([49, 49])
    g = np.array(0, dtype=np.float64)
    for i, data_i in enumerate(block.reshape(49, 10)):
        tsi = np.array(data_i, dtype=np.float64)
        for j, data_j in enumerate(block.reshape(49, 10)):
            tsj = np.array(data_j, dtype=np.float64)
            if i < j:
                dist[i, j] = ERP.erp_similarity(tsi, tsj, g)
                dist[j, i] = dist[i, j]
    numerator = np.sum(np.multiply(weight, dist))
    denominator = np.sum(dist)
    s = numerator / denominator
    global_s.append(s)
    w = np.sum(weight)
    es = w/N
    b = N*np.sum(np.multiply(dist, dist))/(np.sum(dist)**2)
    e_s2 = w*(N*(w-1)+b*(N-w))/(N*N*(N-1))
    e_2_s = es**2
    var = e_s2 - e_2_s
    z = (s - es) / np.sqrt(var)
    z_score.append(z)
    p_value.append(sta.norm.sf(np.abs(z)))

global_s = np.array(global_s).reshape(12,1)
z_score = np.array(z_score).reshape(12,1)
p_value = np.array(p_value).reshape(12,1)
table = np.hstack((global_s, z_score, p_value))
path = "results\global_SA_for_synthetic_STS.csv"
np.savetxt(path, table, delimiter=",", fmt='%f', header='global_s, z_scores, p_values')

