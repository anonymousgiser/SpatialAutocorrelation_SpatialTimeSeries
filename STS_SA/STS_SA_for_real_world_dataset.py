import ERP
import numpy as np
import scipy.stats as sta
from scipy.signal import savgol_filter

# The number of STS
num = 170
N = num * (num - 1)

weight = np.loadtxt('datasets\SWM_for_real_world_dataset.csv', delimiter=",")
sts_ori = np.loadtxt('datasets\\real_World_STS_Data.csv', delimiter=",")

# Smooth the STS data
sts = []
for ts in sts_ori:
    sts.append(savgol_filter(ts, window_length=15, polyorder=2, mode='nearest'))

# Calculate distance matrix with ERP algorithm
g = np.array(0, dtype=np.float64)
dist = np.zeros([num, num])
for i, data_i in enumerate(sts):
    tsi = np.array(data_i, dtype=np.float64)
    for j, data_j in enumerate(sts):
        tsj = np.array(data_j, dtype=np.float64)
        if i < j:
            dist[i, j] = ERP.erp_similarity(tsi, tsj, g)
            print('the ERP between No.%s STS and No.%s STS has been calculated successfully' % (i, j))
            dist[j, i] = dist[i, j]

# Measure local SA of STS and test the significance
def local_STS_SA(dist, weight):
    s = np.zeros([num, 1])
    for index in range(num):
        dist_row = dist[index, :]
        msk = weight[index, :]
        sum_all = np.sum(dist_row)
        sum_neighbor = np.sum(dist_row[np.where(msk)])
        s[index, 0] = sum_neighbor / sum_all
    return s

def local_z_and_p(dist, weight, s):
    z = np.zeros([num, 1])
    p = np.zeros([num, 1])
    for index in range(num):
        wi = np.sum(weight[index, :])
        yi1 = np.sum(dist[index, :]) / (num - 1)
        yi2 = np.dot(dist[index, :], np.transpose(dist[index, :])) / (num - 1) - yi1**2
        E = wi / (num - 1)
        D = wi * (num - 1 - wi) * yi2 / ((num - 1)**2 * (num - 2) * yi1**2)
        z[index, 0] = (s[index, 0] - E) / np.sqrt(D)
        p[index, 0] = sta.norm.sf(np.abs(z[index, 0]))
    return z, p


s = local_STS_SA(dist, weight)
z, p = local_z_and_p(dist, weight, s)
order = np.arange(1, num + 1).reshape(num, 1)
local_table = np.concatenate((order, s, z, p), axis=1)
np.savetxt('results\local_SA_for_real_world_STS.csv', local_table, delimiter=',', fmt='%f', 
           header='id, local_s, z_scores, p_values')


# Measure global SA of STS and test the significance
numerator = np.sum(np.multiply(weight, dist))
denominator = np.sum(dist)
global_s = numerator / denominator
w = np.sum(weight)
es = w/N
b = N*np.sum(np.multiply(dist, dist))/(np.sum(dist)**2)
e_s2 = w*(N*(w-1)+b*(N-w))/(N*N*(N-1))
e_2_s = es**2
var = e_s2 - e_2_s
global_z_score = (global_s - es) / np.sqrt(var)
global_p_value = sta.norm.sf(np.abs(global_z_score))
global_table = np.vstack((global_s, global_z_score, global_p_value))
np.savetxt('results\global_SA_for_real_world_STS.csv', 
           global_table, delimiter=',', fmt='%f')