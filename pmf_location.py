from __future__ import print_function
from evaluations import *
from pmf_model import *
import copy
import csv

print('PMF Recommendation Model Example')

city_name = 'NYC'  # city_name取值为NYC或LA
PROJECT_ROOT = 'C:/workspace/PMF/data/'

# 台式机上的文件路径
# path = os.path.join(
#     '/home/lxq/FourSquare Dataset/Tips/', city_name, city_name + '-tips-records.txt')
# data = np.loadtxt(path, dtype=float)
# # 将用户id与index映射写入文件
# user_id_index = pickle.load(open(os.path.join(
#     '/home/lxq/FourSquare Dataset/Tips/', city_name, city_name + '-userid-index.pkl'), 'rb'))
# # 将位置id与index映射写入文件
# venue_id_index = pickle.load(open(os.path.join(
#     '/home/lxq/FourSquare Dataset/Tips/', city_name, city_name + '-venueid-index.pkl'), 'rb'))

# 笔记本上的文件路径
path = os.path.join(
    'C:/workspace/PMF/data/', city_name, city_name + '-tips-records.txt')
data = np.loadtxt(path, dtype=float)
# 将用户id与index映射写入文件
user_id_index = pickle.load(open(os.path.join(
    'C:/workspace/PMF/data/', city_name, city_name + '-userid-index.pkl'), 'rb'))
# 将位置id与index映射写入文件
venue_id_index = pickle.load(open(os.path.join(
    'C:/workspace/PMF/data/', city_name, city_name + '-venueid-index.pkl'), 'rb'))

# 划分数据集
ratio = 0.8
train_data = data[:int(ratio * data.shape[0])]
vali_data = data[int(ratio * data.shape[0]):int((ratio + (1 - ratio) / 2) * data.shape[0])]
test_data = data[int((ratio + (1 - ratio) / 2) * data.shape[0]):]

NUM_USERS = max(user_id_index.values()) + 1
NUM_ITEMS = max(venue_id_index.values()) + 1
print('dataset density:{:f}'.format(len(data) * 1.0 / (NUM_USERS * NUM_ITEMS)))
print(NUM_USERS, NUM_ITEMS)

R = np.zeros([NUM_USERS, NUM_ITEMS])
for ele in train_data:
    R[int(ele[0]), int(ele[1])] = float(ele[2])

# construct model
print('training model.......')
lambda_alpha = 0.01
lambda_beta = 0.01
latent_size = 200 #TODO 修改潜在特征矩阵的size
lr = 3e-5
iters = 30
model = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta,
            latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)
print('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d}, lr={:f}, iters={:d}'.format(
    ratio, lambda_alpha, lambda_beta, latent_size, lr, iters))
U, V, train_loss_list, vali_rmse_list = model.train(
    train_data=train_data, vali_data=vali_data)

print('testing model.......')
preds = model.predict(data=test_data)
test_rmse = RMSE(preds, test_data[:, 2])

print('test rmse:{:f}'.format(test_rmse))

# 构造测试数据
test_result = {}
input_data =  open(os.path.join(PROJECT_ROOT, city_name, city_name+'_test_data.csv'),'r',encoding='ISO-8859-1')
input_reader = csv.reader(input_data)

for item in input_reader:
    if input_reader.line_num == 1:
        continue
    user_index = user_id_index[int(item[0])]
    venue_index = venue_id_index[item[1]]
    if user_index not in test_result:
        test_result[user_index] = [venue_index]
    else:
        test_result[user_index].append(venue_index)

# print(test_result)

I = copy.deepcopy(R)
I[I!=0] = 1
R_re = np.dot(U, V.T)
total_pre = 0
s = 0
for row in range(len(R_re)):
    if row in test_result:

        # top1
        # temp_top = [-1]
        # temp_res = [-1]

        # top5
        # temp_top = [-1,-1,-1,-1,-1]
        # temp_res = [-1,-1,-1,-1,-1]

        # top10
        # temp_top = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        # temp_res = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

        # top15
        # temp_top = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        # temp_res = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

        # top20
        temp_top = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        temp_res = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        for col in range(len(R_re[0])):
            if I[row][col] == 0:
                if R_re[row][col] > min(temp_top):
                    id = temp_top.index(min(temp_top)) 
                    temp_top[id] = R_re[row][col]
                    temp_res[id] = col
        # temp_res
        # print(temp_res)
        count = 0
        for loc in temp_res:
            if loc in test_result[row]:
                count += 1
        # total_pre += count / min(len(test_result[row]), len(temp_res))
        total_pre += count / len(temp_res)
        if count / len(temp_res) >= (1/len(temp_res)):
            s += 1
        # total_pre = max(count / len(temp_res), total_pre)

# total_pre = total_pre / len(test_result)
print('s = ' + str(s))
print('total precision = ' + str(total_pre/(s + 1)))
