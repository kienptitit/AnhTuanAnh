import numpy as np
import re

with open("line_2021-07-01_19_09_29 (2).dat", 'r') as f:
    d = dict()
    ar = []
    cnt = 1
    d_item = dict()
    d_user = dict()
    cnt_user = 1
    for line in f:
        l = [int(i) for i in re.split(r"\s+", line.strip())]
        if l[1] not in d_item:
            d_item[l[1]] = cnt
            cnt += 1
        if l[0] not in d_user:
            d_user[l[0]] = cnt_user
            cnt_user += 1
        l[1] = d_item[l[1]]
        l[0] = d_user[l[0]]
        ar.append(l[0])
        if l[0] not in d:
            d[l[0]] = [l]
        else:
            d[l[0]].append(l)
train = []
test = []

for key, item in d.items():
    idx = int(len(item) * 0.8)
    for i in item[:idx]:
        train.append(i)
    for i in item[idx:]:
        test.append(i)

# np.save('train.npy', train)
# np.save('test.npy', test)
# print(np.max(np.load('train.npy')[:, 0]))
# print(np.max(np.l))

