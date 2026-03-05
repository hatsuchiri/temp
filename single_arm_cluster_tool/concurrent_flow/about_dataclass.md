# about class Recipe
recipe_id: 0 or 1
group: 0 or 1
flow: np.array, e.g. [1,1,1,0,0,0,1] # include inloadlock and outloadlock
time: np.array, e.g. [0,10,20,0,0,0,0]
self.recipes consist of Recipe instances

# 最初的embedding在embedding.py里，LotStageEmbedding的forward函数里直接用row_feat = torch.zeros(size=(batch_size, self.row_cnt, 1), device=self.device)起手

# loc_id包不包括inloadlock和outloadlock
loc_id includes loadlock，and start from 0
pms doesn't include loadlock, and start from 0
pms[loc_id-1] refers to the PM that loc_id belongs to, idx of pms should equal loc_id-1
eg. pms[0] refers to PM1, loc_id=1 refers to PM1
pm.id start from 1，可以关注sdcfEnv.py的 _init_pms, step

# self.queue 和 self.loadlock_1

