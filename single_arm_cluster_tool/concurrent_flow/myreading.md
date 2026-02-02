# stage
            if pm.group == 0 and pm.stage == len(self.group1_stage):
                return pm.hold_wafer is not None  # Last stage for type 1
            elif pm.group == 1 and pm.stage == len(self.stage):
                return pm.hold_wafer is not None  # Last stage for type 2

    stage = [....]
    stage[0:self.group1_stage] belongs to group 1
    stage[self.group1_stage:] belongs to group 2
    After last stage is LL_out, which is always available

# embedding.py 
    to see the default value of params in LotStageEmbedding class, check the def main() in sdcfEnv.py

state.loc_hold_wafer == []

# about class Recipe
recipe_id: 0 or 1
group: 0 or 1
flow: np.array, e.g. [1,1,1,0,0,0,1] # include inloadlock and outloadlock
time: np.array, e.g. [0,10,20,0,0,0,0]
self.recipes consist of Recipe instances
    


