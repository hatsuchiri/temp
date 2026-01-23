# stage
            if pm.group == 0 and pm.stage == len(self.group1_stage):
                return pm.hold_wafer is not None  # Last stage for type 1
            elif pm.group == 1 and pm.stage == len(self.stage):
                return pm.hold_wafer is not None  # Last stage for type 2

    stage = [....]
    stage[0:self.group1_stage] belongs to group 1
    stage[self.group1_stage:] belongs to group 2