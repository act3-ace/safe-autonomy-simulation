import os
import tqdm
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.inspection as inspection


rng = np.random.default_rng(0)

inspector = inspection.Inspector("inspector")
target = inspection.Target("target", radius=1, num_points=100)
sun = inspection.Sun()
sim = safe_autonomy_simulation.Simulator(
    frame_rate=1, entities=[inspector, target, sun]
)

for _ in tqdm.tqdm(range(1000)):
    inspector.add_control(rng.uniform(-1, 1, size=3))
    target.add_control(rng.uniform(-1, 1, size=3))
    sim.step()
