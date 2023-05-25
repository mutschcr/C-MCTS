from rocksample import Rocksample
import time
import numpy as np

MAX_STEPS = 100000

if __name__ == "__main__":
    env = Rocksample(7, 8, True)
    start = time.time()
    i = 0
    r = 0
    while i < MAX_STEPS:
        r = env.step(np.random.choice(env.get_legalactions()))
        s = env.get_state()
        done = s[-1]
        i += 1
        if done:
            break
    end = time.time()

    time_per_step = (end - start) / MAX_STEPS
    reward_per_step = r / MAX_STEPS
    print(f"total iterations: {i}")
    print(f"Time per step: {time_per_step :.2} s")
    print(f"Reward per step: {reward_per_step :.2}")
