import torch as th
import time

n = 10
dev = 'cuda'



for n in range(10):

    start = time.time()

    n += 10

    mtx = th.zeros(n, n, n).to(dev)

    for i in range(10):
        mtx += th.randint_like(mtx, low=-10, high=10)

    end = time.time()
    print(n, end - start)


