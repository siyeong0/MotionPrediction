import os
import glob
from threading import Thread

from cvt import *

if __name__ == "__main__":
    bvh_files = glob.glob("data/bvh/*.bvh")
    names = [os.path.split(os.path.splitext(bvh_file)[0])[1] for bvh_file in bvh_files]
    
    def work(names, start, end):
        for i in range(start, end):
            name = names[i]
            bvh_to_fbx(f"data/bvh/{name}.bvh", "data/fbx")
            import_fbx(f"data/fbx/{name}.fbx", "data/npy")
            retarget(f"data/npy/{name}.npy", "data/ret")
        
    num_works = len(names)
    num_threads = 8
    n_works = int(num_works / num_threads)
    r_works = num_works % num_threads
    
    workers = [Thread(target=work, args=(names, i * n_works, (i+1) * n_works)) for i in range(num_threads)]
    if r_works > 0: workers.append(Thread(target=work, args=(names, num_threads * n_works, num_works)))
    for i in range(len(workers)):
        workers[i].start()
    for i in range(len(workers)):
        workers[i].join()
        