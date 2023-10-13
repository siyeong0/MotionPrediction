import os
import glob

from cvt import *

if __name__ == "__main__":
    bvh_files = glob.glob("data/bvh/*.bvh")
    names = [os.path.split(os.path.splitext(bvh_file)[0])[1] for bvh_file in bvh_files]
    for i, name in enumerate(names):
        bvh_to_fbx(f"data/bvh/{name}.bvh", "data/fbx")
        import_fbx(f"data/fbx/{name}.fbx", "data/npy")
        retarget(f"data/npy/{name}.npy", "data/ret")
        
        print("===============================")
        print(f"progress : {i+1}/{len(names)}")
        print("===============================")
        