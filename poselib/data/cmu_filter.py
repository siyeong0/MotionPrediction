import shutil

src_path = "cmu"
dst_path = "bvh"

f = open("filtered-motion-index.txt", 'r')
while True:
    line = f.readline()
    if not line: break
    if line=="\n": continue
    print(line)

    name = line.split()[0]
    sub_dir = name.split("_")[0]
    
    shutil.copyfile(f"{src_path}/{sub_dir}/{name}.bvh", f"{dst_path}/{name}.bvh")
f.close()
