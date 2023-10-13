import subprocess

def bvh_to_fbx(src_path, dst_dir):
    cmd_args = ['python']
    cmd_args.append('cvt/bvh_to_fbx.py') # program name
    cmd_args.append(src_path)
    cmd_args.append(dst_dir)
    
    ret = subprocess.call(cmd_args)
        
def import_fbx(src_path, dst_dir):
    cmd_args = ['python']
    cmd_args.append('cvt/import_fbx.py') # program name
    cmd_args.append(src_path)
    cmd_args.append(dst_dir)
    
    subprocess.call(cmd_args)
        
def retarget(src_path, dst_dir, cfg=None):
    cmd_args = ['python']
    cmd_args.append('cvt/retarget.py') # program name
    cmd_args.append(src_path)
    cmd_args.append(dst_dir)
    if cfg != None: cmd_args.append(cfg)    # default cmu
    
    subprocess.call(cmd_args)