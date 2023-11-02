import bpy
import sys
import os

if __name__ == "__main__":
    assert(len(sys.argv)==3)
    src_path = sys.argv[1]
    dst_dir = sys.argv[2]
    
    if not os.path.isfile(src_path):
        print(f"{src_path} not exist.")
        exit(1)
    if not os.path.isdir(dst_dir):
        print(f"{dst_dir} not exist.")
        exit(1)

    src_dir, src_name = os.path.split(src_path)
    name, ext = os.path.splitext(src_name)
    if not ext == ".bvh":
        print(f"{ext} not supported.")
        exit(1)
    
    bpy.context.scene.render.fps = 120
    # See http://www.blender.org/documentation/blender_python_api_2_60_0/bpy.ops.import_anim.html
    bpy.ops.import_anim.bvh(filepath=src_dir+'/'+name+".bvh", 
                            global_scale=1, 
                            frame_start=1, 
                            use_fps_scale=False, 
                            use_cyclic=False, 
                            rotate_mode='NATIVE', 
                            axis_forward='-Z', axis_up='Y')
    bpy.context.scene.render.fps = 120
    # See http://www.blender.org/documentation/blender_python_api_2_62_1/bpy.ops.export_scene.html
    bpy.ops.export_scene.fbx(filepath=dst_dir+'/'+name+".fbx", 
                             axis_forward='-Z', axis_up='Y')