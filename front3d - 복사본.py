import blenderproc as bproc
import argparse
import os
import numpy as np
import csv
import bpy
from mathutils import Vector, Matrix
from blenderproc.python.modules.utility.ConfigParser import ConfigParser
from blenderproc.python.modules.writer.CameraStateWriter import CameraStateWriter
from blenderproc.python.utility.Utility import Utility
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=8888, stdoutToServer=True, stderrToServer=True)
pan_root='D:\\3D_Front\\front3d\\00ad8345-45e0-45b3-867d-4a3c88c2517a\\'
root_path='D:\\3D_Front\\'
config_path= 'C:\\Users\\esc15\\PycharmProjects\\pythonProject2\\BlenderProc\\config.yaml'
output_dir = 'C:\\Users\\esc15\\PycharmProjects\\pythonProject2\\BlenderProc\\mytest\\output\\'
parser = argparse.ArgumentParser()
parser.add_argument("--front",default='D:\\3D_Front\\3D-FRONT\\3D-FRONT\\00ad8345-45e0-45b3-867d-4a3c88c2517a.json', help="Path to the 3D front file")
parser.add_argument("--future_folder", default='D:\\3D_Front\\3D-FUTURE-model\\', help="Path to the 3D Future Model folder.")
parser.add_argument("--front_3D_texture_path", default='D:\\3D_Front\\3D-FRONT-texture\\3D-FRONT-texture\\', help="Path to the 3D FRONT texture folder.")
parser.add_argument("--output_dir",default=output_dir,help="Path to where the data should be saved")
args = parser.parse_args()

arg=[]
arg.append(args.front)
arg.append(args.future_folder)
arg.append(args.front_3D_texture_path)
arg.append(args.output_dir)

def find_near_obj(cam2world_matrix):
    cam2world_matrix = Matrix(cam2world_matrix)
    sqrt_number_of_rays = 10
    cam_ob = bpy.context.scene.camera
    cam = cam_ob.data
    cam2world_matrix = Matrix(cam2world_matrix)
    num_of_rays = sqrt_number_of_rays * sqrt_number_of_rays
    frame = cam.view_frame(scene=bpy.context.scene)
    frame = [cam2world_matrix @ v for v in frame]
    vec_x = frame[1] - frame[0]
    vec_y = frame[3] - frame[0]
    position = cam2world_matrix.to_translation()
    arr = set()

    for x in range(0, sqrt_number_of_rays):
        for y in range(0, sqrt_number_of_rays):
            # Compute current point on plane
            end = frame[0] + vec_x * x / float(sqrt_number_of_rays - 1) + vec_y * y / float(sqrt_number_of_rays - 1)
            # Send ray from the camera position through the current point on the plane
            hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(), position,
                                                                     end - position)
            if hit_object["nyu_category_id"] == 1 or hit_object["nyu_category_id"] == 2:
                continue
            else:
                arr.add(hit_object.name)
    return list(arr)
def read_csv_mapping(path):
    """ Loads an idset mapping from a csv file, assuming the rows are sorted by their ids.

    :param path: Path to csv file
    """

    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        new_id_label_map = []
        new_label_id_map = {}

        for row in reader:
            new_id_label_map.append(row["name"])
            new_label_id_map[row["name"]] = int(row["id"])

        return new_id_label_map, new_label_id_map
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
bproc.camera.set_resolution(320, 240)
bproc.camera.set_intrinsics_from_blender_params(lens=1.0471975511, lens_unit="FOV")


height = 0.75
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
nyu_mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_nyu_mapping.csv"))

mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
_,nyu_mapping =read_csv_mapping(nyu_mapping_file)
# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping,
    nyu_label_mapping = nyu_mapping
)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0


def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False


# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]
room_map=dict()
#for obj in loaded_objects:
    #room_id=obj.get_cp("room_id")
    #print(obj.get_name())
    #obj_name =obj.get_name()
    #room_map[obj_name]=room_id
print(room_map)
# print(obj.get_cp("category_id"))
proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}
hide_obj_num=3
cam_p=[]
while tries < 10000 and poses < 1:
#while tries < 3 and poses < 3:
    # Sample point inside house

    location = point_sampler.sample(height)
    # Sample rotation (fix around X and Y axis)
    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
    # meters and make sure that no background is visible, finally make sure the view is interesting enough
    if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects, special_objects_weight=10.0) > 0.8 \
            and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
        print(location)
        print(rotation)
        loc= np.array(cam2world_matrix)[:3,3]
        hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(
            bpy.context.evaluated_depsgraph_get(),
            Vector(loc), Vector([0,0,1]),
            distance=1.70141e+38)
        room_id=bpy.data.objects[hit_object.name]['room_id']
        frame=bproc.camera.add_camera_pose(room_id,cam2world_matrix)
        arr=find_near_obj(cam2world_matrix)

        for i in range(hide_obj_num):
            #bproc.camera.add_camera_pose(room_id, cam2world_matrix)
            print(arr)
            print(arr[i])
            #frame=bproc.camera.add_camera_pose(room_id, cam2world_matrix)

            bpy.data.objects[arr[i]].location.z-=100
            bpy.data.objects[arr[i]].keyframe_insert(data_path="location", frame=frame)
            #bproc.camera.add_camera_pose(room_id, cam2world_matrix)
        for i in range(hide_obj_num):
            fr=bpy.context.scene.frame_end
            bpy.data.objects[arr[i]].location.z += 100
            bpy.data.objects[arr[i]].keyframe_insert(data_path="location", frame=fr)
            bproc.camera.add_camera_pose(room_id, cam2world_matrix)

        poses += 1
    tries += 1


#save_camera_parameter
config_parser=ConfigParser(silent=True)
config = config_parser.parse( bproc.utility.resolve_resource(config_path), arg)
module=Utility.initialize_modules(config["modules"])

for mod in module:
    mod.run()

# Also render normals
# bproc.renderer.enable_normals_output()
#bproc.renderer.enable_distance_output(activate_antialiasing=False)
bproc.renderer.enable_depth_output(activate_antialiasing=False,output_dir=output_dir)
#bproc.renderer.enable_segmentation_output(map_by=["class", "instance", "name", "jid", "instanceid"],default_values={"jid": "", "instanceid": ""})
data = bproc.renderer.render(output_dir=output_dir)

bproc.renderer.render_segmap(map_by=["class", "instance", "name", "jid", "instanceid"],default_values={"jid": "", "instanceid": ""},output_dir=output_dir)

#print(instance_attribute_maps)
# for i in range(len(class_segmaps)):
#     combined_result_map=[]
#     combined_result_map.append(class_segmaps[i])
#     combined_result_map.append(instance_segmaps[i])
#     resulting_map = np.stack(combined_result_map, axis=2).astype(np.int32)
#
#     fname = output_dir + '/segmap_' + '%4d' % i
#     np.savez_compressed(fname, data=resulting_map)

for i in data:

    print(i)
#     print(data[i])
#print(data)
# write the data to a .hdf5 container

