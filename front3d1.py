import blenderproc as bproc
import argparse
import os
import numpy as np
import csv
import json
from pathlib import Path
import bpy
from mathutils import Vector, Matrix
from blenderproc.python.modules.utility.ConfigParser import ConfigParser
from blenderproc.python.modules.writer.CameraStateWriter import CameraStateWriter
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.CollisionUtility import CollisionUtility
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=8888, stdoutToServer=True, stderrToServer=True)
'''
pan_root='D:\\3D_Front\\front3d\\00ad8345-45e0-45b3-867d-4a3c88c2517a\\'
root_path='D:\\3D_Front\\'
config_path= 'C:\\Users\\esc15\\PycharmProjects\\pythonProject2\\BlenderProc_for_occlusion\\config.yaml'
output_dir = 'C:\\Users\\esc15\\PycharmProjects\\pythonProject2\\BlenderProc_for_occlusion\\mytest\\output2\\'

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
'''

pan_root='G:\\3D_Front\\front3d\\00ad8345-45e0-45b3-867d-4a3c88c2517a\\'
root_path='G:\\3D_Front\\'
config_path= 'C:\\Users\\lab-com\\Desktop\\myspace\\\BlenderProc_for_occlusion\\config.yaml'
output_dir = 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\mytest\\output\\'

parser = argparse.ArgumentParser()



parser.add_argument("--front",default='G:\\3D_Front\\3D-FRONT\\3D-FRONT\\00ad8345-45e0-45b3-867d-4a3c88c2517a.json', help="Path to the 3D front file")
parser.add_argument("--future_folder", default='G:\\3D_Front\\3D-FUTURE-model\\', help="Path to the 3D Future Model folder.")
parser.add_argument("--front_3D_texture_path", default='G:\\3D_Front\\3D-FRONT-texture\\3D-FRONT-texture\\', help="Path to the 3D FRONT texture folder.")
parser.add_argument("--output_dir",default=output_dir,help="Path to where the data should be saved")


'''
parser.add_argument("--front",default='', help="Path to the 3D front file")
parser.add_argument("--future_folder", default='G:\\3D_Front\\3D-FUTURE-model\\', help="Path to the 3D Future Model folder.")
parser.add_argument("--front_3D_texture_path", default='G:\\3D_Front\\3D-FRONT-texture\\3D-FRONT-texture\\', help="Path to the 3D FRONT texture folder.")
parser.add_argument("--output_dir",default='',help="Path to where the data should be saved")
'''
args = parser.parse_args()

arg=[]
arg.append(args.front)
arg.append(args.future_folder)
arg.append(args.front_3D_texture_path)
arg.append(args.output_dir)

output_dir = args.output_dir

def check_name(name):
    for category_name in ["cabinet", "sofa", "table", "bed","shelf","wardrobe","furniture","bookcase","jewrly"]:
        if category_name in name.lower():
            return True
    return False
def IoU(box1, box2):

    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou
def get_2d_bounding_box(obj,cam2world_matrix):

    """
    https://xiuminglib.readthedocs.io/en/latest/_modules/xiuminglib/blender/camera.html#get_2d_bounding_box
    Gets a 2D bounding box of the object in the camera frame.

    This is different from projecting the 3D bounding box to 2D.

    Args:
        obj (bpy_types.Object): Object of interest.
        cam (bpy_types.Object): Camera.

    Returns:
        numpy.ndarray: 2D coordinates of the bounding box corners.
        Of shape 4-by-2. Corners are ordered counterclockwise, following:

        .. code-block:: none

            (0, 0)
            +------------> (w, 0)
            |           x
            |
            |
            |
            v y (0, h)
    """
    rotmat_cam2cv = Matrix((
        (1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)))
    rotmat_world2cam  = Matrix(cam2world_matrix[:3, :3].T)
    t =Vector(cam2world_matrix[:3,3])

    t_world2cam = rotmat_world2cam @ -t
    rotmat_world2cv = rotmat_cam2cv @ rotmat_world2cam
    t_world2cv = rotmat_cam2cv @ t_world2cam

    int_mat = Matrix(((277.12811279296875, 0.0, 160.0),
                      (0.0, 311.7691345214844, 120.0),
                      (0.0, 0.0, 1.0)))

    ext_mat = Matrix((
        rotmat_world2cv[0][:] + (t_world2cv[0],),
        rotmat_world2cv[1][:] + (t_world2cv[1],),
        rotmat_world2cv[2][:] + (t_world2cv[2],)))
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100.
    w = scene.render.resolution_x * scale
    h = scene.render.resolution_y * scale

    # Get camera matrix
    cam_mat = int_mat @ ext_mat

    # Project all vertices to 2D
    pts = np.vstack([v.co.to_4d() for v in obj.data.vertices]).T # 4-by-N
    world_mat = np.array(obj.matrix_world) # 4-by-4
    cam_mat = np.array(cam_mat) # 3-by-4
    xyw = cam_mat.dot(world_mat.dot(pts)) # 3-by-N
    pts_2d = np.divide(xyw[:2, :], np.tile(xyw[2, :], (2, 1))) # 2-by-N

    # Compute bounding box
    x_min, y_min = np.min(pts_2d, axis=1)
    x_max, y_max = np.max(pts_2d, axis=1)
    if x_min<0:
        x_min=0
    if y_min <0:
        y_min=0
    if x_max > w:
        x_max =w
    if y_max > h:
        y_max =h
    # corners = np.vstack((
    #     np.array([x_min, y_min]),
    #     np.array([x_max, y_min]),
    #     np.array([x_max, y_max]),
    #     np.array([x_min, y_max])))


    corners=[x_min, y_min,x_max,y_max]
    return corners
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
def find_near_obj(cam2world_matrix):
    cam2world_matrix = Matrix(cam2world_matrix)
    sqrt_number_of_rays = 10
    cam_ob = bpy.context.scene.camera
    cam = cam_ob.data
    cam2world_matrix = Matrix(cam2world_matrix)

    frame = cam.view_frame(scene=bpy.context.scene)
    frame = [cam2world_matrix @ v for v in frame]
    vec_x = frame[1] - frame[0]
    vec_y = frame[3] - frame[0]
    position = cam2world_matrix.to_translation()
    arr = set()
    d= dict()
    for x in range(0, sqrt_number_of_rays):
        for y in range(0, sqrt_number_of_rays):
            # Compute current point on plane
            end = frame[0] + vec_x * x / float(sqrt_number_of_rays - 1) + vec_y * y / float(sqrt_number_of_rays - 1)
            # Send ray from the camera position through the current point on the plane
            # has_hit, snapped_location, snapped_normal, face_index, object, matrix = bpy.context.scene.ray_cast
            hit, loc, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(), position,
                                                                     end - position)
            if hit_object["nyu_category_id"] == 1 or hit_object["nyu_category_id"] == 2:
                continue
            else:
                dist = euclidean_distance(position[0], position[1], position[2], loc[0], loc[1], loc[2])
                if hit_object.name in d :
                    d[hit_object.name] = min (dist, d[hit_object.name])
                else:
                    d[hit_object.name]=dist
    return d
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
def save_room_mapping(output_path):
    rooms={}
    for room_obj in bpy.context.scene.objects:
        if "is_room" in room_obj and room_obj["is_room"] == 1:
            floors = list(filter(lambda x: x.name.lower().startswith("floor"), room_obj.children))
            if len(floors)>0:
                floor = floors[0]
                rooms[room_obj.name] = room_obj, floor, room_obj["room_id"]
            else:
                rooms[room_obj.name] = room_obj, floors, room_obj["room_id"]
    output_path = Path(output_path) / f"room_mapping.json"
    with open(output_path, "w") as f:
        name_index_mapping = {obj[2]: name for name, obj in rooms.items()}
        json.dump(name_index_mapping, f, indent=4)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

############## Parameters ##############
bproc.init()
bproc.camera.set_resolution(320, 240)
bproc.camera.set_intrinsics_from_blender_params(lens=1.0471975511, lens_unit="FOV")

mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
nyu_mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_nyu_mapping_origin_uk.csv"))
# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

height = 0.75 # Blender Cam position
room_map=dict()

hide_obj_num=2
#save_pictures = 1
save_pictures = 1

tries = 0
poses = 0

########################################


mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
_,nyu_mapping =read_csv_mapping(nyu_mapping_file)

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


# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]


proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}


import json

#print(args.front)

'''
json_name = args.front
print(json_name)
json_name = os.path.basename(json_name)
print(json_name)
json_name = args.output_dir + 

for i in range(0,save_pictures):
    if (i % 3 == 0):
        print(i)
'''


# Find out the camera poses
while tries < 50000 and poses < save_pictures:
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
        # print(location)
        # print(rotation)

        loc= np.array(cam2world_matrix)[:3,3]

        # Ray cast to find objects
        hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(# Cam pos to hit object , dict => sort
            bpy.context.evaluated_depsgraph_get(),
            Vector(loc), Vector([0,0,1]),
            distance=1.70141e+38)
        room_id=bpy.data.objects[hit_object.name]['room_id']

        d = find_near_obj(cam2world_matrix)
        d = dict(sorted(d.items(), key=lambda item: item[1]))
        arr= list(d.keys()) # Object name

        if len(arr) <hide_obj_num or 'lighting' in arr[0] or 'lighting' in arr[1] or d[arr[0]]<0.1:
            continue


        # approximate bbox and iou
        #Is there occlusion each other?
        bbox = []
        for i in range(hide_obj_num):
            bbox.append(get_2d_bounding_box(bpy.data.objects[arr[i]],cam2world_matrix))

        if IoU(bbox[0],bbox[1])<0.2 or IoU(bbox[0],bbox[1])>0.6 :
           continue


        frame=bproc.camera.add_camera_pose(room_id,cam2world_matrix)

        # Erase objects
        for i in range(hide_obj_num):

            bpy.data.objects[arr[i]].location.z-= 100
            bpy.data.objects[arr[i]].keyframe_insert(data_path="location", frame=frame)
            #bproc.camera.add_camera_pose(room_id, cam2world_matrix)
        # for i in range(hide_obj_num):
        #     print(bpy.data.objects[arr[i]].location)
        fr = bpy.context.scene.frame_end
        bpy.data.objects[arr[1]].location.z = 0
        bpy.data.objects[arr[1]].keyframe_insert(data_path="location", frame=fr)
        bproc.camera.add_camera_pose(room_id, cam2world_matrix)


        for i in reversed(range(hide_obj_num)):
            bpy.data.objects[arr[i]].location.z=0
            bpy.data.objects[arr[i]].keyframe_insert(data_path="location", frame=fr+1)
        bproc.camera.add_camera_pose(room_id, cam2world_matrix)

        # for i in range(hide_obj_num):
        #     print(bpy.data.objects[arr[i]].location)
        poses += 1
    tries += 1

print("\n\n\nLDY1 :", output_dir)
save_room_mapping(output_dir)
#save_camera_parameter
config_parser=ConfigParser(silent=True)
config = config_parser.parse( bproc.utility.resolve_resource(config_path), arg)

'''
config["modules"][0]["config"]["output_dir"] = output_dir
config["modules"][0]["config"]["temp_dir"] = output_dir
'''



module=Utility.initialize_modules(config["modules"])

print("\n\n\nLDY1.1 :", config)
print("\n\n\nLDY1.2 :", module)
print("\n\n\nLDY1.3 :", config["modules"])
print("\n\n\nLDY1.4 :", config["modules"][0])
print("\n\n\nLDY1.41 :", config["modules"][0]["config"])
print("\n\n\nLDY1.42 :", config["modules"][0]["config"]["output_dir"])
print("\n\n\nLDY1.43 :", config["modules"][0]["config"]["temp_dir"])
#print("\n\n\nLDY1.5 :", config["modules"][0][0])


print("\n\n\nLDY1.5 :", module[0])
#print("\n\n\nLDY1.6 :", module[0][0])


print("\n\n\nLDY2 :", output_dir)



'''
config["modules"] = output_dir
config["temp_dir"] = output_dir
'''


for mod in module:
    mod.run()


print("\n\n\nLDY2.1 :", output_dir)


# Render with saved cmara poses
bproc.renderer.enable_depth_output(activate_antialiasing=False,output_dir=output_dir)
#bproc.renderer.enable_depth_output(activate_antialiasing=False,output_dir="./output_LDY")
bproc.renderer.enable_segmentation_output(map_by=["class", "instance", "name", "jid", "instanceid"],default_values={"jid": "", "instanceid": ""})
data = bproc.renderer.render(output_dir=output_dir)
#data = bproc.renderer.render(output_dir="./output_LDY")
bproc.renderer.render_segmap(map_by=["class", "instance", "name", "jid", "instanceid"],default_values={"jid": "", "instanceid": ""},output_dir=output_dir)

print("\n\n\nLDY3 :", output_dir)


