"""Loads the 3D FRONT and FUTURE dataset"""

import json
import os
import warnings
from math import radians
from typing import List, Mapping
from urllib.request import urlretrieve
from pathlib import Path

import bmesh
import bpy
import mathutils
import numpy as np
import bpy
#from src.utility.Utility import Utility
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.types.MeshObjectUtility import MeshObject, create_with_empty_mesh, create_from_blender_mesh
from blenderproc.python.utility.Utility import resolve_path
from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.loader.TextureLoader import load_texture
from blenderproc.python.utility.BlenderUtility import get_centroid, write_ply

def load_front3d(output_dir:str, json_path: str, future_model_path: str, front_3D_texture_path: str, label_mapping: LabelIdMapping,
                 nyu_label_mapping: LabelIdMapping,
                 ceiling_light_strength: float = 0.8, lamp_light_strength: float = 7.0) -> List[MeshObject]:
    """ Loads the 3D-Front scene specified by the given json file.

    :param json_path: Path to the json file, where the house information is stored.
    :param future_model_path: Path to the models used in the 3D-Front dataset.
    :param front_3D_texture_path: Path to the 3D-FRONT-texture folder.
    :param label_mapping: A dict which maps the names of the objects to ids.
    :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
    :param lamp_light_strength: Strength of the emission shader used in each lamp.
    :return: The list of loaded mesh objects.
    """
    json_path = resolve_path(json_path)
    future_model_path = resolve_path(future_model_path)
    front_3D_texture_path = resolve_path(front_3D_texture_path)

    loaded_objects = []
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The given path does not exists: {json_path}")
    if not json_path.endswith(".json"):
        raise FileNotFoundError(f"The given path does not point to a .json file: {json_path}")
    if not os.path.exists(future_model_path):
        raise FileNotFoundError(f"The 3D future model path does not exist: {future_model_path}")

    # load data from json file
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if "scene" not in data:
        raise ValueError(f"There is no scene data in this json file: {json_path}")

    created_objects = _Front3DLoader.create_mesh_objects_from_file(data, front_3D_texture_path,
                                                                   ceiling_light_strength, label_mapping, json_path,nyu_label_mapping)

    all_loaded_furniture = _Front3DLoader.load_furniture_objs(data, future_model_path,
                                                              lamp_light_strength, label_mapping,nyu_label_mapping)

    # created_objects += _Front3DLoader.move_and_duplicate_furniture(data, all_loaded_furniture)
    loaded_objects = _Front3DLoader.move_and_duplicate_furniture(data, all_loaded_furniture)
    _assign_parent_node(data, loaded_objects, created_objects)
    # add an identifier to the obj
    for obj in created_objects:
        obj.set_cp("is_3d_front", True)
    for obj in loaded_objects:
        obj.set_cp("is_3d_front", True)

    #print('go _redraw_walls')
    _redraw_walls(created_objects,output_dir)
    _use_floor_shape_as_ceiling(output_dir)
    # print(x)
    created_objects+=loaded_objects
    created_objectss=[]
    for obj in created_objects:
        try:
            obj.has_cp("is_3d_front")
            created_objectss.append(obj)
            #print(obj.get_name())

            #print(obj.get_cp("room_id"))
            #print(obj.get_cp("category_id"))
        except:
            #print(obj)
            continue

    return created_objectss
def _use_floor_shape_as_ceiling(output_dir):
    for room_obj in bpy.context.scene.objects:
        if "is_room" in room_obj and room_obj["is_room"] == 1 and room_obj.name != "unassigned":
            print(room_obj.name)
            # Get floor and ceiling pair
            floor_obj = None
            ceiling_obj = None
            for obj in room_obj.children:
                obj['room_id'] = room_obj["room_id"]
                if obj.name.startswith("Floor") and obj["nyu_category_id"] == 2 and floor_obj is None:
                    floor_obj = obj

                if obj.name.startswith("Ceiling") and obj["nyu_category_id"] == 22 and ceiling_obj is None:
                    ceiling_obj = obj

            if floor_obj is None or ceiling_obj is None:
                print(f"Room {room_obj.name} has no floor or ceiling")
                continue

            ceiling_center = get_centroid(ceiling_obj)
            ceiling_obj.data = floor_obj.data.copy()

            ceiling_obj["room_id"] = room_obj["room_id"]

            for v in ceiling_obj.data.vertices:
                v.co.z = ceiling_center.z

            # assign material
            mat = bpy.data.materials.new(name="ceiling_material")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            # create a principled node and set the default color
            principled_node = Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled")
            principled_node.inputs["Base Color"].default_value = mathutils.Vector([255, 255, 255, 255]) / 255.0
            # if the object is a ceiling add some light output
            links = mat.node_tree.links
            mix_node = nodes.new(type='ShaderNodeMixShader')
            output = Utility.get_the_one_node_with_type(nodes, 'OutputMaterial')
            Utility.insert_node_instead_existing_link(links, principled_node.outputs['BSDF'],
                                                      mix_node.inputs[2], mix_node.outputs['Shader'],
                                                      output.inputs['Surface'])
            # The light path node returns 1, if the material is hit by a ray coming from the camera,
            # else it returns 0. In this way the mix shader will use the principled shader for rendering
            # the color of the lightbulb itself, while using the emission shader for lighting the scene.
            light_path_node = nodes.new(type='ShaderNodeLightPath')
            links.new(light_path_node.outputs['Is Camera Ray'], mix_node.inputs['Fac'])

            emission_node = nodes.new(type='ShaderNodeEmission')
            # use the same color for the emission light then for the ceiling itself
            emission_node.inputs["Color"].default_value = mathutils.Vector([255, 255, 255, 255]) / 255.0
            ceiling_light_strength = 0.8
            emission_node.inputs["Strength"].default_value = ceiling_light_strength

            links.new(emission_node.outputs["Emission"], mix_node.inputs[1])

            # as this material was just created the material is just appened to the empty list
            ceiling_obj.data.materials.clear()
            ceiling_obj.data.materials.append(mat)


            output_path = Path(output_dir) / room_obj.name / "ceiling.ply"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            vertices = [v.co.to_tuple() for v in ceiling_obj.data.vertices]
            indices = [[v for v in face.vertices] for face in ceiling_obj.data.polygons]
            write_ply(vertices, indices, output_path)


def _redraw_walls(created_objects,output_dir):
    # For each room
    # col = bpy.data.collections.get("Collection")
    # bpy.ops.object.mode_set(mode='OBJECT')

    # objects = [obj for obj in bpy.context.scene.objects]
    col = bpy.data.collections.get("Collection")

    for room_obj in bpy.context.scene.objects:
        '''
        room_instance_id = room["instanceid"]
        room_obj = create_with_empty_mesh(room_instance_id, None)
        room_obj.set_cp("3D_future_type", room["type"])
        room_obj.set_cp("is_room", True)
        room_obj.set_cp("room_id", room_id)'''
        # Check if object is from type room and has bbox
        #print(room_obj)
        if "is_room" in room_obj and room_obj["is_room"] == 1 and room_obj.name != "unassigned":
            room_id = room_obj["room_id"]
            room_name = room_obj.name
            # for c_obj in created_objects:
            #     print(c_obj.get_name())
                #if room_name == c_obj.get_name():
                #    print('room_name')
            #room_p=room_obj.parent

            room_obj_ = MeshObject(room_obj)
            # Get floor and ceiling pair
            floor_objs = []
            ceiling_objs = []
            for obj in room_obj.children:
                if obj.name.startswith("Floor") and obj["nyu_category_id"] == 2:
                    floor_objs.append(obj)

                if obj.name.startswith("Ceiling") and obj["nyu_category_id"] == 22:
                    ceiling_objs.append(obj)

            if not floor_objs or not ceiling_objs:
                print(f"Room {room_obj.name} has no floor or ceiling")
                continue

            num_floors = len(floor_objs)
            if len(floor_objs) == 1:
                floor_obj = floor_objs[0]
            else:
                # unselect everything
                for o in bpy.data.objects:
                    o.select_set(False)
                    bpy.context.view_layer.objects.active = o

                # join multiple floor objects into one
                print("Join floor: ", *[o.name for o in floor_objs])
                for o in floor_objs:
                    o.select_set(True)
                    bpy.context.view_layer.objects.active = o

                bpy.ops.object.join()
                floor_obj = bpy.context.selected_objects[0]
                floor_obj["num_floors"] = num_floors

                # unselect everything
                for o in bpy.data.objects:
                    o.select_set(False)

            if 2 > 1:  # always save
                output_path = Path(output_dir) / room_obj.name / f"floor.ply"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                vertices = [v.co.to_tuple() for v in floor_obj.data.vertices]
                indices = [[v for v in face.vertices] for face in floor_obj.data.polygons]
                write_ply(vertices, indices, output_path)

            if len(ceiling_objs) == 1:
                ceiling_obj = ceiling_objs[0]
            else:
                # join multiple floor objects into one
                print("Join ceiling: ", *[o.name for o in ceiling_objs])
                for o in ceiling_objs:
                    o.select_set(True)
                    bpy.context.view_layer.objects.active = o

                bpy.ops.object.join()
                ceiling_obj = bpy.context.selected_objects[0]
                # unselect everything
                for o in bpy.data.objects:
                    o.select_set(False)
                    bpy.context.view_layer.objects.active = o

            print(floor_obj.name, ceiling_obj.name)

            floor_center = get_centroid(floor_obj)
            ceiling_center = get_centroid(ceiling_obj)

            # Determine height
            height = ceiling_center.z - floor_center.z

            # Select floor, get boundary edges
            # bpy.ops.object.mode_set(mode='OBJECT')
            #        bpy.ops.object.select_all(action='DESELECT')

            for o in bpy.data.objects:
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.mode_set(mode='OBJECT')
                o.select_set(False)

            floor_obj.select_set(True)
            bpy.context.view_layer.objects.active = floor_obj

            # Merge close vertices, 3D-Front meshes do not share vertices
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=0.05)
            # floor_obj.update()
            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.ops.object.mode_set(mode='EDIT')
            #        bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.region_to_loop()
            #        bpy.ops.object.mode_set(mode='OBJECT')
            #        bpy.ops.object.mode_set(mode='EDIT')

            # For each edge in polygon
            bm = bmesh.from_edit_mesh(floor_obj.data)

            # Create plane from floor to ceiling
            edge_index = 0
            for edge in bm.edges:
                if edge.select:
                    # print(edge.index, edge.verts[0].co, edge.verts[1].co)
                    mesh = bpy.data.meshes.new(f"Wall_{edge.index}_mesh")
                    #obj = bpy.data.objects.new(mesh.name, mesh)
                    obj = create_from_blender_mesh(mesh,f"Wall_{edge.index}")
                    instanceid=f"wall/{room_obj['room_id']}_{edge.index}"


                    obj.set_parent(room_obj_)
                    obj.set_cp("is_3D_future",True)
                    #obj.set_cp("category_id", 13)
                    obj.set_cp("category_id", 1)
                    obj.set_cp("nyu_category_id",1)
                    obj.set_cp("room_id",room_id)
                    obj.set_cp("instanceid",instanceid)
                    obj.set_cp("is_3d_front", True)
                    #obj.name = mesh.name
                    #obj.parent = room_obj

                    #obj["is_3d_front"] = True
                    #obj["is_3D_future"]=True
                    #obj["nyu_category_id"] = 1
                    #obj["category_id"] = 13
                    #obj["room_id"] = room_obj["room_id"]
                    #obj["instanceid"] = f"wall/{room_obj['room_id']}_{edge.index}"
                    #obj.set_cp("is_3d_front",True)
                    created_objects.append(obj)
                    #bpy.context.collection.objects.link(obj)
                    v0 = edge.verts[0].co
                    v1 = edge.verts[1].co

                    v2 = v0.copy()
                    v2.z += height

                    v3 = v1.copy()
                    v3.z += height

                    # print("Quad", v0, v1, v2, v3)
                    faces = [(0, 3, 2), (0, 1, 3)]
                    mesh.from_pydata([v0.to_tuple(), v1.to_tuple(), v2.to_tuple(), v3.to_tuple()], [], faces)
                    #mat = bpy.data.materials.new(name=mesh.name + "_material")
                    mat = MaterialLoaderUtility.create(name=mesh.name + "_material")
                    #mat.use_nodes = True
                    #nodes = mat.node_tree.nodes
                    # create a principled node and set the default color
                    #principled_node = Utility.get_the_one_node_with_type( nodes,"BsdfPrincipled")
                    principled_node = mat.get_the_one_node_with_type("BsdfPrincipled")
                    principled_node.inputs["Base Color"].default_value = mathutils.Vector([255, 255, 255, 255]) / 255.0
                    obj.add_material(mat)
                    #obj.data.materials.append(mat)
                    if 2 > 1:
                        output_path = Path(
                            output_dir) / room_name / f"wall_{edge_index:02d}.ply"
                        output_path.parent.mkdir(exist_ok=True, parents=True)
                        vertices = [v0.to_tuple(), v1.to_tuple(), v2.to_tuple(), v3.to_tuple()]
                        indices = [(0, 3, 2), (0, 1, 3)]
                        write_ply(vertices, indices, output_path)

                    edge_index += 1

            bpy.ops.object.mode_set(mode='OBJECT')
            #print(x)


def _assign_parent_node(data, loaded_objects, created_objects):
    assigned_objects = set()
    for room_id, room in enumerate(data["scene"]["room"]):
        # create new node for each room
        room_instance_id = room["instanceid"]
        room_obj = create_with_empty_mesh(room_instance_id, None)
        room_obj.set_cp("3D_future_type", room["type"])
        room_obj.set_cp("is_room", True)
        room_obj.set_cp("room_id", room_id)
        #bpy.context.scene.collection.objects.link(room_obj)

        # for each object in that room assign newly created node as parent
        for child in room["children"]:
            for obj in created_objects:

                if obj.get_cp("uid") == child["ref"]:
                    #print(obj)
                    #print(room_obj)
                    obj.set_parent(room_obj)#obj는 mesh room_obj empty
                    #print(obj.get_parent())
                    #print(f"Assign {obj['uid']} to {room_instance_id}")
                    assigned_objects.add(obj)

            for obj in loaded_objects:
                #print(obj.get_cp("room_id"))
                if obj.get_cp("room_id") == room_id:
                    #print(obj)
                    #print(room_obj)
                    #obj.parent = room_obj
                    obj.set_parent(room_obj)
                    #print(f"Assign {obj['uid']} to {room_instance_id}")
                    assigned_objects.add(obj)
                    # break

    all_objects = set([obj for obj in created_objects]).union([obj for obj in loaded_objects])
    missing_objects = all_objects.difference(assigned_objects)

    #node_obj = bpy.data.objects.new("unassigned", None)
    node_obj = create_with_empty_mesh("unassigned", None)
    #bpy.context.scene.collection.objects.link(node_obj)
    for obj in missing_objects:
        obj.set_parent(node_obj)

    print(f"------------Missing objects-------------:", "\n".join([s["uid"] for s in missing_objects]))


class _Front3DLoader:
    """ Loads the 3D-Front dataset.

    https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset

    Each object gets the name based on the category/type, on top of that you can use a mapping specified in the
    resources/front_3D folder.

    The dataset already supports semantic segmentation with either the 3D-Front classes or the nyu classes.
    As we have created this mapping ourselves it might be faulty.

    The Front3DLoader creates automatically lights in the scene, by adding emission shaders to the ceiling and lamps.
    """

    @staticmethod
    def extract_hash_nr_for_texture(given_url: str, front_3D_texture_path: str) -> str:
        """
        Constructs the path of the hash folder and checks if the texture is available if not it is downloaded

        :param given_url: The url of the texture
        :param front_3D_texture_path: The path to where the texture are saved
        :return: The hash id, which is used in the url
        """
        # extract the hash nr from the given url
        hash_nr = given_url.split("/")[-2]
        hash_folder = os.path.join(front_3D_texture_path, hash_nr)
        if not os.path.exists(hash_folder):
            # download the file
            os.makedirs(hash_folder)
            warnings.warn(f"This texture: {hash_nr} could not be found it will be downloaded.")
            # replace https with http as ssl connection out of blender are difficult
            urlretrieve(given_url.replace("https://", "http://"), os.path.join(hash_folder, "texture.png"))
            if not os.path.exists(os.path.join(hash_folder, "texture.png")):
                raise Exception(f"The texture could not be found, the following url was used: "
                                f"{front_3D_texture_path}, this is the extracted hash: {hash_nr}, "
                                f"given url: {given_url}")
        return hash_folder

    @staticmethod
    def get_used_image(hash_folder_path: str, saved_image_dict: Mapping[str, bpy.types.Texture]) -> bpy.types.Texture:
        """
        Returns a texture object for the given hash_folder_path, the textures are stored in the saved_image_dict,
        to avoid that texture are loaded multiple times

        :param hash_folder_path: Path to the hash folder
        :param saved_image_dict: Dict which maps the hash_folder_paths to bpy.types.Texture
        :return: The loaded texture bpy.types.Texture
        """
        if hash_folder_path in saved_image_dict:
            ret_used_image = saved_image_dict[hash_folder_path]
        else:
            textures = load_texture(hash_folder_path)
            if len(textures) != 1:
                raise Exception(f"There is not just one texture: {len(textures)}")
            ret_used_image = textures[0].image
            saved_image_dict[hash_folder_path] = ret_used_image
        return ret_used_image

    @staticmethod
    def create_mesh_objects_from_file(data: dict, front_3D_texture_path: str, ceiling_light_strength: float,
                                      label_mapping: LabelIdMapping, json_path: str,nyu_label_mapping) -> List[MeshObject]:
        """
        This creates for a given data json block all defined meshes and assigns the correct materials.
        This means that the json file contains some mesh, like walls and floors, which have to built up manually.

        It also already adds the lighting for the ceiling

        :param data: json data dir. Must contain "material" and "mesh"
        :param front_3D_texture_path: Path to the 3D-FRONT-texture folder.
        :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :param json_path: Path to the json file, where the house information is stored.
        :return: The list of loaded mesh objects.
        """
        # extract all used materials -> there are more materials defined than used
        used_materials = []
        for mat in data["material"]:
            used_materials.append({"uid": mat["uid"], "texture": mat["texture"],
                                   "normaltexture": mat["normaltexture"], "color": mat["color"]})
        #ignore_object_types=["SlabSide","Front"]
        ignore_object_types = ["WallOuter", "WallBottom", "WallTop", "Pocket", "SlabSide", "SlabBottom", "SlabTop",
                               "Front", "Back", "Baseboard", "Door", "Window", "BayWindow", "Hole", "WallInner", "Beam"]
        
        

        created_objects = []
        # maps loaded images from image file path to bpy.type.image
        saved_images = {}
        saved_normal_images = {}
        # materials based on colors to avoid recreating the same material over and over
        used_materials_based_on_color = {}
        # materials based on texture to avoid recreating the same material over and over
        used_materials_based_on_texture = {}
        for mesh_data in data["mesh"]:
            # extract the obj name, which also is used as the category_id name
            used_obj_name = mesh_data["type"].strip()
            if used_obj_name == "":
                used_obj_name = "void"
            if "material" not in mesh_data:
                warnings.warn(f"Material is not defined for {used_obj_name} in this file: {json_path}")
                continue
            # create a new mesh
            if used_obj_name in ignore_object_types:
                print(f"Ignore {used_obj_name}")
                continue
            
            '''
            LDY_object_types = ["bed", "Ceiling", "Floor"]
            if (used_obj_name in LDY_object_types) != True:
                print(f"1 LDY pass {used_obj_name}")
                continue
    
            print(f"2 LDY check {used_obj_name}")
            '''


            obj = create_with_empty_mesh(used_obj_name, used_obj_name + "_mesh")
            obj.set_cp("uid",mesh_data["uid"])
            created_objects.append(obj)

            # set two custom properties, first that it is a 3D_future object and second the category_id
            obj.set_cp("is_3D_future", True)
            obj.set_cp("category_id", nyu_label_mapping[used_obj_name.lower()])
            #obj.set_cp("category_id", label_mapping.id_from_label(used_obj_name.lower()))
            obj.set_cp("nyu_category_id", nyu_label_mapping[used_obj_name.lower()])

            # get the material uid of the current mesh data
            current_mat = mesh_data["material"]
            used_mat = None
            # search in the used materials after this uid
            for u_mat in used_materials:
                if u_mat["uid"] == current_mat:
                    used_mat = u_mat
                    break
            # If there should be a material used
            if used_mat:
                if used_mat["texture"]:
                    # extract the has folder is from the url and download it if necessary
                    hash_folder = _Front3DLoader.extract_hash_nr_for_texture(used_mat["texture"], front_3D_texture_path)
                    if hash_folder in used_materials_based_on_texture and "ceiling" not in used_obj_name.lower():
                        mat = used_materials_based_on_texture[hash_folder]
                        obj.add_material(mat)
                    else:
                        # Create a new material
                        mat = MaterialLoaderUtility.create(name=used_obj_name + "_material")
                        principled_node = mat.get_the_one_node_with_type("BsdfPrincipled")
                        if used_mat["color"]:
                            principled_node.inputs["Base Color"].default_value = mathutils.Vector(
                                used_mat["color"]) / 255.0

                        used_image = _Front3DLoader.get_used_image(hash_folder, saved_images)
                        mat.set_principled_shader_value("Base Color", used_image)

                        if "ceiling" in used_obj_name.lower():
                            mat.make_emissive(ceiling_light_strength,
                                              emission_color=mathutils.Vector(used_mat["color"]) / 255.0)

                        if used_mat["normaltexture"]:
                            # get the used image based on the normal texture path
                            # extract the has folder is from the url and download it if necessary
                            hash_folder = _Front3DLoader.extract_hash_nr_for_texture(used_mat["normaltexture"],
                                                                                     front_3D_texture_path)
                            used_image = _Front3DLoader.get_used_image(hash_folder, saved_normal_images)

                            # create normal texture
                            normal_texture = MaterialLoaderUtility.create_image_node(mat.nodes, used_image, True)
                            normal_map = mat.nodes.new("ShaderNodeNormalMap")
                            normal_map.inputs["Strength"].default_value = 1.0
                            mat.links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
                            # connect normal texture to principled shader
                            mat.set_principled_shader_value("Normal", normal_map.outputs["Normal"])

                        obj.add_material(mat)
                        used_materials_based_on_texture[hash_folder] = mat
                # if there is a normal color used
                elif used_mat["color"]:
                    used_hash = tuple(used_mat["color"])
                    if used_hash in used_materials_based_on_color and "ceiling" not in used_obj_name.lower():
                        mat = used_materials_based_on_color[used_hash]
                    else:
                        # Create a new material
                        mat = MaterialLoaderUtility.create(name=used_obj_name + "_material")
                        # create a principled node and set the default color
                        principled_node = mat.get_the_one_node_with_type("BsdfPrincipled")
                        principled_node.inputs["Base Color"].default_value = mathutils.Vector(used_mat["color"]) / 255.0
                        # if the object is a ceiling add some light output
                        if "ceiling" in used_obj_name.lower():
                            mat.make_emissive(ceiling_light_strength,
                                              emission_color=mathutils.Vector(used_mat["color"]) / 255.0)
                        else:
                            used_materials_based_on_color[used_hash] = mat

                    # as this material was just created the material is just append it to the empty list
                    obj.add_material(mat)

            # extract the vertices from the mesh_data
            vert = [float(ele) for ele in mesh_data["xyz"]]
            # extract the faces from the mesh_data
            faces = mesh_data["faces"]
            # extract the normals from the mesh_data
            normal = [float(ele) for ele in mesh_data["normal"]]

            # map those to the blender coordinate system
            num_vertices = int(len(vert) / 3)
            vertices = np.reshape(np.array(vert), [num_vertices, 3])
            normal = np.reshape(np.array(normal), [num_vertices, 3])
            # flip the first and second value
            vertices[:, 1], vertices[:, 2] = vertices[:, 2], vertices[:, 1].copy()
            normal[:, 1], normal[:, 2] = normal[:, 2], normal[:, 1].copy()
            # reshape back to a long list
            vertices = np.reshape(vertices, [num_vertices * 3])
            normal = np.reshape(normal, [num_vertices * 3])

            # add this new data to the mesh object
            mesh = obj.get_mesh()
            mesh.vertices.add(num_vertices)
            mesh.vertices.foreach_set("co", vertices)
            mesh.vertices.foreach_set("normal", normal)

            # link the faces as vertex indices
            num_vertex_indicies = len(faces)
            mesh.loops.add(num_vertex_indicies)
            mesh.loops.foreach_set("vertex_index", faces)

            # the loops are set based on how the faces are a ranged
            num_loops = int(num_vertex_indicies / 3)
            mesh.polygons.add(num_loops)
            # always 3 vertices form one triangle
            loop_start = np.arange(0, num_vertex_indicies, 3)
            # the total size of each triangle is therefore 3
            loop_total = [3] * num_loops
            mesh.polygons.foreach_set("loop_start", loop_start)
            mesh.polygons.foreach_set("loop_total", loop_total)

            # the uv coordinates are reshaped then the face coords are extracted
            uv_mesh_data = [float(ele) for ele in mesh_data["uv"] if ele is not None]
            # bb1737bf-dae6-4215-bccf-fab6f584046b.json includes one mesh which only has no UV mapping
            if uv_mesh_data:
                uv = np.reshape(np.array(uv_mesh_data), [num_vertices, 2])
                used_uvs = uv[faces, :]
                # and again reshaped back to the long list
                used_uvs = np.reshape(used_uvs, [2 * num_vertex_indicies])

                mesh.uv_layers.new(name="new_uv_layer")
                mesh.uv_layers[-1].data.foreach_set("uv", used_uvs)
            else:
                warnings.warn(f"This mesh {obj.get_name()} does not have a specified uv map!")

            # this update converts the upper data into a mesh
            mesh.update()
        for room in data["scene"]["room"]:
            # for each object in that room
            for child in room["children"]:
                if "mesh" in child["instanceid"]:
                    # find the object where the uid matches the child ref id
                    for obj in created_objects:
                        if obj.get_cp("uid") == child["ref"]:
                            obj.set_cp("instanceid", child["instanceid"])


        # the generation might fail if the data does not line up
        # this is not used as even if the data does not line up it is still able to render the objects
        # We assume that not all meshes in the dataset do conform with the mesh standards set in blender
        # result = mesh.validate(verbose=False)
        # if result:
        #    raise Exception("The generation of the mesh: {} failed!".format(used_obj_name))

        return created_objects

    @staticmethod
    def load_furniture_objs(data: dict, future_model_path: str, lamp_light_strength: float,
                            label_mapping: LabelIdMapping,nyu_label_mapping) -> List[MeshObject]:
        """
        Load all furniture objects specified in the json file, these objects are stored as "raw_model.obj" in the
        3D_future_model_path. For lamp the lamp_light_strength value can be changed via the config.

        :param data: json data dir. Should contain "furniture"
        :param future_model_path: Path to the models used in the 3D-Front dataset.
        :param lamp_light_strength: Strength of the emission shader used in each lamp.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :return: The list of loaded mesh objects.
        """
        # mapping_file = resolve_path(os.path.join("front_3D", "3D_front_mapping.csv"))

        # collect all loaded furniture objects
        all_objs = []
        # for each furniture element
        #print("LDY check 0 :", data["furniture"])
        for ele in data["furniture"]:


            '''
            print("LDY check 0.1 :", ele)
            print("LDY check 0.1 :", ele["jid"])
            print("LDY check 0.1 :", ele["title"])
            '''



            # create the paths based on the "jid"
            folder_path = os.path.join(future_model_path, ele["jid"])
            obj_file = os.path.join(folder_path, "raw_model.obj")
            # if the object exists load it -> a lot of object do not exist
            # we are unsure why this is -> we assume that not all objects have been made public
            if os.path.exists(obj_file) and not "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                # load all objects from this .obj file
                objs = load_obj(filepath=obj_file)
                # extract the name, which serves as category id
                used_obj_name = ""

                #LDY <-
                '''
                if ("category" in ele) == False:
                    continue
                '''
                '''
                if ("category" in ele) == True:
                    print("LDY -------------------------", ele["category"])
                    if (ele["category"] == "bed"):
                        print("LDY Good -------------------------")
                '''
                #LDY ->



                if "category" in ele:
                    used_obj_name = ele["category"]
                elif "title" in ele:
                    used_obj_name = ele["title"]
                    if "/" in used_obj_name:
                        used_obj_name = used_obj_name.split("/")[0]
                if used_obj_name == "":
                    used_obj_name = "others"
                for obj in objs:
                    obj.set_name(used_obj_name)
                    # add some custom properties
                    obj.set_cp("uid", ele["uid"])
                    # this custom property determines if the object was used before
                    # is needed to only clone the second appearance of this object
                    obj.set_cp("is_used", False)
                    obj.set_cp("is_3D_future", True)
                    obj.set_cp("3D_future_type", "Non-Object")  # is an non object used for the interesting score
                    # set the category id based on the used obj name
                    category_key = used_obj_name.lower()

                    #print("LDY key", category_key) #others, armchair...
                    
                    try:
                        obj.set_cp("category_id", nyu_label_mapping[used_obj_name.lower()])
                        #obj.set_cp("category_id", label_mapping.id_from_label(used_obj_name.lower()))
                        obj.set_cp("nyu_category_id", nyu_label_mapping[used_obj_name.lower()])

                        #print("LDY category_id", nyu_label_mapping[used_obj_name.lower()])
                        #print(("LDY nyu_category_id", nyu_label_mapping[used_obj_name.lower()]))
                    except:
                        print(f"{category_key} not in mapping")
                        obj.set_cp("category_id", -1)
                        obj.set_cp("nyu_category_id", -1)
                    # walk over all materials


                    '''
                    checking = nyu_label_mapping[used_obj_name.lower()]
                    if (checking == 4):
                        print("LDY it is a bed")
                        continue
                    '''

                    for mat in obj.get_materials():
                        if mat is None:
                            continue
                        principled_node = mat.get_nodes_with_type("BsdfPrincipled")
                        if "bed" in used_obj_name.lower() or "sofa" in used_obj_name.lower():
                            if len(principled_node) == 1:
                                principled_node[0].inputs["Roughness"].default_value = 0.5
                        is_lamp = "lamp" in used_obj_name.lower()
                        if len(principled_node) == 0 and is_lamp:
                            # this material has already been transformed
                            continue
                        if len(principled_node) == 1:
                            principled_node = principled_node[0]
                        else:
                            raise ValueError(f"The amount of principle nodes can not be more than 1, "
                                             f"for obj: {obj.get_name()}!")

                        # Front3d .mtl files contain emission color which make the object mistakenly emissive
                        # => Reset the emission color
                        principled_node.inputs["Emission"].default_value[:3] = [0, 0, 0]

                        # For each a texture node
                        image_node = mat.new_node('ShaderNodeTexImage')
                        # and load the texture.png
                        base_image_path = os.path.join(folder_path, "texture.png")
                        image_node.image = bpy.data.images.load(base_image_path, check_existing=True)
                        mat.link(image_node.outputs['Color'], principled_node.inputs['Base Color'])
                        # if the object is a lamp, do the same as for the ceiling and add an emission shader
                        if is_lamp:
                            mat.make_emissive(lamp_light_strength)

                all_objs.extend(objs)
            elif "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                warnings.warn(f"This file {obj_file} was skipped as it can not be read by blender.")
            #print("LDY check : ", all_objs)
        return all_objs

    @staticmethod
    def move_and_duplicate_furniture(data: dict, all_loaded_furniture: list) -> List[MeshObject]:
        """
        Move and duplicate the furniture depending on the data in the data json dir.
        After loading each object gets a location based on the data in the json file. Some objects are used more than
        once these are duplicated and then placed.

        :param data: json data dir. Should contain "scene", which should contain "room"
        :param all_loaded_furniture: all objects which have been loaded in load_furniture_objs
        :return: The list of loaded mesh objects.
        """


        '''
        # this rotation matrix rotates the given quaternion into the blender coordinate system
        blender_rot_mat = mathutils.Matrix.Rotation(radians(-90), 4, 'X')
        created_objects = []
        # for each room
        for room_id, room in enumerate(data["scene"]["room"]):
            # for each object in that room
            for child in room["children"]:
                if "furniture" in child["instanceid"]:
                    # find the object where the uid matches the child ref id
                    for obj in all_loaded_furniture:
                        
                        
                        #print("LDY check :", obj.get_cp("category_id"))
                        #if obj.get_cp("category_id") == 4:
                        #    print("LDY check :", obj.get_cp("category_id"))
                        #    continue
                        


                        if obj.get_cp("uid") == child["ref"]:
                            # if the object was used before, duplicate the object and move that duplicated obj
                            if obj.get_cp("is_used"):
                                new_obj = obj.duplicate()
                            else:
                                # if it is the first time use the object directly
                                new_obj = obj
                            created_objects.append(new_obj)
                            new_obj.set_cp("is_used", True)
                            new_obj.set_cp("room_id", room_id)
                            new_obj.set_cp("3D_future_type", "Object")  # is an object used for the interesting score
                            new_obj.set_cp("coarse_grained_class", new_obj.get_cp("category_id"))
                            # this flips the y and z coordinate to bring it to the blender coordinate system
                            new_obj.set_location(mathutils.Vector(child["pos"]).xzy)
                            new_obj.set_scale(child["scale"])
                            # extract the quaternion and convert it to a rotation matrix
                            rotation_mat = mathutils.Quaternion(child["rot"]).to_euler().to_matrix().to_4x4()
                            # transform it into the blender coordinate system and then to an euler
                            new_obj.set_rotation_euler((blender_rot_mat @ rotation_mat).to_euler())
        return created_objects
        '''

        
        # this rotation matrix rotates the given quaternion into the blender coordinate system
        blender_rot_mat = mathutils.Matrix.Rotation(radians(-90), 4, 'X')
        created_objects = []
                                
        equal_objs = []
        prev_uid = ''
        # for each room
        for room_id, room in enumerate(data["scene"]["room"]):
            # for each object in that room
            for child in room["children"]:
                if "furniture" in child["instanceid"]:
                    # find the object where the uid matches the child ref id
                    for obj in all_loaded_furniture:
                        if obj.get_cp("uid") == child["ref"]:
                            # if the object was used before, duplicate the object and move that duplicated obj
                            if obj.get_cp("is_used"):
                                new_obj = obj.duplicate()
                            else:
                                # if it is the first time use the object directly
                                new_obj = obj
                            new_obj.set_cp("is_used", True)
                            new_obj.set_cp("room_id", room_id)
                            #new_obj.set_cp("type", "Object")  # is an object used for the interesting score
                            new_obj.set_cp("3D_future_type", "Object")
                            new_obj.set_cp("coarse_grained_class", new_obj.get_cp("category_id"))
                            # this flips the y and z coordinate to bring it to the blender coordinate system
                            new_obj.set_location(mathutils.Vector(child["pos"]).xzy)
                            new_obj.set_scale(child["scale"])
                            # extract the quaternion and convert it to a rotation matrix
                            rotation_mat = mathutils.Quaternion(child["rot"]).to_euler().to_matrix().to_4x4()
                            # transform it into the blender coordinate system and then to an euler
                            new_obj.set_rotation_euler((blender_rot_mat @ rotation_mat).to_euler())
                            



                            if new_obj.get_cp("uid") == prev_uid:
                                equal_objs[-1].append(new_obj)
                            else:
                                equal_objs.append([new_obj])
                                prev_uid = new_obj.get_cp("uid")



        for i in equal_objs:
            if len(i) == 1:
                created_objects.append(i[0])
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

            else:
                for obj in bpy.context.selected_objects:
                    obj.select_set(False)
                
                if i[0].blender_obj.type == "MESH":
                    bpy.context.view_layer.objects.active = None
                    print(i[0].blender_obj.name)
                    for obj in i:
                        obj.blender_obj.select_set(True)

                    bpy.context.view_layer.objects.active = i[0].blender_obj
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

                    created_objects.append(i[0])
                    print("JOINED")
                    bpy.ops.object.join()

        return created_objects
        