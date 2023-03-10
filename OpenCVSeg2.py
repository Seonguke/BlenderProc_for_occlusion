    def _move_and_duplicate_furniture(data: dict, all_loaded_furniture: list) -> List[MeshObject]:
        """
        Move and duplicate the furniture depending on the data in the data json dir.
        After loading each object gets a location based on the data in the json file. Some objects are used more than
        once these are duplicated and then placed.
        Merge objects based on multiple parts. Center the origin to the geometry center. 
        :param data: json data dir. Should contain "scene", which should contain "room"
        :param all_loaded_furniture: all objects which have been loaded in _load_furniture_objs
        :return: The list of loaded mesh objects.
        """
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
                            new_obj.set_cp("type", "Object")  # is an object used for the interesting score
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