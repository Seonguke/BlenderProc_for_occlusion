import bpy

from blenderproc.python.modules.utility.ItemWriter import ItemWriter
from blenderproc.python.writer.WriterUtility import _WriterUtility
from blenderproc.python.modules.writer.WriterInterface import WriterInterface


class CameraStateWriter(WriterInterface):
    """ Writes the state of all camera poses to a numpy file, if there was no hdf5 file to add them to.

    **Attributes per object**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - fov_x
          - The horizontal FOV.
          - float
        * - fov_y
          - The vertical FOV.
          - float
        * - half_fov_x
          - Half of the horizontal FOV.
          - float
        * - half_fov_y
          - Half of the vertical FOV.
          - float
    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)
       # print(config.get_list("attributes_to_write"))
        #print(config.get_string("output_dir", ""))
        self.object_writer = ItemWriter(_WriterUtility.get_cam_attribute)

    def run(self):
        """ Collect camera and camera object and write them to a file."""
        
        #self.write_attributes_to_file(self.object_writer, [bpy.context.scene.camera], "campose_", "campose", ["cam2world_matrix", "cam_K"])
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data
        cam_pose = (cam, cam_ob)
        self.write_attributes_to_file(self.object_writer, [cam_ob], "campose_", "campose",
                                      ["name", "matrix", "location", "rotation_euler", "fov_x", "fov_y", "shift_x",
                                       "shift_y"])
