import numpy as np

import struct
import numpy as np
import matplotlib.pyplot as plt


class BinaryReaderEOFException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Not enough bytes in file to satisfy read request'


class BinaryReader(object):

    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.typeNames = {
            'int8': 'b',
            'uint8': 'B',
            'int16': 'h',
            'uint16': 'H',
            'int32': 'i',
            'uint32': 'I',
            'int64': 'q',
            'uint64': 'Q',
            'float': 'f',
            'double': 'd',
            'char': 's'}

    def read(self, typeName, times=1):
        typeFormat = self.typeNames[typeName.lower()] * times
        typeSize = struct.calcsize(typeFormat)
        value = self.file.read(typeSize)
        if typeSize != len(value):
            raise BinaryReaderEOFException
        return struct.unpack(typeFormat, value)


def dishow(disp, p):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth2Disparity image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.plot
    plt.show()
    #plt.savefig(p + '.png')
    plt.close()


#p='C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\mytest\\output\\'
folder = '00154c06-2ee2-408a-9664-b8fd74742897'
p='C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\output_VOXEL\\' + folder + '\\'

num = '0027'
#num = str(num)

Tdf = 'dist_' + num + '.bin'
#Tdf_out = 'my_geom_' + num + '.npz'
Tdf_out = 'mask_my_geom_' + num + '.npz'

wighting_mask = 'dist_' + num + '.bin'
#wighting_mask_out = 'my_weight_' + num + '.npz'
wighting_mask_out = 'mask_my_weight_' + num + '.npz'


#### Tdf
#reader = BinaryReader(p + 'dist_0000.bin')
mask= np.load("./frustum_mask.npz")
mask = mask['mask']

reader = BinaryReader(p + Tdf)
dimX, dimY, dimZ = reader.read('UINT64', 3)
data = reader.read('float', dimX * dimY * dimZ)
data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)
data = np.pad(data, ((12, 13), (41, 41), (34, 35)), 'constant', constant_values=12)

data[data > 8] = 12
data[mask==0] = 12

np.savez(p + Tdf_out, data=data)

#### wighting mask
#reader = BinaryReader(p + 'dist_0000.bin')
reader = BinaryReader(p + wighting_mask)
dimX, dimY, dimZ = reader.read('UINT64', 3)
data = reader.read('float', dimX * dimY * dimZ)
data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)
data = np.pad(data, ((12, 13), (41, 41), (34, 35)), 'constant', constant_values=12)
data[data > 2] = -1
data[data > 1] = -5
data[data > 0] = -10
data[mask==0] = -10
data = -data


#np.savez(p + 'my_weight.npz', data=data)
np.savez(p + wighting_mask_out, data=data)





arr = []
for item in range(18):
    arr.append((str(item).zfill(4)))


def wrtie_txt():
    with open('../1.txt', 'w') as f:
        for item in range(18):
            f.write(str(item).zfill(4) + '\n')
        f.close()


