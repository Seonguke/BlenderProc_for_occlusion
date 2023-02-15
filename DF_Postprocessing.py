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


p = 'C:\\Users\\esc15\\PycharmProjects\\pythonProject2\\vox\\'
sem = p + '00ad8345-45e0-45b3-867d-4a3c88c2517a\\segmentation_0.sem'
np.savez(p + '0000.npz', data=data)
reader = BinaryReader(sem)
dimX, dimY, dimZ = reader.read('UINT64', 3)
data = reader.read('float', dimX * dimY * dimZ)
data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)
# data = np.expand_dims(np.reshape(data, (dimX, dimY, dimZ), order='F'), 0).astype(np.float32)

#### Tdf
reader = BinaryReader(p + 'dist_.bin')
dimX, dimY, dimZ = reader.read('UINT64', 3)
data = reader.read('float', dimX * dimY * dimZ)
data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)
data = np.pad(data, ((12, 13), (41, 41), (34, 35)), 'constant', constant_values=12)
data[data > 8] = 12
np.savez(p + 'my_geom.npz', data=data)

#### wighting mask
reader = BinaryReader(p + 'dist_.bin')
dimX, dimY, dimZ = reader.read('UINT64', 3)
data = reader.read('float', dimX * dimY * dimZ)
data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)
data = np.pad(data, ((12, 13), (41, 41), (34, 35)), 'constant', constant_values=12)
data[data > 2] = -1
data[data > 1] = -5
data[data > 0] = -10
data = -data
np.savez(p + 'my_weight.npz', data=data)

arr = []
for item in range(18):
    arr.append((str(item).zfill(4)))


def wrtie_txt():
    with open('../1.txt', 'w') as f:
        for item in range(18):
            f.write(str(item).zfill(4) + '\n')
        f.close()


