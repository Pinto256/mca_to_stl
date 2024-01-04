import numpy as np
import math
import os
import zlib
import struct
from stl import mesh

def main():

    x1 = -51
    y1 = -64
    z1 = -26

    x2 = -10
    y2 = 73
    z2 = 4

    region_path = r"C:\Users\pontu\GitHub\mca_to_stl\region"

    model = get_model(x1, y1, z1, x2, y2, z2, region_path)

    ignored_blocks = ['air', 'cave_air', 'water', 'short_grass', 'tall_grass', 'seagrass', 'tall_seagrass']

    for i in range(len(model)):
        for j in range(len(model[0])):
            for k in range(len(model[0][0])):
                if model[i][j][k] in ignored_blocks:
                    model[i][j][k] = False
                else:
                    model[i][j][k] = True

    create_stl(model)

def get_model(x1, y1, z1, x2, y2, z2, region_path):

    # Set x1 <= x2
    if x2 < x1:
        temp = x2
        x2 = x1
        x1 = temp

    # Set y1 <= y2
    if y2 < y1:
        temp = y2
        y2 = y1
        y1 = temp

    # Set z1 <= z2
    if z2 < z1:
        temp = z2
        z2 = z1
        z1 = temp

    chunk_x_1 = math.floor(x1 / 16)
    chunk_x_2 = math.floor(x2 / 16)

    chunk_z_1 = math.floor(z1 / 16)
    chunk_z_2 = math.floor(z2 / 16)

    chunks = [[0 for _ in range(math.floor(z2 / 16) - math.floor(z1 / 16) + 1)] for _ in range(math.floor(x2 / 16) - math.floor(x1 / 16) + 1)]

    num_chunks = len(chunks) * len(chunks[0])

    for i in range(len(chunks)):
        for j in range(len(chunks[i])):
            chunks[i][j] = {"chunk_x": math.floor(((x1 + (16 * i)) / 16)), "chunk_z": math.floor(((z1 + (16 * j)) / 16))}

    region_x_from = math.floor(chunks[0][0]["chunk_x"] / 32)
    region_x_to = math.floor(chunks[-1][0]["chunk_x"] / 32)

    region_z_from = math.floor(chunks[0][0]["chunk_z"] / 32)
    region_z_to = math.floor(chunks[0][-1]["chunk_z"] / 32)

    regions = [f'r.{region_x_from + i}.{region_z_from + j}.mca' for i in range(region_x_to - region_x_from + 1) for j in range(region_z_to - region_z_from + 1)]

    # Read locations headers
    locations_header = [[-1 for _ in range(1024)] for _ in range(len(regions))]
    for i in range(len(regions)):
        with open(os.path.join(region_path, regions[i]), 'rb') as region:
            bytes_read = 0
            while(bytes_read < 4096):
                region.seek(bytes_read)
                locations_header[i][math.floor(bytes_read / 4)] = int.from_bytes(region.read(3), byteorder='big')
                bytes_read += 4

    # Get chunk offsets in region files
    offsets = [[0 for _ in range(num_chunks)] for _ in range(len(regions))]
    for region in range(len(regions)):
        # get chunk offsets in region file regions[region]

        region_x = int(regions[region][regions[region].index('.', 0) + 1:regions[region].index('.', 2)])
        region_z = int(regions[region][regions[region].index('.', 2) + 1: -4])

        chunks_in_region = []
        for i in range(len(chunks)):
            for j in range(len(chunks[i])):
                if math.floor(chunks[i][j]['chunk_x'] / 32) == region_x and math.floor(chunks[i][j]['chunk_z'] / 32) == region_z:
                    chunks_in_region.append(chunks[i][j])

        for i in range(len(chunks_in_region)):
                offsets[region][i] = locations_header[region][((chunks_in_region[i]["chunk_x"] % 32) + (chunks_in_region[i]["chunk_z"] % 32) * 32)]

    # Read data (in NTB format) from offsets
    NBT_data = [0 for _ in range(num_chunks)]
    NBT_data_index = 0
    for r in range(len(regions)):
        for i in range(len(offsets[r])):
            if offsets[r][i] == 0: break
            with open(os.path.join(region_path, regions[r]), 'rb') as region:
                region.seek(offsets[r][i] * 4096)
                length = int.from_bytes(region.read(4), byteorder='big', signed=True)
                #region.seek((offsets[r][i] * 4096) + 4)
                #compression_type = int.from_bytes(region.read(1), byteorder='big')
                region.seek((offsets[r][i] * 4096) + 5)
                NBT_data[NBT_data_index] = zlib.decompress(region.read(length - 1))
                NBT_data_index += 1

    with open('example.dat', 'wb') as f:
        f.write(NBT_data[0])
    
    model = [[["" for _ in range((y2-y1)+1)] for _ in range((z2-z1)+1)] for _ in range((x2-x1)+1)]

    section_min = math.floor(y1 / 16)
    section_max = math.floor(y2 / 16)
    sections = [["" for _ in range(4096)] for _ in range(section_min, section_max + 1)]

    for chunk in range(len(NBT_data)):

        chunk_x = int.from_bytes(NBT_data[chunk][NBT_data[chunk].index(bytes('xPos', 'utf-8')) + 4:NBT_data[chunk].index(bytes('xPos', 'utf-8')) + 8], signed=True)
        chunk_z = int.from_bytes(NBT_data[chunk][NBT_data[chunk].index(bytes('zPos', 'utf-8')) + 4:NBT_data[chunk].index(bytes('zPos', 'utf-8')) + 8], signed=True)

        from_x = max(x1, chunk_x * 16)
        to_x = min(x2, (chunk_x * 16 + 15))

        from_z = max(z1, chunk_z * 16)
        to_z = min(z2, (chunk_z * 16 + 15))

        section_index = -1
        section = -1

        # 24 subsections in a chunk
        for _ in range(24):
            section_index = NBT_data[chunk].index(b'\x00\x0cblock_states', section_index + 1)

            section_y_index = section_index
            section_y_index = NBT_data[chunk].index(b'\x01\x00\x01\x59', section_y_index) + 4
            
            section_y = struct.unpack('b', struct.pack('B', NBT_data[chunk][section_y_index]))[0]
            if section_y < section_min:
                continue
            elif section_y > section_max:
                break
            
            section += 1
            
            palette_index = NBT_data[chunk].index(bytes('palette', 'utf-8'), section_index)
            palette_index_end = NBT_data[chunk].index(b'\x00\x00', palette_index+15)
            #print(NBT_data[chunk][palette_index+15:palette_index_end], "\n")

            palette = []

            # Create block palette list
            while True:
                block_index = NBT_data[chunk].index(bytes('minecraft:', 'utf-8'), palette_index) + len('minecraft:')
                block_index_end = NBT_data[chunk].find(b'\x00', block_index)
                if block_index_end == -1 or block_index_end > palette_index_end:
                    break
                palette.append(NBT_data[chunk][block_index:block_index_end].decode('utf-8'))
                palette_index = block_index_end

            if len(palette) == 1:
                for i in range(4096):
                    sections[section][i] = palette[0]
            else:
                data_bit_length = max(math.ceil(math.log2(len(palette))), 4)
                indecies_in_long = math.floor(64 / data_bit_length)
                data_list_length = math.ceil(4096 / indecies_in_long)
                indecies_in_last_long = math.floor(4096 - (data_list_length - 1) * indecies_in_long)

                data_index = NBT_data[chunk].index(bytes('data', 'utf-8'), section_index)

                #print(NBT_data[chunk][data_index + 8:data_index + 8 + (data_list_length * 8)])

                for i in range(data_list_length - 1):
                    long_value = int.from_bytes(NBT_data[chunk][(data_index + 8) + (i * 8):(data_index + 8) + ((i+1) * 8)])
                    long_value_bin = bin(long_value)
                    for j in range(indecies_in_long):
                        mask = (1 << data_bit_length) - 1
                        shifted_value = long_value >> (j * data_bit_length)
                        masked_result = shifted_value & mask
                        sections[section][(i * indecies_in_long) + j] = palette[masked_result]

                # for last long in indecies list (only fraction of long is used)
                for i in range(indecies_in_last_long):
                    mask = (1 << data_bit_length) - 1
                    shifted_value = long_value >> (i * data_bit_length)
                    masked_result = shifted_value & mask
                    sections[section][indecies_in_long * (data_list_length - 1) + i] = palette[masked_result]
        
            from_y = max(y1, (section_y * 16))
            to_y = min(y2, ((section_y * 16) + 15))

            # write data from section list to model

            for i in range(from_y, to_y + 1):
                for j in range(from_z, to_z + 1):
                    for k in range(from_x, to_x + 1):
                        model[k-x1][j-z1][i-y1] = sections[section][(k % 16) + ((j % 16) * 16) + (i % 16) * 256]
    return model

def create_stl(model):
    cube_list = []

    for i in range(len(model)):
        for j in range(len(model[0])):
            for k in range(len(model[0][0])):
                if model[i][j][k]:
                    cube = create_cube()
                    cube.x += i * 2
                    cube.y += j * 2
                    cube.z += k * 2
                    cube_list.append(cube.data.copy())

    minecraft_mdel = mesh.Mesh(np.concatenate(cube_list))
    minecraft_mdel.save('minecraft_model.stl')
    
def create_cube():
    # Define the 8 vertices of the cube
    vertices = np.array([\
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1]])
    # Define the 12 triangles composing the cube
    faces = np.array([\
    [0,3,1],
    [1,3,2],
    [0,4,7],
    [0,7,3],
    [4,5,6],
    [4,6,7],
    [5,1,2],
    [5,2,6],
    [2,3,6],
    [3,7,6],
    [0,1,5],
    [0,5,4]])
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]

    return cube

if __name__ == '__main__':
    main()