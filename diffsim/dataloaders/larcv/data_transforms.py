import numpy

'''
This is a torch-free file that exists to massage data
From sparse to dense or dense to sparse, etc.

This can also convert from sparse to sparse to rearrange formats
For example, larcv BatchFillerSparseTensor2D (and 3D) output data
with the format of
    [B, N_planes, Max_voxels, N_features]

where N_features is 2 or 3 depending on whether or not values are included
(or 3 or 4 in the 3D case)

# The input of a pointnet type format can work with this, but SparseConvNet
# requires a tuple of (coords, features, [batch_size, optional])


'''


def larcv_edeps(input_array, meta):

    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]


    voxel_size = meta['size'][0] / meta['n_voxels'][0]

    # Slice the data apart:
    x_coords = voxel_size[0] * input_array[:,:,0] + meta['origin'][0,0]
    y_coords = voxel_size[1] * input_array[:,:,1] + meta['origin'][0,1]
    z_coords = voxel_size[2] * input_array[:,:,2] + meta['origin'][0,2]
    val_coords = input_array[:,:,3]


    non_zero_locs = val_coords != 0.0

    # Compute the mask:
    mask = numpy.logical_not(non_zero_locs)

    x_coords[mask] = 0.0
    y_coords[mask] = 0.0
    z_coords[mask] = 0.0


    # batch_index, x_index, y_index, z_index voxel_index = numpy.where(filled_locs)
    output_array = numpy.stack([x_coords, y_coords, z_coords, val_coords], axis=-1)

    return output_array

def larcv_event_deps(input_array):

    # This is a larcv particle object array.
    # 

    # Split the pieces we need:
    vertex = input_array['_vtx']
    e_dep  = input_array['_energy_deposit']
    # print("e_dep.shape:" , e_dep.shape)
    # print("vertex['x'].shape:" , vertex['_x'].shape)
    return numpy.stack([vertex['_x'], vertex['_y'], vertex['_z'], e_dep], axis=-1)



def larcvsparse_to_dense_2d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]

    output_array = numpy.zeros((batch_size, dense_shape[0], dense_shape[1]), dtype=numpy.float32)

    x_coords = input_array[:,:,:,0]
    y_coords = input_array[:,:,:,1]
    val_coords = input_array[:,:,:,2]


    filled_locs = val_coords != -999
    non_zero_locs = val_coords != 0.0
    mask = numpy.logical_and(filled_locs,non_zero_locs)
    # Find the non_zero indexes of the input:
    batch_index, plane_index, voxel_index = numpy.where(filled_locs)


    values  = val_coords[batch_index, plane_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])


    # Tensorflow expects format as either [batch, height, width, channel]
    # or [batch, channel, height, width]
    # Fill in the output tensor
    output_array[batch_index, x_index, y_index] = values


    return output_array

def larcvsparse_to_dense_3d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    output_array = numpy.zeros((batch_size, *(dense_shape)) , dtype=numpy.float32)
    # This is the "real" size:
    # output_array = numpy.zeros((batch_size, 1, 45, 45, 275), dtype=numpy.float32)
    x_coords   = input_array[:,0,:,0]
    y_coords   = input_array[:,0,:,1]
    z_coords   = input_array[:,0,:,2]
    val_coords = input_array[:,0,:,3]
    # print(x_coords[0:100])
    # print(y_coords[0:100])
    # print(z_coords[0:100])

    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, voxel_index])
    z_index = numpy.int32(z_coords[batch_index, voxel_index])


    # Fill in the output tensor

    output_array[batch_index, x_index, y_index, z_index] = values

    return output_array

