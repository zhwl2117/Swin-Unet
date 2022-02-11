#  this dataloader and datagenerator is mainly from the nnUNet
#  For more details, please refer to https://github.com/MIC-DKFZ/nnUNet


from collections import OrderedDict
import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.utilities.file_and_folder_operations import *


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0, 
                memmap_mode='r', pseudo_3d_slices=1, pad_mode='edge',
                pad_kwargs_data=None, pad_sides=None, number_of_threads_in_multithreaded=None):
        super().__init__(data, batch_size, number_of_threads_in_multithreaded)

        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint

            need_to_pad = self.need_to_pad
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}
