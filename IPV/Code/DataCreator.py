import csv
import os

import cv2
import numba
from scipy.spatial import distance as dist
from skimage import img_as_ubyte
from skimage import io
from skimage.transform import resize

from IPV.Code.GPU_Utils import *


class PatchCreator:
    def __init__(self, image, sub_patch_scales):
        self.sub_patch_scales = sub_patch_scales
        self.patch_size = sub_patch_scales[0]
        self.image = image

    def create(self, x, y):
        sub_patches = []
        for scale in self.sub_patch_scales:
            sub_patch = createPatch(self.image, x, y, scale)
            sub_patches.append(sub_patch)

        arr_patches = []
        for i in range(4):
            arr_patches.append(resize(sub_patches[i], (self.patch_size, self.patch_size), preserve_range=True))

        return arr_patches


class DataCreator:
    def __init__(self, distance_intervals, angle_intervals, subpatch_scales):
        self.patch_count = None
        self.pc = None
        self.labels_count = None
        self.current_sample_name = None
        self.show_image = None
        self.name_count = None
        self.csv_files = None
        self.save_path = None
        self.sub_patch_scales = subpatch_scales
        self.patch_size = subpatch_scales[0] * 2
        self.distance_intervals = distance_intervals
        self.angle_intervals = angle_intervals
        self.points_dict = {}  # -------> [train] or [test]
        self.paths_dict = {}  # --------> [train] or [test]
        self.fold_list = []  # -----> [[[train],[test]], [[train],[test]], ...]
        self.pix_to_cm_vals_dict = {}

    # Create lists of names for each fold to self.fold_list
    def read_fold_lists(self, frame_name):
        train_list = []
        test_list = [frame_name]
        val_list = []

        self.fold_list.append([train_list, test_list, val_list])
        for trl in train_list:
            for tel in test_list:
                for vl in val_list:
                    if trl == tel or trl == vl or tel == vl:
                        print('Error in folds!')
                        exit()

    def create_csv(self, data_path):
        csvfile1 = open(data_path + '/test_fold.csv', 'w')
        data_file = csv.writer(csvfile1)
        return [data_file]

    # Read points and paths of a given names list to self.points and self.paths
    def read_points(self, data_path, frame_name, num_of_points):
        names_list = self.fold_list[0][1]

        self.points_dict = {}
        self.paths_dict = {}
        all_paths_dict = {}
        lines_dict = {}

        path = f'{frame_name}.png'
        name = f'{frame_name}'
        all_paths_dict[name] = path
        if num_of_points == 4:
            lines_dict[name] = f'{frame_name}.png (350, 350) (350, 350) (350, 350) (350, 350)'
        else:
            lines_dict[name] = f'{frame_name}.png (350, 350) (350, 350)'

        for list_name in names_list:
            self.paths_dict[list_name] = data_path + '/' + all_paths_dict[list_name]
            pt = []
            num = lines_dict[list_name]
            for i in range(num_of_points):
                num = num[num.find(' (') + 1:len(num)]
                p = (int(num[1:num.find(',')]), int(num[num.find(' ') + 1:num.find(')')]))
                pt.append(p)
            self.points_dict[list_name] = pt

    def create_x_y(self, x, y, num_of_points):
        labels = []
        x = int(x)
        y = int(y)
        for i, p in enumerate(self.points_dict[self.current_sample_name][0:num_of_points]):
            pix_distance = dist.euclidean(p, (x, y))
            dist_class = get_label(pix_distance, numba.typed.List(self.distance_intervals))
            ang_class = get_label(int(getAngle(p, (x, y))), numba.typed.List(self.angle_intervals))
            if dist_class is None:
                print('Error with dist_class, a None was returned, ensure all possible distances are accounted for.')
                exit()
            labels.append(dist_class)
            labels.append(ang_class)
            self.labels_count[i][dist_class] += 1

        self.patch_count += 1
        created = self.pc.create(x, y)
        for img_num, created_image in enumerate(created):
            image = img_as_ubyte(created_image)
            curr_save_path = self.save_path + '/' + str(self.name_count) + '_' + str(x) + '_' + str(y) + '_' + str(
                img_num + 1) + '.png'
            io.imsave(curr_save_path, image, check_contrast=False)
            if num_of_points == 2:
                self.csv_files[0].writerow([str(self.name_count) + '_' + str(x) + '_' + str(y),
                                            curr_save_path, self.current_sample_name, x, y,
                                            labels[0], labels[1], labels[2], labels[3]])
            else:
                self.csv_files[0].writerow([str(self.name_count) + '_' + str(x) + '_' + str(y),
                                            curr_save_path, self.current_sample_name, x, y,
                                            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6],
                                            labels[7]])

        cv2.circle(self.show_image, (x, y), 1, (255, 0, 0), 1)

    # Create and save patches with labels
    def create_data(self, frame_name, step, data_path, centre, radius, num_of_points):
        self.read_points(data_path, frame_name, num_of_points)

        self.save_path = f'{data_path}/test_patches'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(f'{data_path}/test_images/'):
            os.makedirs(f'{data_path}/test_images/')

        self.csv_files = self.create_csv(data_path)

        # Create patches with labels
        self.name_count = 0
        for name in self.points_dict:  # four points of a sample
            self.name_count += 1
            frame_path = self.paths_dict[name]
            image = io.imread(frame_path, as_gray=True)
            self.show_image = io.imread(frame_path)
            self.current_sample_name = name
            if num_of_points == 2:
                self.labels_count = [numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64)]
            else:
                self.labels_count = [numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64)]

            self.pc = PatchCreator(image, sub_patch_scales=self.sub_patch_scales)
            self.patch_count = 0
            # Limit region of interest.
            if radius != 0:
                x_mid = int(centre[0])
                y_mid = int(centre[1])
                roi_r = radius

                for x in range(x_mid - roi_r - 1, x_mid + roi_r + 1, step):
                    for y in range(y_mid - roi_r - 1, y_mid + roi_r + 1, step):
                        # Circle around (x_mid, y_mid).
                        if (x_mid - x) ** 2 + (y_mid - y) ** 2 <= roi_r ** 2:
                            self.create_x_y(x, y, num_of_points)
                cv2.circle(self.show_image, (x_mid, y_mid), 1, (255, 255, 255), 1)
            else:
                for x in range(0, image.shape[1], step):
                    for y in range(0, image.shape[0], step):
                        self.create_x_y(x, y, num_of_points)
            io.imsave(f'{data_path}/test_images/frame_with_patch_centres.png', self.show_image)

    def create(self, frame_name, step, data_path, centre, radius, num_of_points):
        if not self.fold_list:
            self.read_fold_lists(frame_name)
        self.create_data(frame_name, step, data_path, centre, radius, num_of_points)

        del self.csv_files[0]
