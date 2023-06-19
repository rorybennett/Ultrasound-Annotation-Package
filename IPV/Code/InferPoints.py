import csv
import os

import cv2
import numpy as np
import torch
from scipy import ndimage as nd
from scipy.spatial import distance as dist
from skimage import io

from IPV.Code import parameters as pms, My_Dataset, Quadruplet
from IPV.Code.My_Dataset import ToTensor


# Read points and paths of a given names list to self.points and self.paths
def read_points(data_path, frame_name, num_of_points):
    points_dict_i = {}
    paths_dict_i = {f'{frame_name}': data_path + f'/{frame_name}.png'}
    pt = []
    for j in range(num_of_points):
        p = (350, 350)
        pt.append(p)
    points_dict_i[f'{frame_name}'] = pt
    return points_dict_i, paths_dict_i


def infer(data_path, model_path, frame_name, num_of_points):
    results_path = data_path + '/TestResults'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    points_dict, paths_dict = read_points(data_path, frame_name, num_of_points)
    dataset = My_Dataset.My_Dataset(data_path + '/test_fold.csv', ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    net = Quadruplet.Quadruplet(num_of_points).cuda().half()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    csvfile = open(results_path + "/Results.csv", 'w')
    results_writer = csv.writer(csvfile)
    results_writer.writerow(
        ["Name", "P1_distance", "P2_distance", "P3_distance", "P4_distance", "Mean_distance", "h_difference",
         "w_difference"])
    csvfile2 = open(results_path + "/DetectedPoints.csv", 'w')
    detected_points_writer = csv.writer(csvfile2)
    csvfile3 = open(results_path + "/PatchBasedResults.csv", 'w')
    pbr_writer = csv.writer(csvfile3)
    prev_sample_name = ""
    sample_image = np.array([])
    # Loop through all images.
    for i, data in enumerate(data_loader, 0):
        image = data["image"].cuda().half()
        sample_name = data["sample_name"][0]
        coordinates = data["coordinates"][0].numpy()
        labels = data["labels"][0].numpy()
        map_results_path = "/MapWithAngleResults_f0_s1"

        if sample_name != prev_sample_name or i + 1 == len(data_loader.dataset):
            if np.any(sample_image):
                if not os.path.exists(results_path + map_results_path):
                    os.makedirs(results_path + map_results_path)
                distances = []
                detected_points = []
                for s in range(num_of_points):
                    convolved_map = nd.gaussian_filter(maps[s], 5)
                    max_i = np.argmax(convolved_map) + 1
                    max_y = (max_i // convolved_map.shape[1]) + 1
                    max_x = max_i % convolved_map.shape[1]
                    cv2.circle(sample_image, (max_x, max_y), 1, (0, 0, 0), 2)
                    cv2.circle(sample_image2, (max_x, max_y), 1, (255, 0, 0), 2)
                    cv2.circle(sample_image2,
                               (points_dict[prev_sample_name][s][0], points_dict[prev_sample_name][s][1]), 1,
                               (0, 255, 0), 2)
                    convolved_map = cv2.normalize(convolved_map, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
                    io.imsave(
                        results_path + map_results_path + "/" + prev_sample_name + "_convolved_map" + str(
                            s) + ".png", np.array(convolved_map, dtype=np.uint8), check_contrast=False)
                    distances.append(dist.euclidean((max_x, max_y), (
                        points_dict[prev_sample_name][s][0], points_dict[prev_sample_name][s][1])))
                    detected_points.append((max_x, max_y))

                io.imsave(results_path + map_results_path + "/" + prev_sample_name + ".png", sample_image)
                io.imsave(results_path + map_results_path + "/" + prev_sample_name + "_2.png",
                          sample_image2)
                if num_of_points == 4:
                    real_w = dist.euclidean(points_dict[prev_sample_name][1], points_dict[prev_sample_name][3])
                    real_h = dist.euclidean(points_dict[prev_sample_name][0], points_dict[prev_sample_name][2])
                    detected_w = dist.euclidean(detected_points[1], detected_points[3])
                    detected_h = dist.euclidean(detected_points[0], detected_points[2])
                    diff_w = abs(real_w - detected_w)
                    diff_h = abs(real_h - detected_h)
                    results_writer.writerow(
                        [prev_sample_name, distances[0], distances[1], distances[2], distances[3],
                         np.mean(distances), diff_h, diff_w])
                    detected_points_writer.writerow(
                        [prev_sample_name, detected_points[0][0], detected_points[0][1], detected_points[1][0],
                         detected_points[1][1], detected_points[2][0], detected_points[2][1],
                         detected_points[3][0], detected_points[3][1]])
                else:
                    real_h = dist.euclidean(points_dict[prev_sample_name][0], points_dict[prev_sample_name][1])
                    detected_h = dist.euclidean(detected_points[0], detected_points[1])
                    diff_h = abs(real_h - detected_h)
                    results_writer.writerow(
                        [prev_sample_name, distances[0], distances[1], np.mean(distances), diff_h])
                    detected_points_writer.writerow(
                        [prev_sample_name, detected_points[0][0], detected_points[0][1], detected_points[1][0],
                         detected_points[1][1]])

                for pvals in TP_TN_FP_FN:
                    for cvals in pvals:
                        pbr_writer.writerow([prev_sample_name, cvals])

            prev_sample_name = sample_name
            sample_image = io.imread(data_path + "/" + sample_name + ".png")
            sample_image2 = io.imread(data_path + "/" + sample_name + ".png")
            maps = [np.zeros(sample_image.shape[0:2], np.uint64),
                    np.zeros(sample_image.shape[0:2], np.uint64),
                    np.zeros(sample_image.shape[0:2], np.uint64),
                    np.zeros(sample_image.shape[0:2], np.uint64)]

            TP_TN_FP_FN = np.zeros((num_of_points, len(pms.tasks_classes[0]), len(pms.tasks_classes[0])))

        outs = net(image)
        ei = 0
        mno = -1
        for out, label in zip(outs, labels):
            ei += 1
            attended_classes = 7
            if ei % 2 == 0 and ei < 9:
                mno += 1
                res = torch.argsort(prev_out, descending=True).detach().cpu().numpy()[0]
                res_a = torch.argsort(out, descending=True).detach().cpu().numpy()[0]
                TP_TN_FP_FN[mno][prev_label][res[0]] += 1
                temp_map = np.zeros(sample_image.shape[0:2], np.uint8)
                if res[0] <= len(pms.tasks_classes[0]) - attended_classes and abs(res[0] - res[1]) == 1 and abs(
                        res[0] - res[2]) == 1 and abs(res_a[0] - res_a[1]) == 1 and abs(res_a[0] - res_a[2]) == 1:
                    angle = (res_a[0] + 4) % 8
                    start_angle = pms.tasks_classes[1][angle][0]
                    end_angle = pms.tasks_classes[1][angle][1]

                    for r_i in range(pms.layers_to_vote):
                        if pms.weighted_map:
                            vote_val = int(prev_out[0][res[r_i]])
                        else:
                            vote_val = 1
                        rad = (pms.tasks_classes[0][res[r_i]][0] + (
                                pms.tasks_classes[0][res[r_i]][1] - pms.tasks_classes[0][res[r_i]][0]) / 2)
                        th = (pms.tasks_classes[0][res[r_i]][1] - pms.tasks_classes[0][res[r_i]][0])
                        cv2.ellipse(temp_map, (coordinates[0], coordinates[1]), (int(rad), int(rad)), 0,
                                    start_angle, end_angle, vote_val, thickness=int(th))
                        cv2.ellipse(sample_image, (coordinates[0], coordinates[1]), (int(rad), int(rad)), 0,
                                    start_angle, end_angle, (255, 0, 0), thickness=1)

                    maps[mno] += temp_map

            prev_out = out
            prev_label = label
            if i == dataset.__len__() - 2:
                sample_name = "Finish"
    csvfile.close()
    csvfile2.close()
    csvfile3.close()
