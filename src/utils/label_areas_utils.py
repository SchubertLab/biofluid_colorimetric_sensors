import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table


def get_area_labels(mask_input, n_sensing_areas=16, n_connectivity=2):

    # Label each spot with region props
    labeled_array = label(mask_input, connectivity=n_connectivity)
    df_circles = pd.DataFrame(
        regionprops_table(
            labeled_array,
            properties=['label', 'centroid', 'area', 'major_axis_length', 'minor_axis_length']
        )
    )

    df_circles['radius'] = df_circles['minor_axis_length'] / 2

    # filter areas by size
    sorted_area_df = df_circles.sort_values(by=['area'], ascending=False).head(n_sensing_areas)
    list_indices = sorted_area_df['label']
    mask = (~df_circles['label'].isin(list_indices))
    df_circles.drop(df_circles[mask].index, axis=0, inplace=True)

    if df_circles.shape[0] == n_sensing_areas:
        # find order of the labels
        circles = np.array(df_circles[['centroid-1', 'centroid-0', 'radius', 'area', 'label']])
        circles_ordered, circle_names = find_labels_order(
            circles=circles,
            n_sensing_areas=n_sensing_areas
        )

        labels_ordered = [int(x[4]) for x in circles_ordered]
        s_order_for_df = []
        s_names_for_df = []
        labels_ordered_sorted = sorted(labels_ordered)
        for i in range(n_sensing_areas):
            label_index_temp = labels_ordered_sorted[i]
            index_temp = labels_ordered.index(label_index_temp)
            s_order_for_df.append(index_temp)
            s_names_for_df.append(circle_names[index_temp])

        df_circles['sensing_area_order'] = s_order_for_df
        df_circles['sensing_area_name'] = s_names_for_df
        df_circles.sort_values(by=['sensing_area_order'])
    else:
        labeled_array = None
        df_circles = None

    return labeled_array, df_circles


def find_labels_order(circles, n_sensing_areas=16):
    if circles.shape[0] > n_sensing_areas:
        # Select circles closer to the center if more than n_sensors
        center_sensor_all = np.mean(circles, axis=0)
        center_image = [center_sensor_all[1], center_sensor_all[0]]
        order_array = [
            np.sqrt((a[0] - center_image[0]) ** 2 + (a[1] - center_image[1]) ** 2)
            for a in circles
        ]
        circle_indices = np.argsort(order_array)
        circle_indices = circle_indices[0:n_sensing_areas]
        new_circles_unsorted = [circles[i] for i in circle_indices]
    else:
        new_circles_unsorted = circles

    # Order Circles by distance to global centroid
    center_sensor = np.mean(new_circles_unsorted, axis=0)
    distance_to_center_sensor = [
        np.sqrt((a[0] - center_sensor[0]) ** 2 + (a[1] - center_sensor[1]) ** 2)
        for a in new_circles_unsorted
    ]
    new_indices = np.argsort(distance_to_center_sensor)
    new_circles = [new_circles_unsorted[i] for i in new_indices]

    # Order Circles by angle
    circle_angles_deg = [
        np.arctan2(a[1] - center_sensor[1], a[0] - center_sensor[0])
        for a in new_circles
    ]

    # glucose
    glucose_angles = circle_angles_deg[0:3]
    glucose = new_circles[0:3]
    indices_angle = np.argsort(glucose_angles)
    new_glucose = [glucose[i] for i in indices_angle]

    # reference and ph
    ref_ph_angles = circle_angles_deg[3:7]
    ref_ph = new_circles[3:7]
    indices_angle = np.argsort(ref_ph_angles)
    new_ref_ph = [ref_ph[i] for i in indices_angle]
    new_ph = [new_ref_ph[0], new_ref_ph[2], new_ref_ph[3]]
    new_sodium = [new_ref_ph[1]]

    # temperature
    temp_angles = circle_angles_deg[7:]
    temperature = new_circles[7:]
    indices_angle = np.argsort(temp_angles)
    temp_t = [temperature[i] for i in indices_angle]
    new_temperature = [
        temp_t[1], temp_t[2], temp_t[3], temp_t[4],
        temp_t[5], temp_t[6], temp_t[0]
    ]

    array_all = [new_temperature, new_ph, new_glucose, new_sodium]

    final_circles = []
    for x in array_all:
        for c in x:
            final_circles.append(c)

    circle_names = [
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'ph_ul', 'ph_ur', 'ph_d',
        'g_ul', 'g_ur', 'g_d',
        'Na',
    ]
    return final_circles, circle_names

