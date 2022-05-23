class_list = ['aircraft_carrier', 'alarm_clock', 'ant', 'anvil', 'asparagus', 'axe', 'banana', 'basket', 'bathtub', 'bear', 'bee', 'bird', 'blackberry', 'blueberry', 'bottlecap', 'broccoli', 'bus', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon', 'canoe', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'coffee_cup', 'compass', 'computer', 'cow', 'crab', 'crocodile', 'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck', 'dumbbell', 'elephant', 'eyeglasses', 'feather', 'fence', 'fish', 'flamingo', 'flower', 'foot', 'fork', 'frog', 'giraffe', 'goatee', 'grapes', 'guitar', 'hammer', 'helicopter', 'helmet', 'horse', 'kangaroo', 'lantern', 'laptop', 'leaf', 'lion', 'lipstick', 'lobster', 'microphone', 'monkey', 'mosquito', 'mouse', 'mug', 'mushroom', 'onion', 'panda', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'pig', 'pillow', 'pineapple', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone', 'screwdriver', 'sea_turtle', 'see_saw', 'sheep', 'shoe', 'skateboard', 'snake', 'speedboat', 'spider', 'squirrel', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'swan', 'table', 'teapot', 'teddy-bear', 'television', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'tiger', 'toe', 'train', 'truck', 'umbrella', 'vase', 'watermelon', 'whale', 'zebra']

import numpy as np
def make_dataset_fromlist(image_list):
    # print("image_list", image_list)
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


for target in ['sketch','painting','clipart','real']:
    root = 'data/multi/'
    image_list_file_path = root + 'unique_image_paths_{}.txt'.format(target)
    image_paths, labels = make_dataset_fromlist(image_list_file_path)

    dd = {}
    for image in image_paths:
        label = image.split('/')[1]
        assert label in class_list
        ind = class_list.index(label)
        if not ind in dd:
            dd[ind] = [image]
        else:
            dd[ind].append(image)

    labeled_images_filename = root + 'labeled_target_images_{}_3.txt'.format(target)
    unlabeled_images_filename = root + 'unlabeled_target_images_{}_3.txt'.format(target)

    from random import shuffle

    labeled_images_file = open(labeled_images_filename, 'w')
    unlabeled_images_file = open(unlabeled_images_filename, 'w')

    for ind in range(len(class_list)):
        shuffle(dd[ind])
        # three labeled
        labeled_images = dd[ind][:3]
        unlabeled_images = dd[ind][3:]
        for image in labeled_images:
            labeled_images_file.write(image + ' ' + str(ind) + '\n')
        for image in unlabeled_images:
            unlabeled_images_file.write(image + ' ' + str(ind) + '\n')

    labeled_images_file.close()
    unlabeled_images_file.close()