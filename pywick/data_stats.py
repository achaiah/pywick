import json
import os
import os.path
import argparse
import numpy as np
from tqdm import tqdm

from pywick.datasets.FolderDataset import FolderDataset, rgb_image_loader

opt = {}
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=False, type=str, help='Path to root directory of the images')
parser.add_argument('--output_path', required=False, type=str, help='Path to save computed statistics to. If not provided, will save inside root_path')

opt = vars(parser.parse_args())

# clean up the dictionary so it doesn't contain 'None' values
removals = []
for key, val in opt.items():
    if val is None:
        removals.append(key)
for rem in removals:
    # print('removing: ', rem)
    opt.pop(rem)

dataset_mean_std = {
    'imagenet': ([0.485, 0.456, 0.406]),
    'general': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
}


def get_dataset_mean_std(dataset=None, img_size=None, output_div=255.0, dataset_name=None):
    """
        Computes channel-wise mean and std of the dataset. The process is memory-intensive as the entire dataset must fit into memory.
        Therefore, each image is scaled down to img_size first (default: 256).
        Assumptions: 1. dataset uses PIL to read images    2. Images are in RGB format.
        :param dataset: pytorch Dataset
        :param img_size: scale of images at which to compute mean/std (default: 256)
        :param output_div: float {1.0, 255.0} - Image values are naturally in 0-255 value range so the returned output is divided by output_div.
        For example, if output_div = 255.0 then mean/std will be in 0-1 range.
        :param dataset_name: name of a well-known dataset to return (one of {'imagenet', 'general'})

        :return: (mean, std) as per-channel values ([r,g,b], [r,g,b])
    """
    if dataset_name in dataset_mean_std.keys():
        return dataset_mean_std[dataset_name]
    else:
        total = np.zeros((3, (len(dataset) * img_size * img_size)), dtype=int)
        position = 0  # keep track of position in the total array

        for src, _ in tqdm(dataset, ascii=True, desc="Process", unit='images'):
            src = src.resize((img_size, img_size))  # resize to same size
            src = np.array(src)

            # reshape into correct shape
            src = src.reshape(img_size * img_size, 3)
            src = src.swapaxes(1, 0)

            # np.concatenate((a, b, c), axis=1)  # NOPE NOPE NOPE -- makes a memory re-allocation for every concatenate operation

            # -- In-place value substitution -- #
            place = img_size * img_size * position
            total[0:src.shape[0], place:place + src.shape[1]] = src  # copies the src data into the total position at specified index

            position = position + 1

        return total.mean(1) / output_div, total.std(1) / output_div  # return channel-wise mean for the entire dataset


def create_dataset_stats(data_path, output_path=None, verbose=False):
    '''
    Generates statistics for the given dataset and writes them to a JSON file. Expects the data to be in the following dir structure:
    dataroot
     | - Class Dir
         | - image 1
         | - image 2
         | - image N

    :param data_path: string - path to dataroot
    :param output_path: - path/filename to write the stats to (default: None - will output stats.json file in the dataroot)

    :return: None
    '''

    stats = {}
    if output_path is None:
        output_path = os.path.join(data_path, 'stats.json')

    dataset = FolderDataset(root=data_path, class_mode='label', default_loader=rgb_image_loader)

    stats['num_items'] = len(dataset)
    mean, std = get_dataset_mean_std(dataset=dataset, img_size=256)
    stats['mean'], stats['std'] = mean.tolist(), std.tolist()       # convert from numpy array to python

    if verbose:
        print('------- Dataset Stats --------')
        print(stats)
        print('Written to: ', output_path)
        print('------ End Dataset Stats ------')

    with open(output_path, 'a') as statsfile:
        json.dump(stats, statsfile)

    return stats

if __name__ == "__main__":
    '''
        Sample command: python3 data_stats.py --root_path /Users/Shared/test/images
    '''
    import sys
    sys.path.append("..")
    sys.path.append("../..")

    # path = opt.get('root_path','/Users/Shared/test/images')
    path = opt.get('root_path','/Users/Shared/test/deleteme')
    stats = create_dataset_stats(data_path=path, output_path=opt.get('output_path', None))
    # print('----- RESULT -----')
    # print(stats)
    # print('------------------')