import json
import os
import os.path
import argparse

from .datasets.FolderDataset import FolderDataset, rgb_image_loader

opt = dict()
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=False, type=str, help='Path to root directory of the images')
parser.add_argument('--output_path', required=False, type=str, help='Path to save computed statistics to. If not provided, will save inside root_path')

opt = vars(parser.parse_args())

# clean up the dictionary so it doesn't contain 'None' values
removals = list()
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


def get_dataset_mean_std(dataset_name='imagenet'):
    return dataset_mean_std[dataset_name]


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
    mean, std = get_dataset_mean_std(dataset, img_size=256)
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