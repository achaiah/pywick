import numpy as np
from tqdm import tqdm

def get_dataset_mean_std(dataset, img_size=256, output_div=255.0):
    '''
    Computes channel-wise mean and std of the dataset. The process is memory-intensive as the entire dataset must fit into memory.
    Therefore, each image is scaled down to img_size first (default: 256).

    Assumptions: 1. dataset uses PIL to read images    2. Images are in RGB format.

    :param dataset: pytorch Dataset
    :param img_size: scale of images at which to compute mean/std (default: 256)
    :param output_div: float {1.0, 255.0} - Image values are naturally in 0-255 value range so the returned output is divided by output_div.
        For example, if output_div = 255.0 then mean/std will be in 0-1 range.
    :return: (mean, std) as per-channel values ([r,g,b], [r,g,b])
    '''

    total = np.zeros((3, (len(dataset)*img_size*img_size)), dtype=int)
    position = 0        # keep track of position in the total array

    for src, _ in tqdm(dataset, ascii=True, desc="Process", unit='images'):
        src = src.resize((img_size, img_size))      # resize to same size
        src = np.array(src)

        # reshape into correct shape
        src = src.reshape(img_size * img_size, 3)
        src = src.swapaxes(1,0)

        # np.concatenate((a, b, c), axis=1)  # NOPE NOPE NOPE -- makes a memory re-allocation for every concatenate operation

        # -- In-place value substitution -- #
        place = img_size * img_size * position
        total[0:src.shape[0], place:place+src.shape[1]] = src   # copies the src data into the total position at specified index

        position = position+1

    return total.mean(1) / output_div, total.std(1) / output_div        # return channel-wise mean for the entire dataset


if __name__ == "__main__":
    from datasets.FolderDataset import FolderDataset
    from datasets.data_utils import pil_loader_rgb

    dataset = FolderDataset(root='/Users/alexei_samoylov/Desktop/PartPictures/grn_truth_masks_v1-v4_gimp_11-21-18/images', class_mode='label', default_loader=pil_loader_rgb)
    mean, std = get_dataset_mean_std(dataset)
    print('----- RESULT -----')
    print('mean: {}'.format(mean))
    print('std:  {}'.format(std))
    print ('----- DONE ------')
