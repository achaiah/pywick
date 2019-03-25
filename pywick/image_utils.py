import numpy as np

def draw_dice_on_image(label, prob, threshold=125, is_0_255=False):
    '''
    Draws a combined color map depicting how closely the guessed image mask (prob) corresponds to
    the ground truth (label).  Both label and prob must be black-and-white images of the same size.

    :param label: numpy array - the ground truth (b/w image)
    :param prob: numpy array - the prediction (b/w image)
    :param threshold: int - threshold {0-255} below which all values will be set to zero (default 125)
    :param is_0_255: bool - whether the labels and probs are in 0-255 range or 0-1 range (default: False)

    :return: numpy array depicting the overlap of prob image on top of label (ground truth)
    '''

    if not is_0_255:
        label = label * 255
        prob = prob * 255

    label = label > threshold
    prob  = prob > threshold

    H,W  = label.shape
    results = np.zeros((H*W,3),np.uint8)
    a = (2*label+prob).reshape(-1)
    miss = np.where(a==2)[0]
    hit  = np.where(a==3)[0]
    fp   = np.where(a==1)[0]

    results[miss] = np.array([255,255,255])
    results[hit]  = np.array([19,138,249])
    results[fp]   = np.array([246,249,16])
    results = results.reshape(H,W,3)

    return results


def draw_mask_on_image(image, mask, bg_color=(19, 138, 249), mask_color=[255, 255, 0], threshold=125, foreground_alpha=[1.0, 1.0, 0.5], is_0_255=False):
    '''

    Draws a mask on top of the original image. This is pretty CPU intensive so may want to revise for production environment
    image and mask must be the same size!

    :param image: numpy array of the image [Width, Height, RGB]
    :param mask: numpy array representing b/w image mask (black pixels - mask, white pixels - background)
    :param bg_color: numpy tuple (R,G,B) desired color to fill background with (default: magenta)
    :param mask_color: numpy array [R,G,B] desired color to colorize the extracted pixels with (default: yellow)
    :param threshold: threshold value used to determine which pixels in the mask to keep (default: >= 125)
    :param foreground_alpha: the proportions of blending between the pixels from the image and from mask_color (default: [1.0, 1.0, 0.5] of image mixed with 1-foreground_alpha of mask_color)
    :param is_0_255: whether the image and mask are in 0-1 floating range or 0-255 integer range (default: False)

    :return: numpy array containing composite image [RGB]

    '''
    if not is_0_255:
        image = image * 255
        mask = mask * 255

    mask = mask > threshold     # make sure all values below threshold are zero!

    H, W, _ = image.shape

    assert (H,W) == mask.shape, "image size does not equal mask size!"

    results = np.zeros((H, W, 3), np.uint8)  # create new image and fill with zeros
    results[...] = bg_color     # fill entire image with bg_color at first

    for x in range(W):          # iterate over every pixel and calculate new values
        for y in range(H):
            if mask[x][y] > 0:
                results[x][y][0] = (image[x][y][0] * foreground_alpha[0]) + (mask_color[0] * (1.0 - foreground_alpha[0]))
                results[x][y][1] = (image[x][y][1] * foreground_alpha[1]) + (mask_color[1] * (1.0 - foreground_alpha[1]))
                results[x][y][2] = (image[x][y][2] * foreground_alpha[2]) + (mask_color[2] * (1.0 - foreground_alpha[2]))

    return results