import matplotlib.pyplot as plt
import random
import math
import PIL
from PIL import Image
import cv2
import numpy as np
from PIL import ImageFilter

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.ndimage
import enum

def gaussian_blur(sharp_image, sigma):
    # Filter channels individually to avoid gray scale images
    blurred_image_r = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 0], sigma=sigma)
    blurred_image_g = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 1], sigma=sigma)
    blurred_image_b = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 2], sigma=sigma)
    blurred_image = np.dstack((blurred_image_r, blurred_image_g, blurred_image_b))
    return blurred_image


def uniform_blur(sharp_image, uniform_filter_size):
    # The multidimensional filter is required to avoid gray scale images
    multidim_filter_size = (uniform_filter_size, uniform_filter_size, 1)
    blurred_image = scipy.ndimage.filters.uniform_filter(sharp_image, size=multidim_filter_size)
    return blurred_image


def blur_image_locally(sharp_image, mask, use_gaussian_blur, gaussian_sigma, uniform_filter_size):
    one_values_f32 = np.full(sharp_image.shape, fill_value=1.0, dtype=np.float32)
    sharp_image_f32 = sharp_image.astype(dtype=np.float32)
    sharp_mask_f32 = mask.astype(dtype=np.float32)

    # if use_gaussian_blur:
    #     blurred_image_f32 = gaussian_blur(sharp_image_f32, sigma=gaussian_sigma)
    #     blurred_mask_f32 = gaussian_blur(sharp_mask_f32, sigma=gaussian_sigma)
    #
    # else:
    #     blurred_image_f32 = uniform_blur(sharp_image_f32, uniform_filter_size)
    #     blurred_mask_f32 = uniform_blur(sharp_mask_f32, uniform_filter_size)

    blurred_mask_inverted_f32 = one_values_f32 - blurred_mask_f32
    weighted_sharp_image = np.multiply(sharp_image_f32, blurred_mask_f32)
    weighted_blurred_image = np.multiply(blurred_image_f32, blurred_mask_inverted_f32)
    locally_blurred_image_f32 = weighted_sharp_image + weighted_blurred_image

    locally_blurred_image = locally_blurred_image_f32.astype(dtype=np.uint8)

    return locally_blurred_image

def test_artifical_depth_estimation_broken():
    parent = "CAMtest-depth60100-1_16-12_53-154T84"
    img_orig = Image.open("/".join([parent,"imgmain-epi001-step0000.jpg"]))
    sharp_image = np.array(img_orig, dtype=np.uint8)
    sharp_mask = np.array(Image.open("C:/Users/Meriel/Documents/GitHub/DPT/output_monodepth/imgmain-epi001-step0000.png"), dtype=np.uint8)
    sharp_mask = sharp_mask / sharp_mask.max()
    img_blur = gaussian_blur(np.array(sharp_image), 31)
    result = np.array((1 - sharp_mask[:, :, None]) * sharp_image + sharp_mask[:, :, None] * img_blur, dtype=np.uint8)
    plt.imshow(result)
    plt.show()
    im = Image.fromarray(result)
    im.save("artifical-depth-estimation.jpeg")

# test image manipulations
def test_depth_of_field_transformation_broken():
    # parent = "CAMtest-DDPG-depth50-1_14-13_29-K8PHIZ"
    parent = "CAMtest-depth60100-1_16-12_53-154T84"
    img_orig = Image.open("/".join([parent,"imgmain-epi001-step0000.jpg"]))
    img_d50 = Image.open("/".join([parent, "imgtest-epi001-step0000.jpg"]))
    img_blur = img_orig.filter(ImageFilter.BoxBlur(4))
    img_orig = np.array(img_orig, dtype=np.uint8)
    img_d50 = np.array(img_d50, dtype=np.uint8)

    # mask matching pixels
    same = np.array(img_orig == img_d50, dtype=np.uint8)
    # patch in blur
    diff = (1 - same) * img_blur
    img_blur = np.array(img_blur, dtype=np.uint8)
    out = diff * img_blur + same * img_orig
    im = Image.fromarray(out)
    im.save("test_depth_of_field_transformation-out.jpeg")

def test_depth_of_field_transformation():
    parent = "CAMtest-depth50100-1_16-12_33-8L3IHC"
    img_orig_pil = Image.open("/".join([parent,"imgmain-epi001-step0000.jpg"]))
    # img_d50 = Image.open("/".join([parent, "imgtest-epi001-step0000.jpg"]))
    # img_d100 = Image.open("/".join([parent, "imgmiddist-epi001-step0000.jpg"]))
    img_orig_ann = Image.open("/".join([parent,"imgmain-epi001-step0000-annot.jpg"]))
    img_d50_ann = Image.open("/".join([parent, "imgtest-epi001-step0000-annot.jpg"]))
    img_d100_ann = Image.open("/".join([parent, "imgmiddist-epi001-step0000-annot.jpg"]))
    img_orig = np.array(img_orig_pil, dtype=np.uint8)
    img_orig_ann = np.array(img_orig_ann, dtype=np.uint8)

    # mask matching pixels for dist=(50,100)
    img_blur_far = img_orig_pil.filter(ImageFilter.BoxBlur(3))
    img_blur_far.save("test_depth_of_field_transformation-blur3.jpeg")
    img_blur_far = np.array(img_blur_far, dtype=np.uint8)
    img_d100_ann = np.array(img_d100_ann, dtype=np.uint8)
    same = img_orig_ann == img_d100_ann
    idx = (same == 0)
    img_orig[idx] = img_blur_far[idx]
    im = Image.fromarray(img_orig)
    im.save("test_depth_of_field_transformation-far.jpeg")

    # mask matching pixels for dist=(1,50)
    img_blur = img_orig_pil.filter(ImageFilter.BoxBlur(1))
    img_blur.save("test_depth_of_field_transformation-blur1.jpeg")
    img_d50_ann = np.array(img_d50_ann, dtype=np.uint8)
    img_blur = np.array(img_blur, dtype=np.uint8)
    same = img_orig_ann == img_d50_ann
    idx = (same == 0)
    img_orig[idx] = img_blur[idx]
    im = Image.fromarray(img_orig)
    im.save("test_depth_of_field_transformation-nearfar.jpeg")

def test_resolution_increase():
    filepath = ""
    img_orig = Image.open(filepath)

def distort_point(undistorted):
    undistorted = (undistorted.copy() / 255.0).astype(np.float64)

    K = np.array([[689.21, 0., 1295.56],
                  [0., 690.48, 942.17],
                  [0., 0., 1.]])
    K = np.array([[1, 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    # zero distortion coefficients work well for this image
    D = np.array([0., 0., 0., 0.])

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    undistorted[:, 0] = (undistorted[:, 0] - cx) / fx
    undistorted[:, 1] = (undistorted[:, 1] - cy) / fy
    plt.imshow(undistorted)
    plt.show()
    cv2.imwrite("test_fisheye-undistorted.jpg", undistorted[:, :, ::-1])

    distorted = cv2.fisheye.distortPoints(undistorted.reshape(1, -1, 2), K, D).reshape(135, 240, 3) #.reshape(-1, 3)
    plt.imshow(distorted)
    plt.show()
    cv2.imwrite("test_fisheye-distorted.jpg", distorted[:, :, ::-1])
    return distorted

class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'

# https://github.com/Synthesis-AI-Dev/fisheye-distortion
def distort_to_fisheye(img):
    # def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
    #                   mode: DistortMode = DistortMode.LINEAR, crop_output: bool = True,
    #                   crop_type: str = "corner") -> np.ndarray:
        """Apply fisheye distortion to an image
        Args:
            img (numpy.ndarray): BGR image. Shape: (H, W, 3)
            cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                                Shape: (3, 3)
            dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                                Shape: (1, 4)
            mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                                RGB images = linear, Mask/Surface Normals/Depth = nearest
            crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                                image will be mapped to 4 corners of the distorted image for cropping.
            crop_type (str): How to crop.
                "corner": We crop to the corner points of the original image, maintaining FOV at the top edge of image.
                "middle": We take the widest points along the middle of the image (height and width). There will be black
                          pixels on the corners. To counter this, original image has to be higher FOV than the desired output.
        Returns:
            numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
        """
        # fx =
        # cx =
        # cy =

        cam_intr = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        dist_coeff = np.array([])
        assert cam_intr.shape == (3, 3)
        assert dist_coeff.shape == (4,)

        imshape = img.shape
        if len(imshape) == 3:
            h, w, chan = imshape
        elif len(imshape) == 2:
            h, w = imshape
            chan = 1
        else:
            raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

        imdtype = img.dtype

        # Get array of pixel co-ords
        xs = np.arange(w)
        ys = np.arange(h)
        xv, yv = np.meshgrid(xs, ys)
        img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
        img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

        # Get the mapping from distorted pixels to undistorted pixels
        undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
        undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
        undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
        undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
        undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
        undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

        # Map RGB values from input img using distorted pixel co-ordinates
        if chan == 1:
            img = np.expand_dims(img, 2)
        interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode.value,
                                                                   bounds_error=False, fill_value=0)
                         for channel in range(chan)]
        img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

        if imdtype == np.uint8:
            # RGB Image
            img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
        elif imdtype == np.uint16:
            # Mask
            img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
        elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
            img_dist = img_dist.astype(imdtype)
        else:
            raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

        if crop_output:
            # Crop rectangle from resulting distorted image
            # Get mapping from undistorted to distorted
            distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
            cam_intr_inv = np.linalg.inv(cam_intr)
            distorted_px = np.tensordot(distorted_px, cam_intr_inv,
                                        axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
            distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
            distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
            distorted_px = distorted_px.reshape((h, w, 2))
            if crop_type == "corner":
                # Get the corners of original image. Round values up/down accordingly to avoid invalid pixel selection.
                top_left = np.ceil(distorted_px[0, 0, :]).astype(np.int)
                bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(np.int)
                img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            elif crop_type == "middle":
                # Get the widest point of original image, then get the corners from that.
                width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
                width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
                height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
                height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
                img_dist = img_dist[height_min:height_max, width_min:width_max]
            else:
                raise ValueError

        if chan == 1:
            img_dist = img_dist[:, :, 0]

        return img_dist

def fish_eye(image):
    """ Fish eye algorithm """
    w, h, c = image.shape
    w2 = w / 2
    h2 = h / 2
    image_copy = image.copy()
    for y in range(h):
        # Normalize every pixels along y axis
        # when y = 0 --> ny = -1
        # when y = h --> ny = +1
        ny = ((2 * y) / h) - 1
        # ny * ny pre calculated
        ny2 = ny ** 2
        for x in range(w):
            # Normalize every pixels along x axis
            # when x = 0 --> nx = -1
            # when x = w --> nx = +1
            nx = ((2 * x) / w) - 1
            # pre calculated nx * nx
            nx2 = nx ** 2

            # calculate distance from center (0, 0)
            r = math.sqrt(nx2 + ny2)

            # discard pixel if r below 0.0 or above 1.0
            if 0.0 <= r <= 1.0:

                nr = (r + 1 - math.sqrt(1 - r ** 2)) / 2
                if nr <= 1.0:

                    theta = math.atan2(ny, nx)
                    nxn = nr * math.cos(theta)
                    nyn = nr * math.sin(theta)
                    x2 = int(nxn * w2 + w2)
                    y2 = int(nyn * h2 + h2)

                    if 0 <= int(y2 * w + x2) < w * h:

                        pixel = image[x2, y2]
                        image_copy[x, y] = pixel

    return image_copy

def fisheye_wand(image, filename):
    from wand.image import Image
    # with Image(filename=filename) as img:
    with Image.from_array(image) as img:
        img.virtual_pixel = 'transparent'
        img.distort('barrel', (0.1, 0.0, -0.05, 1.0))
        # img.distort('barrel_inverse', (0.0, 0.0, -0.5, 1.5))
        img.save(filename="test-fisheye-wand-barrel1.jpg")
        return np.array(img)

def test_pinhole_to_fisheye():
    parent = "CAMtest-depth60100-1_16-12_53-154T84"
    sharp_image = np.array(Image.open("/".join([parent, "imgmain-epi001-step0000.jpg"])))
    # result = fish_eye(sharp_image)
    # result = distort_to_fisheye(sharp_image)
    result = fisheye_wand(sharp_image, "/".join([parent, "imgmain-epi001-step0000.jpg"]))
    cv2.imwrite("test_fisheye.jpg", result[:, :, ::-1])

# https://stackoverflow.com/questions/54826757/what-is-the-most-elegant-way-to-blur-parts-of-an-image-using-python
def test_artifical_depth_estimation():
    parent = "CAMtest-depth60100-1_16-12_53-154T84"
    sharp_image = np.array(Image.open("/".join([parent, "imgmain-epi001-step0000.jpg"])))
    sharp_mask = cv2.imread("C:/Users/Meriel/Documents/GitHub/DPT/output_monodepth/imgmain-epi001-step0000.png")
    sharp_mask = np.array(sharp_mask)
    # info = np.iinfo(sharp_mask.dtype)
    # sharp_mask = np.clip(sharp_mask / info.max, 0, 255)

    # im = Image.fromarray(sharp_mask).convert('RGB')
    # im.save("test-depthmask.jpg")
    # sharp_mask_f32 = np.repeat(np.transpose(sharp_mask[None], (1,2,0)), 3, axis=2)

    one_values_f32 = np.full(sharp_image.shape, fill_value=1.0, dtype=np.float64)
    sharp_image_f32 = sharp_image.astype(dtype=np.float64)
    sharp_mask_f32 = sharp_mask.astype(dtype=np.float64) / 255.0

    gaussian_sigma = 1.5
    blurred_image_f32 = gaussian_blur(sharp_image_f32, sigma=gaussian_sigma) # HWC 0-255
    # blurred_mask_f32 = gaussian_blur(sharp_mask_f32, sigma=gaussian_sigma) # HWC 0-1

    blurred_mask_inverted_f32 = one_values_f32 - sharp_mask_f32
    # im = Image.fromarray(blurred_mask_inverted_f32).convert('RGB')
    # im.save("test_blurred_mask_inverted_f32.jpg")
    weighted_sharp_image = sharp_image_f32 * sharp_mask_f32 # np.multiply(sharp_image_f32, sharp_mask_f32)
    weighted_blurred_image = blurred_image_f32 * blurred_mask_inverted_f32 # np.multiply(blurred_image_f32, blurred_mask_inverted_f32)
    locally_blurred_image_f32 = weighted_sharp_image + weighted_blurred_image

    # locally_blurred_image = locally_blurred_image_f32.astype(dtype=np.uint8)
    result = locally_blurred_image_f32.astype(dtype=np.uint8)

    plt.imshow(result)
    plt.show()
    cv2.imwrite("test_result2.jpg", result[:,:,::-1])

if __name__ == '__main__':
    # test_artifical_depth_estimation()
    # test_depth_of_field_transformation()
    # # test_resolution_increase()
    test_pinhole_to_fisheye()