import os
import cv2
# command line arguments
import argparse

# take the input and output directory
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='masks', help='input directory of masks. default: masks')
parser.add_argument('--output_dir', type=str, default='labels', help='output directory of labels. default: labels')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

# create the output dir if it does not exists
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))

        f.close()