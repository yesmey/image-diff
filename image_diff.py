from multiprocessing import Pool, freeze_support
from skimage import metrics
import cv2
import argparse
import os

def save_image(image, filename):
    filename, file_extension = os.path.splitext(filename)
    output_file = os.path.join(output_folder, os.path.basename(filename) + file_extension)
    cv2.imwrite(output_file, image)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculate_difference(source_image, target_image):
    source_grayscale = convert_to_grayscale(source_image)
    target_grayscale = convert_to_grayscale(target_image)
    (_, diff) = metrics.structural_similarity(source_grayscale, target_grayscale, full=True)
    return find_contours((diff * 255).astype("uint8"))

def grab_contours(cnts):
    # utility method returning correct index depending on OpenCV version
    if len(cnts) == 2:
        return cnts[0]
    elif len(cnts) == 3:
        return cnts[1]
    return cnts

def find_contours(diff):
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return grab_contours(cnts)

def calculate_file_difference(source, target):
    source_image = cv2.imread(source)
    target_image = cv2.imread(target)

    contours = calculate_difference(source_image, target_image)

    if contours:
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(target_image, (max(x - 5, 0), max(y - 5, 0)), (x + w, y + h), (0, 0, 255), 2)
        save_image(target_image, target)

def is_valid_extension(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in ['.bmp', '.jpeg', '.jpg', '.png', '.webp', '.pic', '.hdr']

def get_all_files(path):
    if not os.path.exists(path):
        return []

    result = []
    for f in os.listdir(path):
        if not f.startswith('.') and is_valid_extension(f):
            result.append(os.path.join(path, f))
    return result

def initializer(folder):
    global output_folder
    output_folder = folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", default="source", help="source folder")
    ap.add_argument("-t", "--target", default="target", help="target folder")
    ap.add_argument("-o", "--output", default="output", help="output folder")
    args = vars(ap.parse_args())

    source_files = get_all_files(args["source"])
    target_files = get_all_files(args["target"])
    output = args["output"]

    source_files.sort()
    target_files.sort()

    dataset = list(zip(source_files, target_files))

    with Pool(processes=2,initializer=initializer, initargs=(output,)) as pool:
        pool.starmap(calculate_file_difference, dataset)

if __name__=="__main__":
    freeze_support()
    main()
