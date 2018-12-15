# Mosaic Silhouettes - Download some images from Google Images
# Leverages this package: https://github.com/hardikvasa/google-images-download
# Geoff Sims <geoffrey.sims@gmail.com>

from google_images_download import google_images_download
import os
from PIL import Image

# Fixed directory to download images into
img_dir = os.path.join(os.getcwd(),'img')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

def download_example_frog_images(n=100):
    # Download a whole host of various coloured frog images
    frog_cols = ['green', 'red', 'blue', 'yellow', 'brown', 'pink', 'poison', 'black', 'purple']
    keyword_string = "{} frog"
    arguments = {
            "limit":n,
            "output_directory":img_dir,
            "no_directory":True
    }

    for c in frog_cols:
        print(c)
        arguments['keywords'] = keyword_string.format(c)
        response = google_images_download.googleimagesdownload()
        paths = response.download(arguments)
    
    # Return
    return True

def download_single_keyword(keyword, n=100):
    arguments = {
            "keywords": keyword,
            "limit":n,
            "output_directory":img_dir,
            "no_directory":True
    }
    response = google_images_download.googleimagesdownload()
    paths = response.download(arguments)
    
    # Return
    return True

def basic_clean():
    # Do some cleaning to remove any bad images
    print("Removing bad images...")
    file_list = os.listdir(img_dir)
    file_list.sort()
    for img_file in file_list:
        img_filepath = os.path.join(img_dir, img_file)
        try:
            im = Image.open(img_filepath)
        except:
            print("Error with {}: attempting deletion...".format(img_file))
            try:
                os.remove(img_filepath)
            except:
                pass

# Main function
if __name__ == '__main__':
    # Run example script
    download_example_frog_images()
    
    # Run arbitrary keyword
    #download_single_keyword('emoji', n=5)

    # Delete any malformed images
    basic_clean()

