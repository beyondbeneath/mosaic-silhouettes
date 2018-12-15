# Mosaic Silhouettes - Generate an actual mosaic
# Geoff Sims <geoffrey.sims@gmail.com>

import argparse
from collections import Counter
import lap
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import sys
import tensorflow as tf
import umap

# Fixed directory to output data into
output_dir = os.path.join(os.getcwd(),'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_file', type=str, required=False, help='Filename of previously generated vectors (e.g., output/vectors.json')
    parser.add_argument('--thumb_dir', type=str, required=False, help='Directory of previously generated thumbnail tiles (e.g., img_thumbs/')
    parser.add_argument('--thumb_size', type=int, default=20, required=False, help='Size [pixels] of thumbnail tile')
    parser.add_argument('--dim_red_skip', type=bool, default=False, required=False, help='Skip the dimension reduction if it has already been done')
    parser.add_argument('--dim_red_type', type=str, default='umap', choices=['tsne','umap'], required=False, help='t-SNE or UMAP dimension reduction')
    parser.add_argument('--dim_red_output', type=bool, default=True, required=False, help='Whether or not to output a scatter plot of the t-SNE and UMAP output')
    parser.add_argument('--mask', type=str, default='frog_mask.png', required=True, help='Filename of the shape mask')
    parser.add_argument('--mask_channel', type=int, default=0, required=False, help='Which channel to use for the mask')
    parser.add_argument('--mask_value', type=int, default=0, required=False, help='Which colour value to use for the mask')
    parser.add_argument('--mask_output', type=bool, default=True, required=False, help='Whether or not to output an image of the mask channels')
    parser.add_argument('--coord_output', type=bool, default=True, required=False, help='Whether or not to output an image of the coordinate transform')
    args = parser.parse_args()
    return args

def vectorise():
    """
    Use the Inception model to get the second-to-last layer to use as image vectors
    Adapted from: https://github.com/RohanDoshi2018/visualsearch
    """
    feature_list = []
    img_dir = os.path.join(os.getcwd(), 'img')
    
    # Creates graph from saved graph_def.pb.
    graph_file = os.path.join(os.getcwd(),'model','classify_image_graph_def.pb')
    with tf.gfile.FastGFile(graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #  float description of the image.
            last_layer = sess.graph.get_tensor_by_name('pool_3:0')
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

            for file in os.listdir(img_dir):
                try:
                    img_path = os.path.join(img_dir,file)
                    # JPEG handling
                    if file.endswith(".jpg") or file.endswith(".jpe"):
                        image_data = tf.gfile.FastGFile(img_path, 'rb').read()
                        features = sess.run(last_layer, {'DecodeJpeg/contents:0': image_data})
                        feature_list.append({'filename':file, 'vector':list([float(x) for x in features[0][0][0]])})
                    # PNG/GIF handling
                    if (file.endswith(".png") or file.endswith("gif")):
                        image = Image.open(img_path).convert('RGB')
                        image_array = np.array(image)[:, :, 0:3]
                        features = sess.run(last_layer, {'DecodeJpeg:0': image_array})
                        feature_list.append({'filename':file, 'vector':list([float(x) for x in features[0][0][0]])})
                except:
                    print("Error vectorising {}".format(file))
    return pd.DataFrame(feature_list)

def generate_or_load_vectors(args):
    """
    If no vectors exist, generate new ones using the Inception model
    Otherwise load the file which has been produced previously
    """
    # Generate new vectors
    if args.vector_file is None:
        print("Generating vectors...")
        df_vectors = vectorise()
        # Save vectors for later
        #df_vectors.to_csv(os.path.join(os.getcwd(), 'data', 'vectors.txt'), index=False, sep='\t')
        df_vectors.to_json(os.path.join(os.getcwd(), 'output', 'vectors.json'))
    # Load from file
    else:
        print("Loading vectors...")
        df_vectors = pd.read_json(args.vector_file)
        #df_vectors['vector'] = df_vectors.vector.map(lambda x: np.array([float(i) for i in x[1:-1].split(',')]))

    # Return
    return df_vectors

def resize_and_crop(img_path, modified_path, size, crop_type='middle'):
    """
    Resize and crop an image to fit the specified size.

    args:
    img_path: path for the image to resize.
    modified_path: path to store the modified image.
    size: `(width, height)` tuple.
    crop_type: can be 'top', 'middle' or 'bottom', depending on this
    value, the image will cropped getting the 'top/left', 'middle' or
    'bottom/right' of the image to fit the size.
    raises:
    Exception: if can not open the file in img_path of there is problems
    to save the image.
    ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, int(round((img.size[1] - size[1]) / 2)), img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(round(size[1] * img.size[0] / img.size[1])), size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (int(round((img.size[0] - size[0]) / 2)), 0,
                int(round((img.size[0] + size[0]) / 2)), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]),
            Image.LANCZOS)
    # If the scale is the same, we do not need to crop
    img.save(modified_path)

def generate_thumbnails(args, df_vectors):
    """
    Generate thumbnails from the raw image directory
    """
    if args.thumb_dir is None:
        print("Generating thumbnails...")
        img_dir = os.path.join(os.getcwd(), 'img')
        args.thumb_dir = os.path.join(os.getcwd(), 'img_thumbs')
        # Create directory if needed
        if not os.path.exists(args.thumb_dir):
            os.makedirs(args.thumb_dir)
        # Generate thumbnails
        for img_file in os.listdir(img_dir):
            try:
                source_img = os.path.join(img_dir, img_file)
                dest_img = os.path.join(args.thumb_dir, img_file)
                resize_and_crop(source_img, dest_img, (args.thumb_size,args.thumb_size))
            except:
                print("Error resizing image {}: deleting".format(img_file))
                try:
                    os.remove(source_img)
                    os.remove(dest_img)
                except:
                    pass

    # Clip the vectors dataframe to only those with a valid thumbnail & save it again
    df_vectors = df_vectors[df_vectors.filename.isin(os.listdir(args.thumb_dir))]
    df_vectors.to_json(os.path.join(os.getcwd(), 'output', 'vectors.json'))

    # Return
    return df_vectors 

def dimensionality_reduction(args, df_vectors):
    """
    Reduce the full vectors down to 2 dimensions using both UMAP and t-SNE
    """
    # Skip it only if it's true, and the columns exist
    if args.dim_red_skip == True and 'x' in df_vectors.columns and 'y' in df_vectors.columns:
        print("Skipping dimension reduction as already exists")
    else:
        print("Reducing dimensionality...")
        # Do dimensionality reduction
        reduced_xy_umap = umap.UMAP().fit_transform(df_vectors.vector.tolist())
        reduced_xy_tsne = TSNE().fit_transform(df_vectors.vector.tolist())
        if args.dim_red_type == 'umap': reduced_xy = reduced_xy_umap
        elif args.dim_red_type == 'tsne': reduced_xy = reduced_xy_tsne
        
        # Add all this data to the data frame
        df_vectors['umap_x'] = reduced_xy_umap[:,0]
        df_vectors['umap_y'] = reduced_xy_umap[:,1]
        df_vectors['tsne_x'] = reduced_xy_tsne[:,0]
        df_vectors['tsne_y'] = reduced_xy_tsne[:,1]
        df_vectors['x'] = reduced_xy[:,0]
        df_vectors['y'] = reduced_xy[:,1]
        
        # Save it again
        df_vectors.to_json(os.path.join(os.getcwd(), 'output', 'vectors.json'))

        # Plot the output?
        if args.dim_red_output:
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.scatter(reduced_xy_tsne[:,0], reduced_xy[:,1], c='k', s=5)
            plt.title('t-SNE')
            plt.subplot(122)
            plt.scatter(reduced_xy_umap[:,0], reduced_xy[:,1], c='k', s=5)
            plt.title('UMAP')
            plt.tight_layout()
            plt.savefig(os.path.join(os.getcwd(), 'output', 'dimensionality_reduction.png'), bbox_inches='tight')

    # Return
    return df_vectors

def process_mask(args, num_thumbs):
    """
    Open the mask and determine the scaling parameters based on the number of thumbnails
    """
    # Process the mask & generate the scale
    print ("Processing mask...")
    img_mask = Image.open(os.path.join(os.getcwd(),args.mask))
    np_img = np.array(img_mask)
    np_mask = np_img[:,:,args.mask_channel]
    num_mask_pixels = (np_mask==args.mask_value).sum()
    scale = np.sqrt(num_mask_pixels / num_thumbs)

    # Save mask image for debugging
    if args.mask_output:
        np_img = np.array(img_mask)
        num_channels = np_img.shape[2]
        plt.figure(figsize=(12,6))
        for i in range(num_channels):
            print("Most common colour values for channel {} are:".format(i))
            print(Counter(np_img[:,:,i].flatten()).most_common(5))
            plt.subplot(1,num_channels,i+1)
            plt.imshow(np_img[:,:,i], cmap=plt.cm.Greys_r)
            plt.title('Channel {}'.format(i))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'output', 'mask_debug.png'), bbox_inches='tight')

    # Return
    return np_mask, np_mask.shape, scale

def coordinate_transform(args, df_vectors, np_mask, np_mask_shape, scale):
    """
    Optimally transform dimensionality reduction coordinates to the gridded mask coordinates
    Adapted from: https://github.com/kylemcdonald/ofxAssignment
    Also: https://github.com/genekogan/ofxTSNE
    """

    # First use the calculated scale to generate a gridded coordinate system
    h, w = np_mask_shape
    x1, x2 = 0, w
    y1, y2 = 0, h
    num_x = (x2-x1)/(scale+0.1)
    num_y = (y2-y1)/(scale+0.1)
    x_coords = np.linspace(x1, x2, np.ceil(num_x))
    y_coords = np.linspace(y1, y2, np.ceil(num_y))
    lx, ly = [], []
    # For each grid point, check whether the mask is there or not
    for x in x_coords:
        for y in y_coords:
            cx, cy = int(round(x)), int(round(y))
            try:
                if np_mask[cy,cx] == args.mask_value:
                    lx.append(x)
                    ly.append(y)
            except:
                pass

    print("Found {} positions within the mask (compared with {} images in the dataframe)".format(len(lx), len(df_vectors)))
    print("Finding optimal mapping from scatter to mask...")
    # Normalise the grid values
    xv, yv = np.array(lx)/w, np.array(ly)/h
    grid = np.array((xv,yv)).T
        
    # Convert to 2D array and normalise
    data2d = np.array(df_vectors[['x','y']])
    data2d -= data2d.min(axis=0)
    data2d /= data2d.max(axis=0)

    # Generate cost function
    cost = cdist(grid, data2d, 'sqeuclidean')
    cost = cost * (10000000. / cost.max())

    # Lap 
    min_cost, row_assigns, col_assigns = lap.lapjv(np.copy(cost), extend_cost=True)
    grid_jv = grid[col_assigns]

    # Output image?
    if args.coord_output:
        # Mask with gridded coordinates
        plt.figure(figsize=(8,8))
        plt.imshow(np_mask, cmap=plt.cm.Greys_r)
        plt.scatter(lx, ly, s=2, c='r')
        plt.savefig(os.path.join(os.getcwd(), 'output', 'mask_grid.png'), bbox_inches='tight')

        # Coordinate transforms
        plt.figure(figsize=(8, 8))
        for start, end in zip(data2d, grid_jv):
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_length=0.01, head_width=0.01)
        plt.savefig(os.path.join(os.getcwd(), 'output', 'coordinate_transform.png'), bbox_inches='tight')

    # Add the new gridded coordinates to the dataframe too and save it
    df_vectors['mask_grid_x'] = grid_jv[:,0] * np.ceil(num_x)
    df_vectors['mask_grid_y'] = grid_jv[:,1] * np.ceil(num_y)
    
    # Save it as JSON and text (without the vector)
    df_vectors.to_json(os.path.join(os.getcwd(), 'output', 'vectors.json'))
    df_vectors.drop(['vector'], axis=1).to_csv(os.path.join(os.getcwd(), 'output', 'vectors.txt'), index=False, sep='\t') 

    width = int(np.ceil(num_x))
    height = int(np.ceil(num_y))

    # Return
    return df_vectors, width, height

def generate_mosaic(args, df_vectors, width, height):
    """
    Using the newly transformed coordinates, produce the actual mosaic image
    """
    print("Generating mosaic image...")
    fig = plt.figure(figsize=(width,height))
    for index, row in df_vectors.iterrows():
        filename = os.path.join(args.thumb_dir,row['filename'])
        gridx = int(row['mask_grid_x'])
        gridy = int(row['mask_grid_y'])
        img = Image.open(filename).convert('RGBA')
        ax = plt.subplot2grid((height,width),(gridy-1,gridx-1), xticks=[], yticks=[], frameon=False)
        plt.imshow(img)
                
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(os.getcwd(), 'output', 'mosaic.png'), bbox_inches='tight')

# Main function
if __name__ == '__main__':
    args = get_args()
    df_vectors = generate_or_load_vectors(args)
    df_vectors = generate_thumbnails(args, df_vectors)
    df_vectors = dimensionality_reduction(args, df_vectors)
    np_mask, np_mask_shape, scale = process_mask(args, len(df_vectors))
    df_vectors, width, height = coordinate_transform(args, df_vectors, np_mask, np_mask_shape, scale)
    generate_mosaic(args, df_vectors, width, height)
