# Mosaic Silhouettes

NB: code was originally written in August 2018, and first published here in December 2018.

This project uses Google's pre-trained Inception-v3 model to generate vectors for an arbitrary directory of images (be it emojis, holiday happy-snaps, or even photos of your family). With these vectors, it applies a dimensionality reduction (either [UMAP](https://github.com/lmcinnes/umap) or [t-SNE](https://lvdmaaten.github.io/tsne/)), and then maps those coordinates onto a silhouette mask while preserving the structure. Finally, a mosaic is generated using square thumbnails of the original images.

The net result is a gridded image, in the shape of a masked silhouette, where the image placement is clustered by image similarity (as according to the Inception-v3 vectors):

|iOS Emojis|Christmas time!|Various coloured frogs|
|---|---|---|
| <img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/examples/apple.png" height=300px> | <img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/examples/tree.png" height=300px> | <img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/output/mosaic.png" height=300px> |

This is really just a combination & extension of a few existing projects:

* https://github.com/RohanDoshi2018/visualsearch - obtain the image vectors using Inception-v3 (for similarity)
* https://github.com/kylemcdonald/ofxAssignment - map arbitrary coordinates onto a regular 2D grid
* https://github.com/genekogan/ofxTSNE - gridding similar images onto a regular 2D grid 

# Usage

1. Place the images (`JPG`, `PNG`, or `GIF`) into a subfolder named `img`
2. Find a mask to use (ideally a black [pixel=0] & white [pixel=255] PNG) and place it in the root directory
3. Run `python mosaic.py --mask mask_image.png`

At this point the script will run through a variety of sub-routines, and save the progress as it goes (into `output/vectors.json`) incase something goes wrong. The broad steps are:

1. Vectorise images
2. Generate thumbnails
3. Reduce dimensionality
4. Load mask & determine scale
5. Transform coordinates
6. Produce image

# Configuration

There are a variety of arguments available.

| Argument | Type | Default | Description |
|---|---|---|---|
|`mask`|string|`frog_mask.png`|Filepath of the mask image to use (relative to the root directory)|
|`mask_channel`|int|`0`|Which channel to use for the mask|
|`mask_value`|int|`0`|Which integer value of the mask channel to identify as the silhouette|
|`mask_output`|bool|`true`|Whether or not to save `output/mask_debug.png` to help visual identification of channel & values|
|`vector_file`|str|`None`|Filepath (relative to the root directory) of pre-generated `vectors.json`. Use this if you want to skip this step in future runs to save time (for any given image the vector will never change)|
|`thumb_dir`|str|`None`|Directory path (relative to the root directory) of thumbnail images. Use this if you want to skip this step in future runs to save time |
|`thumb_size`|int|`20`|Size (in pixels) of the thumbnail square which is produced. Increase this if you want a higher resolution image |
|`dim_red_skip`|bool|`False`|Skip the dimension reduction if it has already been done to save time. If this is selected it will read the reduced `x` and `y` coordinates from the `output/vectors.json` file|
|`dim_red_type`|str|`umap`|Which dimensionality algorithm to use. Should be either `umap` or `tsne`|
|`dim_red_output`|bool|`True`|Whether or not to output a scatter plot of the t-SNE and UMAP output|
|`coord_output`|bool|`True`|Whether or not to output an image of the coordinate transforms|
    
# Output

For a successful run of the script, the following files are saved (in `output/`):

|File|Description|Example|
|---|---|---|
| `mosaic.png` | The actual mosaic silhouette PNG |<img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/output/mosaic.png" width=100px>|
| `vectors.json` | A JSON file containing the filename, image vector, dimension reduced coordinates (both UMAP and t-SNE), and gridded coordinates. ||
| `vectors.txt` | A tab-separated text flie containing all the features above except the vector. This can be used for easy post-analysis (e.g., `df = pd.read_txt('vectors.txt', sep='\t')`. ||
| `mask_debug.png` | An image showing the various mask channels and an idea of their colour values. For further debugging, the top 5 values are further printed out in the `stdout` of the script. |<img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/output/mask_debug.png" width=100px>|
| `mask_grid.png` | The mask channel, overlaid with the prospective gridded locations of the images |<img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/output/mask_grid.png" width=100px>|
| `dimension_reduction.png` | A scatterplot showing the UMAP and t-SNE reducations, to aid with algorithm choice |<img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/output/dimensionality_reduction.png" width=100px>|
| `coordinate_transform.png` | Shows the direction of movement from dimension reduced coordinates to the gridded mosaic coordinates |<img src="https://github.com/beyondbeneath/mosaic-silhouettes/blob/master/output/coordinate_transform.png" width=100px>|

# Download Google Images by keyword

The script `download_images.py` is provided as an example of how to easily download a bunch of images from Google Images, using the `google_images_download` (see https://github.com/hardikvasa/google-images-download for more details and installation instructions).

The main function as it stands runs `download_example_frog_images()` which just downloads 100 images of various coloured frogs. The other function `download_single_keyword()` (commented out) allows you to download any arbitrary keyword. It is recommended to see the official documentation for the full suite of options available.

For example, simply running `python download_images.py` will download 900 frog images into `img/` as a starting example.
