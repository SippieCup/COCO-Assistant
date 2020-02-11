import ast
import json
import logging
import os
import shutil
from pycocotools import mask
from pycocotools.coco import COCO

from tqdm import tqdm

from . import coco_converters as converter
from . import coco_stats as stats
from . import coco_visualiser as cocovis
from coco_assistant.utils import anchors
from coco_assistant.utils import det2seg

logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('parso.python.diff').disabled = True

"""
Expected Directory Structure

.
├── images
│   ├── train
│   ├── val
|   ├── test
|
├── annotations
│   ├── train.json
│   ├── val.json
│   ├── test.json


"""


class COCO_Assistant():
    def __init__(self, img_dir=None, ann_dir=None, res_dir=None):
        """
        :param img_dir (str): path to images folder.
        :param ann_dir (str): path to annotations folder.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.res_dir = res_dir

        # if os.path.dirname(ann_dir) != os.path.dirname(img_dir):
        #     raise AssertionError('Directory not in expected format')

        if self.res_dir is None:
            self.res_dir = os.path.join(os.path.dirname(ann_dir), 'result')

        # Create Results Directory
        if os.path.exists(self.res_dir) is False:
            os.mkdir(self.res_dir)

        self.imgfolders = sorted([i for i in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, i)) is True])
        self.jsonfiles = sorted([j for j in os.listdir(ann_dir) if j[-5:] == ".json"])
        self.names = [n[:-5] for n in self.jsonfiles]
        self.matches = list(set(self.names).intersection(set(self.imgfolders)))

        logging.debug(f'Names: {self.names}')
        logging.debug(f'Image folders: {self.imgfolders}')
        logging.debug(f'Matches: {self.matches}')
        
        # Note: Add check for confirming these folders only contain .jpg and .json respectively
        logging.debug("Number of compatible datasets: = %s", len(self.matches))

        if not self.names:
            raise AssertionError("Annotation files not passed")
        if not self.imgfolders:
            raise AssertionError("Image folders not passed")
        if not self.matches:
            raise AssertionError("Matching Annotation and Image not passed")

        self.annfiles = [COCO(os.path.join(ann_dir, i)) for i in self.jsonfiles]
        self.anndict = dict(zip(self.jsonfiles, self.annfiles))
        self.ann_anchors = []

    def merge(self, res_dir=None, merge_images=True):
        """
        Function for merging multiple coco datasets
        """
        if res_dir:
            self.res_dir=res_dir
        self.merge_images = merge_images
        self.resim_dir = os.path.join(os.path.dirname(self.ann_dir), 'result', 'images')
        self.resann_dir = os.path.join(os.path.dirname(self.ann_dir), 'result', 'annotations')

        # Create directories for merged results and clear the previous ones
        # The exist_ok is for dealing with merged folder
        # TODO: Can be done better
        
        if os.path.exists(self.resann_dir) is False:
            os.makedirs(self.resann_dir, exist_ok=True)
        else:
            shutil.rmtree(self.resann_dir)
            os.makedirs(self.resann_dir, exist_ok=True)

        # Merge images
        if self.merge_images:
            if os.path.exists(self.resim_dir) is False:
                os.makedirs(self.resim_dir, exist_ok=True)
            else:
                shutil.rmtree(self.resim_dir)
                os.makedirs(self.resim_dir, exist_ok=True)

            print("Merging image dirs")
            im_dirs = [os.path.join(self.img_dir, folder) for folder in self.matches]
            imext = [".png", ".jpg"]

            logging.debug("Merging Image Dirs...")

            for imdir in tqdm(im_dirs):
                ims = [i for i in os.listdir(imdir) if i[-4:].lower() in imext]
                for im in ims:
                    shutil.copyfile(os.path.join(imdir, im), os.path.join(self.resim_dir, im))

        else:
            logging.debug("Not merging images")

        cann = {
                    'images': [],
                    'annotations': [],
                    'info': None,
                    'licenses': None,
                    'categories': None
                }

        logging.debug("Merging Annotations...")

        dst_ann = os.path.join(self.resann_dir, 'merged.json')

        for j in tqdm(self.jsonfiles):
            with open(os.path.join(self.ann_dir, j)) as a:
                cj = json.load(a)

            ind = self.jsonfiles.index(j)
            # Check if this is the 1st annotation.
            # If it is, continue else modify current annotation
            if ind == 0:
                cann['images'] = cann['images'] + cj['images']
                
                #Convert to RLE
                for i, annotation in enumerate(cj['annotations']):
                    rle = self.annfiles[ind].annToRLE(annotation)
                    rle['counts'] = rle['counts'].decode('ascii')
                    annotation['segmentation'] = rle

                cann['annotations'] = cann['annotations'] + cj['annotations']
                if 'info' in list(cj.keys()):
                    cann['info'] = cj['info']
                if 'licenses' in list(cj.keys()):
                    cann['licenses'] = cj['licenses']
                cann['categories'] = cj['categories']

                last_annid = cann['annotations'][-1]['id']

                # If last imid or last_annid is a str, convert it to int
                if isinstance(last_annid, str):
                    logging.debug("String Ids detected. Converting to int")
                    id_dict = {}
                    # Change image id in images field
                    for i, im in enumerate(cann['images']):
                        id_dict[im['id']] = i
                        im['id'] = i

                    # Change annotation id & image id in annotations field
                    for i, im in enumerate(cann['annotations']):
                        im['id'] = i
                        if isinstance(last_imid, str):
                            im['image_id'] = id_dict[im['image_id']]

                last_annid = cann['annotations'][-1]['id']

            else:
                # Change annotation and image ids in annotations field
                for i, ann in enumerate(cj['annotations']):
                    ann['id'] = last_annid + i + 1

                cann['annotations'] = cann['annotations'] + cj['annotations']
                if 'info' in list(cj.keys()):
                    cann['info'] = cj['info']
                if 'licenses' in list(cj.keys()):
                    cann['licenses'] = cj['licenses']
                cann['categories'] = cj['categories']

                last_annid = cann['annotations'][-1]['id']

        with open(dst_ann, 'w') as aw:
            json.dump(cann, aw)

    def remove_cat(self, interactive=True, jc=None, rcats=None, res_dir=None):
        """
        Function for removing certain categories.
        In interactive mode, you can input the json and the categories
        to be removed (as a list, see README for example)
        In non-interactive mode, you manually pass in json filename and
        categories to be removed. Note that jc and
        rcats cannot be None if run with interactive=False.

        :param interactive: Run category removal in interactive mode
        :param jc: Json choice
        :param rcats: Categories to be removed
        """
        if res_dir:
            self.resrm_dir = res_dir
        else:
            self.resrm_dir = os.path.join(self.res_dir, 'removal')
        
        if os.path.exists(self.resrm_dir) is False:
            os.makedirs(self.resrm_dir, exist_ok=True)
        else:
            shutil.rmtree(self.resrm_dir)
            os.makedirs(self.resrm_dir, exist_ok=True)

        if interactive:
            print(self.jsonfiles)
            print("Who needs a cat removal?")
            self.jc = input()
            if self.jc.lower() not in [item.lower() for item in self.jsonfiles]:
                raise AssertionError("Choice not in json file list")
            ind = self.jsonfiles.index(self.jc.lower())
            ann = self.annfiles[ind]

            print("\nCategories present:")
            cats = [i['name'] for i in ann.cats.values()]
            print(cats)

            self.rcats = []
            print("\nEnter categories you wish to remove as a list:")
            x = input()
            x = ast.literal_eval(x)
            if isinstance(x, list) is False:
                raise AssertionError("Input must be a list of categories to be removed")
            if all(elem in cats for elem in x):
                self.rcats = x
            else:
                print("Incorrect entry.")

        else:
            if jc is None or rcats is None:
                raise AssertionError("Both json choice and rcats need to be provided in non-interactive mode")
            self.jc = jc
            ind = self.jsonfiles.index(self.jc.lower())
            ann = self.annfiles[ind]
            self.rcats = rcats

        print("Removing specified categories...")

        # Gives you a list of category ids of the categories to be removed
        catids_remove = ann.getCatIds(catNms=self.rcats)
        # Gives you a list of ids of annotations that contain those categories
        annids_remove = ann.getAnnIds(catIds=catids_remove)

        # Remove from category list
        cats = ann.loadCats(catids_remove)
        # Remove from annotation list
        anns = ann.loadAnns(annids_remove)

        with open(os.path.join(self.ann_dir, self.jc)) as it:
            x = json.load(it)

        x['categories'] = [i for i in x['categories'] if i not in cats]
        x['annotations'] = [i for i in x['annotations'] if i not in anns]
        print(f"Categories Kept: {x['categories']}")
        with open(os.path.join(self.resrm_dir, self.jc), 'w') as oa:
            json.dump(x, oa)

    def ann_stats(self, stat, arearng=None, show_count=False, save=False):
        """
        Function for displaying statistics.
        """
        if stat == "area":
            stats.pi_area_split(self.annfiles, self.matches, areaRng=arearng, save=save)
        elif stat == "cat":
            stats.cat_count(self.annfiles, self.matches, show_count=show_count, save=save)

    def anchors(self, n, fmt=None, recompute=False):
        """
        Function for generating top 'n' anchors

        :param n: Number of anchors
        :param fmt: Format of anchors ['square', None]
        :param recompute: Rerun k-means and recompute anchors
        """
        if recompute or not self.ann_anchors:
            print("Calculating anchors...")
            a = [anchors.generate_anchors(j, n, fmt) for j in self.annfiles]
            self.ann_anchors = dict(zip(self.matches, a))
        else:
            print("Loading pre-computed anchors")
            print(self.ann_anchors)

    def get_segmasks(self):
        """
        Function for generating segmentation masks.
        """
        for ann, name in zip(self.annfiles, self.matches):
            output_dir = os.path.join(self.res_dir, 'segmasks', name)
            det2seg.det2seg(ann, output_dir)

    def converter(self, to="TFRecord"):
        """
        Function for converting annotations to other formats

        :param to: Format to which annotations are to be converted
        """
        print("Choose directory:")
        print(self.matches)

        dir_choice = input()

        if dir_choice.lower() not in [item.lower() for item in self.matches]:
            raise AssertionError("Choice not in images folder")
        ind = self.matches.index(dir_choice.lower())
        ann = self.annfiles[ind]
        img_dir = os.path.join(self.img_dir, dir_choice)

        converter.convert(ann, img_dir, _format=to)

    def visualise(self):
        """
        Function for visualising annotations.
        """
        print("Choose directory:")
        print(self.matches)

        dir_choice = input()

        if dir_choice.lower() not in [item.lower() for item in self.matches]:
            raise AssertionError("Choice not in images folder")
        ind = self.matches.index(dir_choice.lower())
        ann = self.annfiles[ind]
        img_dir = os.path.join(self.img_dir, dir_choice)
        cocovis.visualise_all(ann, img_dir)


if __name__ == "__main__":
    p = "./data/test"
    img_dir = os.path.join(p, 'images')
    ann_dir = os.path.join(p, 'annotations')

    # TODO: Create tiny dummy datasets and test these functions on them

    cas = COCO_Assistant(img_dir, ann_dir)

    #cas.merge()
    #cas.remove_cat()
    #cas.ann_stats(stat="area",arearng=[10,144,512,1e5],save=False)
    #cas.ann_stats(stat="cat", show_count=False, save=False)
    #cas.visualise()
    #cas.anchors(2)
