import collections
import json
import os

from absl import logging
import tensorflow.compat.v2 as tf

import tensorflow_datasets.public_api as tfds

"""mvtec_screws COCO STYLE DATASET."""
_CITATION = """\
@article{NA,
  author    = {NA},
  title     = {MVTEC SCREWS DATASET},
  journal   = {NA},
  volume    = {NA},
  year      = {NA},
  url       = {NA},
  archivePrefix = {NA},
  eprint    = {NA},
  timestamp = {NA},
  biburl    = {NA},
  bibsource = {NA},
}
"""

_DESCRIPTION = """
Note:
 * 
"""

_CONFIG_DESCRIPTION = """
This version contains images, bounding boxes, orientation and labels.
"""

Split = collections.namedtuple(
    'Split', ['name', 'images', 'annotations', 'annotation_type'])


class AnnotationType(object):
  """Enum of the annotation format types.

  Splits are annotated with different formats.
  """
  BBOXES = 'bboxes'
  NONE = 'none'


class MVTEC_SCREWSConfig(tfds.core.BuilderConfig):
  """BuilderConfig for mvtec_screwsConfig."""

  def __init__(self, splits=None, **kwargs):
    super(MVTEC_SCREWSConfig, self).__init__(
        version=tfds.core.Version('1.1.0'), **kwargs)
    self.splits = splits



class MVTEC_SCREWS(tfds.core.GeneratorBasedBuilder):
  """Base Screws dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """

  BUILDER_CONFIGS = [
      MVTEC_SCREWSConfig(
          name='mvtec_screws',
          description=_CONFIG_DESCRIPTION.format(year=2022),
          splits=[
              Split(
                  name=tfds.Split.TRAIN,
                  images='train',
                  annotations='annotations_trainval',
                  annotation_type=AnnotationType.BBOXES,
              ),
          ],
      ),
  ]

  def _info(self):
    features = {
        # Images can have variable shape
        'image': tfds.features.Image(),
        'image/filename': tfds.features.Text(),
        'image/id': tf.int64,
    }
    # Uses original annotations
    if True:
      features.update({
          'objects':
              tfds.features.Sequence({
                  'id': tf.int64,
                  # Coco has unique id for each annotation. The id can be used
                  # for mapping panoptic image to semantic segmentation label.
                  'area': tf.int64,
                  'bbox': tfds.features.BBoxFeature(),
                  'phi': tf.float32,
                  'label': tfds.features.ClassLabel(num_classes=13),
                  'is_crowd': tf.bool,
              }),
      })
    # More info could be added, like segmentation (as png mask), captions,
    # person key-points, more metadata (original flickr url,...).

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # More info could be added, like the segmentation (as png mask),
        # captions, person key-points. For caption encoding, it would probably
        # be better to have a separate class CocoCaption2014 to avoid poluting
        # the main class with builder config for each encoder.
        features=tfds.features.FeaturesDict(features),
        homepage='NA',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'mvtec_screws_data.zip'
    print(dl_manager.manual_dir)
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)

    extracted_paths = dict()
    with os.scandir(extracted_path) as file_list:
        for f in file_list:
            print(f.name)
            extracted_paths[f.name] = f.path


    splits = []
    for split in self.builder_config.splits:
      image_dir = extracted_paths['{}_images'.format(split.name)]
      annotations_dir = extracted_paths['{}_annotations'.format(split.name)]

      splits.append(
          tfds.core.SplitGenerator(
              name=split.name,
              gen_kwargs=dict(
                  image_dir=image_dir,
                  annotation_dir=annotations_dir,
                  split_name=split.images,
                  annotation_type=split.annotation_type,
              ),
          ))
    return splits

  def _generate_examples(self, image_dir, annotation_dir, split_name,
                         annotation_type):
    """Generate examples as dicts.

    Args:
      image_dir: `str`, directory containing the images
      annotation_dir: `str`, directory containing annotations
      split_name: `str`, <split_name><year> (ex: train2014, val2017)
      annotation_type: `AnnotationType`, the annotation format (NONE, BBOXES)

    Yields:
      example key and data
    """

    if annotation_type == AnnotationType.BBOXES:
      instance_filename = 'instances_{}.json'
    elif annotation_type == AnnotationType.NONE:  # No annotation for test sets
      instance_filename = 'image_info_{}.json'

    # Load the annotations (label names, images metadata,...)
    instance_path = os.path.join(
        annotation_dir,
        'annotations',
        instance_filename.format(split_name),
    )
    coco_annotation = ANNOTATION_CLS[annotation_type](instance_path)
    # Each category is a dict:
    # {
    #    'id': 51,  # From 1-91, some entry missing
    #    'name': 'bowl',
    #    'supercategory': 'kitchen',
    # }
    categories = coco_annotation.categories
    # Each image is a dict:
    # {
    #     'id': 262145,
    #     'file_name': 'COCO_train2017_000000262145.jpg'
    #     'flickr_url': 'http://farm8.staticflickr.com/7187/xyz.jpg',
    #     'coco_url': 'http://images.cocodataset.org/train2017/xyz.jpg',
    #     'license': 2,
    #     'date_captured': '2013-11-20 02:07:55',
    #     'height': 427,
    #     'width': 640,
    # }
    images = coco_annotation.images

    # TODO(b/121375022): ClassLabel names should also contains 'id' and
    # and 'supercategory' (in addition to 'name')
    # Warning: As Coco only use 80 out of the 91 labels, the c['id'] and
    # dataset names ids won't match.
    if True:
      objects_key = 'objects'
    self.info.features[objects_key]['label'].names = [
        c['name'] for c in categories
    ]
    # TODO(b/121375022): Conversion should be done by ClassLabel
    categories_id2name = {c['id']: c['name'] for c in categories}

    # Iterate over all images
    annotation_skipped = 0
    for image_info in sorted(images, key=lambda x: x['id']):
      if annotation_type == AnnotationType.BBOXES:
        # Each instance annotation is a dict:
        # {
        #     'iscrowd': 0,
        #     'bbox': [116.95, 305.86, 285.3, 266.03],
        #     'image_id': 480023,
        #     'segmentation': [[312.29, 562.89, 402.25, ...]],
        #     'category_id': 58,
        #     'area': 54652.9556,
        #     'id': 86,
        # }
        instances = coco_annotation.get_annotations(img_id=image_info['id'])

      else:
        instances = []  # No annotations

      if not instances:
        annotation_skipped += 1

      def build_bbox(x, y, width, height):
        # pylint: disable=cell-var-from-loop
        # build_bbox is only used within the loop so it is ok to use image_info
        return tfds.features.BBox(
            ymin= (y-width/2) / image_info['width'],
            xmin= (x-height/2) / image_info['height'],
            ymax= (y+width/2)/ image_info['width'],
            xmax= (x+height/2)/ image_info['height'],
            #ymin=y,
            #xmin=x,
            #ymax=y+height,
            #xmax=x+width,
        )
        # pylint: enable=cell-var-from-loop

      example = {
          'image': os.path.join(image_dir, split_name, image_info['file_name']),
          'image/filename': image_info['file_name'],
          'image/id': image_info['id'],
          objects_key: [{   # pylint: disable=g-complex-comprehension
              'id': instance['id'],
              'area': instance['area'],
              'bbox': build_bbox(instance['bbox'][0],instance['bbox'][1],instance['bbox'][2], instance['bbox'][3]),
              'phi': instance['bbox'][4],
              'label': categories_id2name[instance['category_id']],
              'is_crowd': bool(instance['is_crowd']),
          } for instance in instances]
      }

      yield image_info['file_name'], example

    logging.info(
        '%d/%d images do not contains any annotations',
        annotation_skipped,
        len(images),
    )


class CocoAnnotation(object):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    with tf.io.gfile.GFile(annotation_path) as f:
      data = json.load(f)
    self._data = data

  @property
  def categories(self):
    """Return the category dicts, as sorted in the file."""
    return self._data['categories']

  @property
  def images(self):
    """Return the image dicts, as sorted in the file."""
    return self._data['images']

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    raise NotImplementedError  # AnotationType.NONE don't have annotations


class CocoAnnotationBBoxes(CocoAnnotation):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    super(CocoAnnotationBBoxes, self).__init__(annotation_path)

    img_id2annotations = collections.defaultdict(list)
    for a in self._data['annotations']:
      img_id2annotations[a['image_id']].append(a)
    self._img_id2annotations = {
        k: list(sorted(v, key=lambda a: a['id']))
        for k, v in img_id2annotations.items()
    }

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id  string."""
    # Some images don't have any annotations. Return empty list instead.
    return self._img_id2annotations.get(img_id, [])


class CocoAnnotationPanoptic(CocoAnnotation):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    super(CocoAnnotationPanoptic, self).__init__(annotation_path)
    self._img_id2annotations = {
        a['image_id']: a for a in self._data['annotations']
    }

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    return self._img_id2annotations[img_id]

ANNOTATION_CLS = {
    AnnotationType.NONE: CocoAnnotation,
    AnnotationType.BBOXES: CocoAnnotationBBoxes,
}

if __name__ == "__main__":
    ds = tfds.load("mvtec_screws")