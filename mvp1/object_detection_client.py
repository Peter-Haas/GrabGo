import tensorflow as tf
import numpy as np
from object_detection.TFServingClient import BaseClient

from object_detection.utils import visualization_utils
from object_detection.utils import label_map_util
from settings import OBJECT_DETECTION_HOST, OBJECT_DETECTION_PORT, OBJECT_DETECTION_MODEL


class ObjectDetectionClient(BaseClient):
    """ Object Detection API compliant client for a TensorFlow ModelServer.

        Performs inference on a directory of images by sending them
        to a TensorFlow-Serving ModelServer, using its RESTful API.
    """

    def __init__(
        self,
        url: str = f"http://{OBJECT_DETECTION_HOST}:{OBJECT_DETECTION_PORT}"
        f"{OBJECT_DETECTION_MODEL}:predict",
        # "http://localhost:8501/v1/models/frcnn:predict",
        output_dir: str = "/Users/raj/Desktop/NeyonProjects/test_image/",
        output_filename: str = "output",
        encoding: str = "utf-8",
        channels: int = 3,
        label_path: str = "/Users/raj/Desktop/tendies/full_functionality/object_detection/data/mscoco_label_map.pbtxt",
        image: bytes = b"",
    ):

        """ Initializes an ObjectDetectionClient object.

            Args:
                url: The URL of the TensorFlow ModelServer.
                input_dir: The name of the input directory.
                input_extension: The file extension of input files.
                output_dir: The name of the output directory.
                output_filename: The filename (less extension) of output files.
                output_extension: The file extension of output files.
                encoding: The type of string encoding to be used.
                image_size: The size of the input images.
                label_path: The path to the label mapping file.
        """

        # Initializes a Client object
        super().__init__(url, output_dir, output_filename, encoding, image)
        # Adds child class specific member variables
        self.channels = channels
        self.label_path = label_path

    def get_category_index(self):
        """ Transforms label map into category index for visualization.

            Returns:
                category_index: The category index corresponding to the given
                    label map.
        """
        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True
        )
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def clean_response(self, input_image, response):
        """ Decodes JSON data and converts it to a bounding box overlay
            on the input image, then saves the image to a directory.

            Args:
                input_image: The string representing the input image.
                response: The list of response dictionaries from the server.
                i: An integer used in iteration over input images.
        """

        # Processes response for visualization
        detection_boxes = response["detection_boxes"]
        detection_classes = response["detection_classes"]
        detection_scores = response["detection_scores"]

        # Converts image bitstring to uint8 tensor
        input_bytes = tf.reshape(input_image, [])
        image = tf.image.decode_jpeg(input_bytes, channels=self.channels)

        # Gets value of image tensor
        with tf.Session() as sess:
            image = image.eval()

        return self.get_data_and_images_of_detected_objects(
            image=image,
            boxes=np.asarray(detection_boxes, dtype=np.float32),
            classes=np.asarray(detection_classes, dtype=np.uint8),
            scores=np.asarray(detection_scores, dtype=np.float32),
            category_index=self.get_category_index(),
            min_score_thresh=0.5,
        )

    def get_data_and_images_of_detected_objects(
        self, image, boxes, classes, scores, category_index, min_score_thresh
    ):

        detected_objects_data = []

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]["name"]
                else:
                    class_name = "N/A"
                display_str = str(class_name)
                probability = int(100 * scores[i])

                ymin, xmin, ymax, xmax = box

                im_width = image.shape[1]
                im_height = image.shape[0]
                xmin = int(xmin * im_width)
                xmax = int(xmax * im_width)
                ymin = int(ymin * im_height)
                ymax = int(ymax * im_height)
                # crops.append(image[ymin:ymax, xmin:xmax, :])

                detected_objects_data.append(
                    {
                        "class": display_str,
                        "probability": probability,
                        "ymin": ymin,
                        "ymax": ymax,
                        "xmin": xmin,
                        "xmax": xmax,
                        "image": image[ymin:ymax, xmin:xmax, :],
                    }
                )
        return detected_objects_data

    def visualize(self, input_image, response):
        """ Decodes JSON data and converts it to a bounding box overlay
            on the input image, then saves the image to a directory.

            Args:
                input_image: The string representing the input image.
                response: The list of response dictionaries from the server.
                i: An integer used in iteration over input images.
        """

        # Processes response for visualization
        detection_boxes = response["detection_boxes"]
        detection_classes = response["detection_classes"]
        detection_scores = response["detection_scores"]

        # Converts image bitstring to uint8 tensor
        input_bytes = tf.reshape(input_image, [])
        image = tf.image.decode_jpeg(input_bytes, channels=self.channels)

        # Gets value of image tensor
        with tf.Session() as sess:
            image = image.eval()

            # Overlays bounding boxes and labels on image
            # visualization_utils.visualize_boxes_and_labels_on_image_array(
        image, crops = self.crop_detected_boxes_on_image_array(
            image=image,
            boxes=np.asarray(detection_boxes, dtype=np.float32),
            classes=np.asarray(detection_classes, dtype=np.uint8),
            scores=np.asarray(detection_scores, dtype=np.float32),
            category_index=self.get_category_index(),
            use_normalized_coordinates=True,
            line_thickness=2,
            output_dir=self.output_dir,
        )

        # Saves image
        if isinstance(crops, list):
            for j, crop in enumerate(crops):
                visualization_utils.save_image_array_as_png(crop, f"{self.output_dir}/{j}.png")

        output_file = self.output_dir + "/images/"
        output_file += self.output_filename + ".png"
        visualization_utils.save_image_array_as_png(image, output_file)

        num_detected_items = len([_ for _ in detection_scores if _ >= 0.5])
        print(f"Number of items detected: {num_detected_items}")

    def crop_detected_boxes_on_image_array(
        self,
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=0.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color="black",
        skip_scores=False,
        skip_labels=False,
        output_dir="",
    ):
        """Overlay labeled boxes on an image with formatted scores and label names.

        This function groups boxes that correspond to the same location
        and creates a display string for each detection and overlays these
        on the image. Note that this function modifies the image in place, and returns
        that same image.

        Args:
          image: uint8 numpy array with shape (img_height, img_width, 3)
          boxes: a numpy array of shape [N, 4]
          classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
          scores: a numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
          category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
          instance_masks: a numpy array of shape [N, image_height, image_width] with
            values ranging between 0 and 1, can be None.
          instance_boundaries: a numpy array of shape [N, image_height, image_width]
            with values ranging between 0 and 1, can be None.
          keypoints: a numpy array of shape [N, num_keypoints, 2], can
            be None
          use_normalized_coordinates: whether boxes is to be interpreted as
            normalized coordinates or not.
          max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
            all boxes.
          min_score_thresh: minimum score threshold for a box to be visualized
          agnostic_mode: boolean (default: False) controlling whether to evaluate in
            class-agnostic mode or not.  This mode will display scores but ignore
            classes.
          line_thickness: integer (default: 4) controlling line width of the boxes.
          groundtruth_box_visualization_color: box color for visualizing groundtruth
            boxes
          skip_scores: whether to skip score when drawing a single detection
          skip_labels: whether to skip label when drawing a single detection

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
          [uint8 numpy array] of crops
        """
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        box_to_display_str_map = visualization_utils.collections.defaultdict(list)
        box_to_color_map = visualization_utils.collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_instance_boundaries_map = {}
        box_to_keypoints_map = visualization_utils.collections.defaultdict(list)
        crops = []
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if instance_boundaries is not None:
                    box_to_instance_boundaries_map[box] = instance_boundaries[i]
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ""
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in category_index.keys():
                                class_name = category_index[classes[i]]["name"]
                            else:
                                class_name = "N/A"
                            display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = "{}%".format(int(100 * scores[i]))
                        else:
                            display_str = "{}: {}%".format(display_str, int(100 * scores[i]))
                    box_to_display_str_map[box].append(display_str)
                    if agnostic_mode:
                        box_to_color_map[box] = "DarkOrange"
                    else:
                        box_to_color_map[box] = visualization_utils.STANDARD_COLORS[
                            classes[i] % len(visualization_utils.STANDARD_COLORS)
                        ]

                        # Draw all boxes onto image.
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box

            im_width = image.shape[1]
            im_height = image.shape[0]
            xmin = int(xmin * im_width)
            xmax = int(xmax * im_width)
            ymin = int(ymin * im_height)
            ymax = int(ymax * im_height)
            crops.append(image[ymin:ymax, xmin:xmax, :])

            if instance_masks is not None:
                visualization_utils.draw_mask_on_image_array(
                    image, box_to_instance_masks_map[box], color=color
                )
            if instance_boundaries is not None:
                visualization_utils.draw_mask_on_image_array(
                    image, box_to_instance_boundaries_map[box], color="red", alpha=1.0
                )
            visualization_utils.draw_bounding_box_on_image_array(
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates,
            )

            if keypoints is not None:
                visualization_utils.draw_keypoints_on_image_array(
                    image,
                    box_to_keypoints_map[box],
                    color=color,
                    radius=line_thickness / 2,
                    use_normalized_coordinates=use_normalized_coordinates,
                )

        return image, crops
