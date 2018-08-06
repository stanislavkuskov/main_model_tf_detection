import sys
sys.path.append("..")
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

########################################################################################################################
# [1] Загружаем модель TensorFlow
# Модель можно загрузить из Tensorflow detection model zoo. Из каталога с моделью нужен всего один файл -
# frozen_inference_graph.pb. Это "замороженная" предварительно обученная сеть, которую можно использовать для
# распознавания (inference).
#
# Есть два типа моделей, отличаются выходными данными, указано в слобце Outputs в таблице:
#
# Boxes - модель выдает прямоугольник, внутри которого находится найденный объект
# Masks - модель выдает маску пикселей, которые соответствуют объекту
########################################################################################################################
# Тип выходных данных модели - Boxes
# model_file_name = 'models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
model_file_name = 'models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
detection_graph = tf.Graph()

# Загружаем предварительно обученную модель в память
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_file_name, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

########################################################################################################################
# [2]Загружаем метки классов
# Метки классов из набра данных Common Objects in Context, 90 классов объектов. Файлы с метками классов для разных
# наборов данных находятся в репозитории моделей TensorFlow.
########################################################################################################################
label_map = label_map_util.load_labelmap('object_detection/data/mscoco_label_map.pbtxt')
# print (label_map)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)

category_index = label_map_util.create_category_index(categories)

# print(category_index)

########################################################################################################################
# [3]
# Загружаем изображение и преобразуем его в массив
########################################################################################################################
image_file_name = 'catdog.jpeg'
image = Image.open(image_file_name)
(im_width, im_height) = image.size
image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
plt.figure(figsize=(12, 8))
plt.imshow(image_np)
plt.show()
# Добавляем размерность к изображению (так как обычно tf работает с несколькими изображениями).
image_np_expanded = np.expand_dims(image_np, axis=0)

########################################################################################################################
# [4]
# Запускаем поиск объектов на изображении
########################################################################################################################

with detection_graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Запуск поиска объектов на изображении
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image_np, 0)})

      # Преобразуем выходные тензоры типа float32 в нужный формат
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

########################################################################################################################
# [5]
# Визуализируем результаты распознавания
# Выводится исходная картинка, на ней найденные объекты выделяются прямоугольниками. Рядом с каждым пряоугольником
# написано название класса объекта и вероятность, с которой объект относится к этому классу.
########################################################################################################################
vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
plt.figure(figsize=(24, 16))
plt.imshow(image_np)
plt.show()