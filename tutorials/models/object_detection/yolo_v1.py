import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print('TensorFlow:', tf.__version__)


H, W = 448, 448
num_classes = 2
strategy = tf.distribute.MirroredStrategy()
batch_size = 4 * strategy.num_replicas_in_sync

print('\nNumber of Accelerators :', strategy.num_replicas_in_sync)


def convert_boxes_xywh(boxes):
    x = (1 + boxes[..., 0] + boxes[..., 2]) / 2.0
    y = (1 + boxes[..., 1] + boxes[..., 3]) / 2.0
    w = 1 + boxes[..., 2] - boxes[..., 0]
    h = 1 + boxes[..., 3] - boxes[..., 1]
    return tf.stack([x, y, w, h], axis=-1)


def convert_boxes_x1y1x2y2(boxes):
    x1 = boxes[..., 0] - boxes[..., 2] / 2.0
    y1 = boxes[..., 1] - boxes[..., 3] / 2.0
    x2 = (boxes[..., 0] + boxes[..., 2] / 2.0) - 1
    y2 = (boxes[..., 1] + boxes[..., 3] / 2.0) - 1
    return tf.stack([x1, y1, x2, y2], axis=-1)


def imshow(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)


def draw_boxes_cv2(image, boxes, categories):
    image = np.array(image, dtype=np.uint8)
    boxes = np.array(boxes, dtype=np.int32)
    categories = np.array(categories)
    for _box, _cls in zip(boxes, categories):
        text = _cls
        char_len = len(text) * 9
        text_orig = (_box[0] + 5, _box[1] - 6)
        text_bg_xy1 = (_box[0], _box[1] - 20)
        text_bg_xy2 = (_box[0] + char_len, _box[1])
        image = cv2.rectangle(image, text_bg_xy1, text_bg_xy2, [255, 252, 150],
                              -1)
        image = cv2.putText(image,
                            text,
                            text_orig,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            .6, [0, 0, 0],
                            5,
                            lineType=cv2.LINE_AA)
        image = cv2.putText(image,
                            text,
                            text_orig,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            .6, [255, 255, 255],
                            1,
                            lineType=cv2.LINE_AA)
        image = cv2.rectangle(image, (_box[0], _box[1]), (_box[2], _box[3]),
                              [30, 15, 30], 2)
    return image


shapes_dataset_tfrecords_path = '../../../src/data/shapes_dataset_tfrecords'
train_tfrecords_pattern = shapes_dataset_tfrecords_path + '/train*'
val_tfrecords_pattern = shapes_dataset_tfrecords_path + '/val*'

train_tfrecords = tf.data.Dataset.list_files(train_tfrecords_pattern)
val_tfrecords = tf.data.Dataset.list_files(val_tfrecords_pattern)


def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'xmins': tf.io.VarLenFeature(tf.float32),
        'ymins': tf.io.VarLenFeature(tf.float32),
        'xmaxs': tf.io.VarLenFeature(tf.float32),
        'ymaxs': tf.io.VarLenFeature(tf.float32),
        'classes': tf.io.VarLenFeature(tf.int64),
    }

    parsed_example = tf.io.parse_single_example(example_proto,
                                                feature_description)
    image = tf.io.decode_image(parsed_example['image'], channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image.set_shape([None, None, 3])
    boxes = tf.stack([
        tf.sparse.to_dense(parsed_example['xmins']),
        tf.sparse.to_dense(parsed_example['ymins']),
        tf.sparse.to_dense(parsed_example['xmaxs']),
        tf.sparse.to_dense(parsed_example['ymaxs']),
    ],
        axis=-1)
    classes = tf.sparse.to_dense(parsed_example['classes'])
    return image, boxes, classes


def create_label(boxes,
                 classes,
                 input_size=[448, 448],
                 grid_shape=[7, 7],
                 stride=64,
                 num_classes=2):
    boxes = tf.cast(boxes, dtype=tf.float32)
    boxes_xywh = convert_boxes_xywh(boxes)

    classes = tf.one_hot(classes, depth=num_classes, dtype=tf.float32)

    num_objects = tf.shape(classes)[0]
    pc = tf.ones(shape=[num_objects, 1], dtype=tf.float32)
    box_centers = boxes_xywh[:, :2]
    box_wh = boxes_xywh[:, 2:]

    grid_offset = tf.math.floordiv(box_centers, stride)
    normalized_box_centers = box_centers / stride - grid_offset
    normalized_wh = box_wh / tf.constant(input_size, dtype=tf.float32)

    label_shape = [grid_shape[0], grid_shape[1], 2 * 5 + num_classes]
    label = tf.zeros(shape=label_shape, dtype=tf.float32)

    normalized_box = tf.concat([normalized_box_centers, normalized_wh], axis=-1)

    targets = tf.concat([pc, pc, normalized_box, normalized_box, classes],
                        axis=-1)
    targets = tf.reshape(targets, shape=[1, num_objects, 2 * 5 + num_classes])

    # x, y -> y, x;  because axis 0 of the tensor is the y axis of the 2d coordinate system
    grid_offset_reversed = tf.reverse(grid_offset, axis=[1])
    indices = tf.cast(grid_offset_reversed, dtype=tf.int32)
    indices = tf.reshape(indices, shape=[1, num_objects, 2])

    label = tf.tensor_scatter_nd_update(label, indices, targets)
    return label


def random_flip_data(image, boxes):
    w = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]],
            axis=-1)
    return image, boxes


def parse_and_create_label(example_proto):
    image, boxes, classes = parse_example(example_proto)
    image = (image - 127.5) / 127.5
    image, boxes = random_flip_data(image, boxes)
    label = create_label(boxes, classes)
    return image, label


def build_dataset(tfrecords, batch_size):
    dataset = tfrecords.interleave(
        tf.data.TFRecordDataset,
        cycle_length=8,
        block_length=32,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(512)
    dataset = dataset.map(parse_and_create_label,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def conv_block(tensor, filters, kernel_size, stride):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=stride,
                               padding='same',
                               use_bias=False,
                               kernel_initializer='he_normal')(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    return x


def build_model(H, W, num_classes):
    stride = 64
    grid_shape = [H // stride, W // stride]

    image_input = tf.keras.Input(shape=[H, W, 3], name='image_input')
    x = conv_block(image_input, filters=16, kernel_size=3, stride=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = conv_block(x, filters=32, kernel_size=3, stride=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = conv_block(x, filters=64, kernel_size=3, stride=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = conv_block(x, filters=128, kernel_size=3, stride=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = conv_block(x, filters=256, kernel_size=3, stride=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = conv_block(x, filters=512, kernel_size=3, stride=1)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = conv_block(x, filters=1024, kernel_size=3, stride=1)
    x = conv_block(x, filters=1024, kernel_size=3, stride=1)
    x = conv_block(x, filters=1024, kernel_size=3, stride=1)
    predictions = tf.keras.layers.Conv2D(filters=(2 * 5 + num_classes),
                                         kernel_size=1)(x)
    return tf.keras.Model(inputs=[image_input],
                          outputs=[predictions],
                          name='yolo')


class DecodePredictions(tf.keras.layers.Layer):

    def __init__(self,
                 input_size=[448, 448],
                 stride=64.0,
                 iou_threshold=0.5,
                 score_threshold=0.6,
                 **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self.input_size = tf.constant(input_size, dtype=tf.float32)
        self.stride = tf.constant(stride, dtype=tf.float32)
        self.grid_shape = self.input_size / self.stride
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self._build_meshgrid()

    def _build_meshgrid(self):
        self.meshgrid = tf.stack(tf.meshgrid(tf.range(self.grid_shape[0]),
                                             tf.range(self.grid_shape[1])),
                                 axis=-1)
        self.meshgrid = tf.reshape(
            self.meshgrid,
            shape=[1, self.grid_shape[0], self.grid_shape[1], 1, 2])
        self.meshgrid = tf.tile(self.meshgrid, multiples=[1, 1, 1, 2, 1])

    def call(self, inputs, square_wh, extract_boxes):
        cls_pred = inputs[:, :, :, 10:]
        object_scores = inputs[:, :, :, :2]

        box_coords = tf.reshape(
            inputs[:, :, :, 2:10],
            shape=[-1, self.grid_shape[0], self.grid_shape[1], 2, 4])
        box_centers = box_coords[:, :, :, :, :2]
        box_wh = box_coords[:, :, :, :, 2:]

        denormalized_centers = (box_centers + self.meshgrid) * self.stride
        if square_wh:
            box_wh = tf.square(box_wh)
        denormalized_wh = box_wh * tf.reshape(self.input_size,
                                              shape=[1, 1, 1, 1, 2])
        denormalized_boxes = tf.concat([denormalized_centers, denormalized_wh],
                                       axis=-1)

        if not extract_boxes:
            return denormalized_boxes
        denormalized_boxes = tf.reshape(denormalized_boxes, [-1, 4])

        cls_ids = tf.argmax(cls_pred, axis=-1)
        cls_ids = tf.tile(tf.expand_dims(cls_ids, axis=-1),
                          multiples=[1, 1, 1, 2])
        cls_ids = tf.reshape(cls_ids, [-1])
        cls_scores = tf.reduce_max(cls_pred, axis=-1)
        cls_scores = tf.tile(tf.expand_dims(cls_scores, axis=-1),
                             multiples=[1, 1, 1, 2])

        class_probs = cls_scores * object_scores
        class_probs = tf.reshape(class_probs, [-1])

        denormalized_boxes_x1y1x2y2 = convert_boxes_x1y1x2y2(denormalized_boxes)
        indices = tf.image.non_max_suppression(denormalized_boxes_x1y1x2y2,
                                               class_probs, 100,
                                               self.iou_threshold,
                                               self.score_threshold)

        nms_boxes = tf.gather(denormalized_boxes_x1y1x2y2,
                              indices,
                              name='boxes')
        nms_class_probs = tf.gather(class_probs, indices, name='scores')
        nms_cls_ids = tf.gather(cls_ids, indices, name='class_ids')
        return {
            'boxes': nms_boxes,
            'class_ids': nms_cls_ids,
            'scores': nms_class_probs
        }

    def compute_output_shape(self, input_shape):
        return ([None, 4], [None], [None])


class YoloLoss(tf.losses.Loss):

    def __init__(self,
                 input_size=[448, 448],
                 num_classes=10,
                 stride=64.0,
                 lambdacoord=5.0,
                 lambdanoobj=0.5):
        super(YoloLoss, self).__init__(reduction=tf.losses.Reduction.NONE,
                                       name='YoloLoss')
        self.lambdacoord = lambdacoord
        self.lambdanoobj = lambdanoobj
        self.decoder = DecodePredictions(input_size=input_size, stride=stride)

    def compute_iou(self, boxes1, boxes2):
        boxes1_t = convert_boxes_x1y1x2y2(boxes1)
        boxes2_t = convert_boxes_x1y1x2y2(boxes2)

        lu = tf.maximum(boxes1_t[:, :, :, :, :2], boxes2_t[:, :, :, :, :2])
        rd = tf.minimum(boxes1_t[:, :, :, :, 2:], boxes2_t[:, :, :, :, 2:])

        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        square1 = boxes1[:, :, :, :, 2] * boxes1[:, :, :, :, 3]
        square2 = boxes2[:, :, :, :, 2] * boxes2[:, :, :, :, 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    @tf.function
    def call(self, y_true, y_pred):
        y_pred_shape = tf.shape(y_pred)

        cls_true = y_true[:, :, :, 10:]
        cls_pred = y_pred[:, :, :, 10:]

        pc_true = y_true[:, :, :, :1]
        pc_pred = y_pred[:, :, :, :2]

        box_coord_true = y_true[:, :, :, 2:10]
        box_coord_true = tf.reshape(
            box_coord_true,
            shape=[y_pred_shape[0], y_pred_shape[1], y_pred_shape[2], 2, 4])

        box_coord_pred = y_pred[:, :, :, 2:10]
        box_coord_pred = tf.reshape(
            box_coord_pred,
            shape=[y_pred_shape[0], y_pred_shape[1], y_pred_shape[2], 2, 4])

        box_true_xy = box_coord_true[:, :, :, :, :2]
        box_pred_xy = box_coord_pred[:, :, :, :, :2]

        box_true_wh_sqrt = tf.sqrt(box_coord_true[:, :, :, :, 2:])
        box_pred_wh_sqrt = box_coord_pred[:, :, :, :, 2:]

        box_coord_true_denormalized = self.decoder(y_true,
                                                   square_wh=False,
                                                   extract_boxes=False)
        box_coord_pred_denormalized = self.decoder(y_pred,
                                                   square_wh=True,
                                                   extract_boxes=False)

        ious = self.compute_iou(box_coord_true_denormalized,
                                box_coord_pred_denormalized)
        ious_max = tf.reduce_max(ious, axis=-1, keepdims=True)

        object_mask = tf.cast(tf.equal(pc_true, 1.0), dtype=tf.float32)
        predictor_mask = tf.cast(tf.equal(ious, ious_max), dtype=tf.float32)
        predictor_mask = predictor_mask * tf.tile(object_mask,
                                                  multiples=[1, 1, 1, 2])
        predictor_mask_loc = tf.tile(tf.expand_dims(predictor_mask, axis=-1),
                                     multiples=[1, 1, 1, 1, 2])
        noobj_mask = tf.ones_like(predictor_mask,
                                  dtype=tf.float32) - predictor_mask

        localization_loss_xy = tf.reduce_sum(
            predictor_mask_loc * tf.square(box_true_xy - box_pred_xy),
            axis=[1, 2, 3, 4])
        localization_loss_wh = tf.reduce_sum(
            predictor_mask_loc * tf.square(box_true_wh_sqrt - box_pred_wh_sqrt),
            axis=[1, 2, 3, 4])
        localization_loss = localization_loss_xy + localization_loss_wh

        confidence_loss_obj = tf.reduce_sum(predictor_mask *
                                            tf.square(ious - pc_pred),
                                            axis=[1, 2, 3])
        confidence_loss_noobj = tf.reduce_sum(noobj_mask * tf.square(pc_pred),
                                              axis=[1, 2, 3])

        classification_loss = tf.reduce_sum(object_mask *
                                            tf.square(cls_true - cls_pred),
                                            axis=[1, 2, 3])

        loss = (self.lambdacoord * localization_loss) + confidence_loss_obj + (
            self.lambdanoobj * confidence_loss_noobj) + classification_loss
        return loss


with strategy.scope():
    model = build_model(H, W, 2)
    train_dataset = build_dataset(train_tfrecords, batch_size)
    val_dataset = build_dataset(val_tfrecords, batch_size)
    loss_fn = YoloLoss([H, W], 2, 64.0)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir='../../../model_files/yolo_v1/tensorboard'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../../../model_files/yolo_v1/checkpoints/yolo_v1_weights_{val_loss:.2f}',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
    ]


lr = 0.00004 * strategy.num_replicas_in_sync
model.compile(optimizer=tf.optimizers.Adam(lr), loss=loss_fn)
model.fit(train_dataset,
          validation_data=train_dataset,
          epochs=20,
          steps_per_epoch=10000 // batch_size,
          validation_steps=2500 // batch_size,
          callbacks=callbacks)

best_weights = tf.train.latest_checkpoint('../../../model_files/yolo_v1/checkpoints')
model.load_weights(best_weights)
print('Loaded weights from :', best_weights)

predictions = model.output
x = DecodePredictions(input_size=[448, 448],
                      stride=64.0,
                      iou_threshold=0.5,
                      score_threshold=0.5)(predictions,
                                           extract_boxes=True,
                                           square_wh=True)

inference_model = tf.keras.Model(inputs=model.input, outputs=x)
tf.saved_model.save(inference_model, export_dir='../model_files/yolo_v1/v1')
