# Add the total path to image name
import string
import tensorflow as tf
import numpy as np


def get_global_var():
    MAX_HIGHT = 64
    MAX_WIDTH = 128  # 104 / 8 = 13 --> predicted label has maximum length of 13
    ALPHABETS = string.digits + '- '


    MAX_STR_LEN = 32 #Maximale Länge der Label, die vom Modell vorhergesagt werden können. Entspricht der Dense2 Dimension.
    BATCH_SIZE = 512
    return MAX_HIGHT, MAX_WIDTH, ALPHABETS, MAX_STR_LEN, BATCH_SIZE

MAX_HIGHT, MAX_WIDTH, ALPHABETS, MAX_STR_LEN, BATCH_SIZE = get_global_var()
BLANK_LABEL = -1 #len(ALPHABETS)
MIN_PAD_NUM = 5 # Es dürfen nur Labels zum Trainieren verwendet werden, die kleiner PREDICTION_LABEL_LENGTH - MIN_PAD_NUM sind


def num_to_label(num_label, blank_label=BLANK_LABEL):
    label = ''.join([ALPHABETS[i] if i < len(ALPHABETS) else '' for i in num_label if i != blank_label])
    return label

def decode_predictions(preds):
    # Berechnung der Input-Längen
    input_length = np.ones(preds.shape[0]) * preds.shape[1]

    # CTC Decoding
    pred_nums, _ = tf.keras.backend.ctc_decode(preds, input_length=input_length, greedy=True)
    pred_nums = pred_nums[0].numpy()  # Konvertierung von Tensor zu NumPy Array

    # Vektorisierte num_to_label-Anwendung
    pred_texts = [num_to_label(pred_num) for pred_num in pred_nums]

    return pred_texts


def process_single_sample(img, labels_padded, len_labels_padded, len_labels_not_padded):
    # 1. Read image
    if type(img) == np.ndarray:
        img = tf.convert_to_tensor(img)
        img = tf.image.rgb_to_grayscale(img)
    else:
        img = tf.io.read_file(img)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size. its important for the resize_with_pad funktion. When picture is way bigge with strange aspect ratio ist rans into an error
    img = tf.image.resize(img, [MAX_HIGHT, MAX_WIDTH], preserve_aspect_ratio=True, antialias=True)
    # 5. Pad the image to MAX_HIGHT and MAX_WIDTH
    img = tf.image.resize_with_pad(img, target_height=MAX_HIGHT, target_width=MAX_WIDTH, antialias=True)
    # 6. Transpose the image because we want the time
    img = tf.transpose(img, perm=[1, 0, 2])
    # 7. Return a dict as our model is expecting two inputs

    labels_padded = tf.convert_to_tensor(labels_padded, dtype="int32")
    len_labels_padded = tf.convert_to_tensor([len_labels_padded], dtype="int32")
    len_labels_not_padded = tf.convert_to_tensor([len_labels_not_padded], dtype="int32")
    #label = tf.keras.utils.pad_sequences(label, maxlen=MAX_STR_LEN)
    #return {"input": img, "gtruth_labels": label, "input_length": img, "label_length": label}
    #return img, labels_padded, len_labels_padded,len_labels_not_padded, [0]
    return {"input_data": img, "input_label": labels_padded, "input_length": len_labels_padded, "label_length":len_labels_not_padded}

