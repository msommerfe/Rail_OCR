import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import util as util

from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
from domain_specifics.evn_checker import EVNChecker
from common import enums

class Rail_OCR:
    def __init__(self, path_to_keras_ocr_model = None, path_to_keras_ocr_model_1_line_EVN= None, path_to_keras_ocr_model_2_line_EVN=None, path_to_keras_ocr_model_3_line_EVN = None):
        self.prediction_model = None
        self.prediction_model_1_line_EVN = None
        self.prediction_model_2_line_EVN = None
        self.prediction_model_3_line_EVN = None

        if path_to_keras_ocr_model_1_line_EVN is None:
            path_to_keras_ocr_model_1_line_EVN = "weights/90_3Prozent_2000k_only_EVNH.keras"

        if path_to_keras_ocr_model_3_line_EVN is None:
            path_to_keras_ocr_model_3_line_EVN = "weights/89_7Prozent_1000k_only_EVNV.keras"


        if path_to_keras_ocr_model is not None:
            self.model = load_model(path_to_keras_ocr_model)
            self.prediction_model = tf.keras.models.Model(self.model.input[0], self.model.get_layer(name="dense2").output)

        if path_to_keras_ocr_model_1_line_EVN is not None:
            self.model_1_line_EVN = load_model(path_to_keras_ocr_model_1_line_EVN)
            self.prediction_model_1_line_EVN = tf.keras.models.Model(self.model_1_line_EVN.input[0], self.model_1_line_EVN.get_layer(name="dense2").output)

        if path_to_keras_ocr_model_2_line_EVN is not None:
            self.model_2_line_EVN = load_model(path_to_keras_ocr_model_2_line_EVN)
            self.prediction_model_2_line_EVN = tf.keras.models.Model(self.model_2_line_EVN.input[0], self.model_2_line_EVN.get_layer(name="dense2").output)

        if path_to_keras_ocr_model_3_line_EVN is not None:
            self.model_3_line_EVN = load_model(path_to_keras_ocr_model_3_line_EVN)
            self.prediction_model_3_line_EVN = tf.keras.models.Model(self.model_3_line_EVN.input[0], self.model_3_line_EVN.get_layer(name="dense2").output)


        self.evnchecker = EVNChecker("domain_specifics/railway_data.json")

    @tf.function
    def fast_predict(self, input_data, class_to_predict):
        if class_to_predict == enums.prediction_classes.ONE_LINE_EVN.value and self.prediction_model_1_line_EVN is not None:
            return self.prediction_model_1_line_EVN(input_data, training=False)
        if class_to_predict == enums.prediction_classes.TWO_LINE_EVN.value and self.prediction_model_2_line_EVN is not None:
            return self.prediction_model_2_line_EVN(input_data, training=False)
        if class_to_predict == enums.prediction_classes.THREE_LINE_EVN.value and self.prediction_model_3_line_EVN is not None:
            return self.prediction_model_3_line_EVN(input_data, training=False)

        return self.prediction_model(input_data, training=False)

    def get_only_digets(self, decoded_prediction):
        return ''.join(filter(str.isdigit, decoded_prediction[0]))


    def predict_single_cropped_img(self, image, class_to_predict = None, advanced_prediction = True,  doPlot=False):
        """
        Diese Funktion liest die auf dem übergebenen Bild aus. Das Bild sollte bereits
        gecropped sein, und nicht mehr als die EVN selbst zeigen.

        Parameter:
        - prediction_model: Ein trainiertes TensorFlow-Vorhersagemodell.
        - img: Entweder ein Np.ndarray oder ein Pfad auf ein Bild

        Rückgabewert:
        - predicted_evn: die durch das Vorhersagemodell vorhergesagt EVN
        - das für das NN vorverarbeitete Bild (informativ)
        """

        # Verarbeite das Eingabebild
        image_array = util.process_single_sample(image, [], [], [])

        # Transformiere das Bild für das Modell
        image_to_predict = tf.expand_dims(image_array["input_data"], axis=0)

        # Mache die Vorhersage
        prediction = self.fast_predict(image_to_predict, class_to_predict)

        # Dekodiere die Vorhersage
        decoded_prediction = util.decode_predictions(prediction)

        # EVN Erstellen. Alles was keine Zahl ist löschen
        predicted_evn = self.get_only_digets(decoded_prediction)

        # Visualisiere das Bild
        if doPlot:
            self.visualize_image(image_to_predict, predicted_evn)

        if advanced_prediction:
            if not self.evnchecker.is_valid_EVN(predicted_evn):
                second_try_img = self.__prepare_img_for_second_run(image_to_predict)
                prediction = self.fast_predict(second_try_img, class_to_predict)
                decoded_prediction = util.decode_predictions(prediction)
                predicted_evn = self.get_only_digets(decoded_prediction)
                if self.evnchecker.is_valid_EVN(predicted_evn):
                    print('BIIIIINNNNNNNGGGOGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                # Visualisiere das Bild
                if doPlot:
                    self.visualize_image(second_try_img, predicted_evn)
        #@TODO: Eine weitere advanced Prediction einbauen, die das Bild in eine Richtung verzerrt.Könnte gut kommen




        return predicted_evn, image_to_predict

    def __prepare_img_for_second_run(self, image_to_predict):
        # Konvertiere das Tensor-Image in ein NumPy-Array
        image_np = image_to_predict.numpy()

        # Entferne die Batch-Dimension
        image_np = np.squeeze(image_np, axis=0)

        # Entferne die Kanal-Dimension
        image_np = np.squeeze(image_np, axis=-1)

        # Konvertiere das NumPy-Array in ein PIL-Image
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        # Speichere die Positionen der vollständig schwarzen Pixel
        black_pixel_mask = (image_np == 0)

        # Kontrasterhöhung
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(1.3)

        # Schärfen des Bildes
        image_pil = image_pil.filter(ImageFilter.SHARPEN)

        # Konvertiere das PIL-Image zurück in ein NumPy-Array
        image_pil = np.array(image_pil).astype(np.float32) / 255.0

        # Setze die vollständig schwarzen Pixel wieder auf 0
        image_pil[black_pixel_mask] = 0

        # Füge die Kanal-Dimension wieder hinzu
        image_pil = np.expand_dims(image_pil, axis=-1)

        # Füge die Batch-Dimension wieder hinzu
        image_pil = np.expand_dims(image_pil, axis=0)

        # Konvertiere das NumPy-Array zurück in ein Tensor
        binary_image_tensor = tf.convert_to_tensor(image_pil, dtype=tf.float32)

        return binary_image_tensor


    def predict_cropped_images_in_folder(self, image_folder,advanced_prediction = True, doPlot=True, plot_only_false = False):
        """
        Diese Funktion erwartet ein Trainiertes Vorhersagemodell (Es sollte keinen CTC-Layer mehr enthalten)
        verarbeitet Bilder aus einem angegebenen Ordner und zeigt die Vorhersagen an.

        Parameter:
        - model: Das trainierte TensorFlow-Modell.
        - image_folder: Der Pfad zum Ordner, der die Bilder enthält.

        Rückgabewert:
        - List of [predicted_evn, predicted_image].
        """

        # Liste der Bilder im Ordner erhalten
        image_names = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        prediction_results = []
        for img_name in image_names:
            img_path = os.path.join(image_folder, img_name)
            prediction_result = self.predict_single_cropped_img(img_path,advanced_prediction = advanced_prediction)
            prediction_results.append([prediction_result[0], prediction_result[1], img_path])

        if doPlot:
            self.plot_prediction(prediction_results, plot_only_false = plot_only_false)

        return prediction_results


    def visualize_image(self, image_tensor,predicted_evn = "nan"):
        # Entferne die Batch-Dimension und den Kanal, um das Bild korrekt darzustellen
        image_tensor = tf.squeeze(image_tensor, axis=0)  # Form: (104, 64, 1)
        image_tensor = tf.squeeze(image_tensor, axis=-1)  # Form: (104, 64)

        # Transponiere und zeige das Bild an
        plt.imshow(tf.transpose(image_tensor), cmap='gray')
        plt.suptitle(f"{self.evnchecker.is_valid_EVN(predicted_evn)}: {predicted_evn}")
        plt.show()

    def __plot_image_in_multiplot(self, ax, image_path, predicted_evn):
        image_array = util.process_single_sample(image_path, [], [], [])
        image_to_predict = tf.expand_dims(image_array["input_data"], axis=0)

        # Entferne die zusätzlichen Dimensionen für die Darstellung
        image_tensor = tf.squeeze(image_to_predict, axis=0)  # Form: (104, 64, 1)
        image_tensor = tf.squeeze(image_tensor, axis=-1)  # Form: (104, 64)

        # Zeige das Bild an
        ax.imshow(tf.transpose(image_tensor), cmap='gray')
        dateiname = os.path.basename(image_path)
        validEVN = self.evnchecker.is_valid_EVN(predicted_evn)

        if validEVN:  ax.set_title(f"{validEVN}: {predicted_evn}", fontsize=18, pad=10)
        else:         ax.set_title(f"{validEVN}: {predicted_evn}  {dateiname} ", fontsize=12, pad=10)

        ax.axis('off')

    def plot_prediction(self, prediction, plot_only_false = False):
        images_per_plot = 30
        # Filtere die Bilder mit falscher EVN
        if plot_only_false:
            prediction = [item for item in prediction if not self.evnchecker.is_valid_EVN(item[0])]

        num_images = len(prediction)
        num_plots = (num_images + images_per_plot - 1) // images_per_plot  # Berechne die Anzahl der benötigten Plots

        for plot_index in range(num_plots):
            fig, axs = plt.subplots(5, 6, figsize=(20, 12))
            axs = axs.flatten()
            start_index = plot_index * images_per_plot
            end_index = min(start_index + images_per_plot, num_images)

            for i in range(start_index, end_index):
                predicted_evn, predicted_image, image_path = prediction[i]
                self.__plot_image_in_multiplot(axs[i - start_index], image_path, predicted_evn)

            # Verstecke verbleibende Subplots, falls weniger als 30 Bilder in diesem Plot sind
            for j in range(end_index - start_index, images_per_plot):
                axs[j].axis('off')

            plt.tight_layout()
            plt.show()

    def evaluate_model(self, image_folder, advanced_prediction=True, debug=False):
        count_true = 0
        count_false = 0

        # Get a list of images in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)

            predicted_evn, _ = self.predict_single_cropped_img(image_path,advanced_prediction=advanced_prediction)

            if self.evnchecker.is_valid_EVN(predicted_evn, debug=debug):
                count_true += 1
            else:
                count_false += 1

            if (count_true+count_false) % 100 == 0:
                print(f"{count_true * 100 / (count_true + count_false)}% correct")
                print(f"{count_true} EVNs correct")
                print(f"{count_false} EVNs wrong")

        return count_true, count_false