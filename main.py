from ocr.rail_ocr import Rail_OCR
from common.enums import prediction_classes

# Initialize the Rail_OCR instance
rail_ocr = Rail_OCR()

# Predict the 3-line EVN from an example image
example_evn_3_line = rail_ocr.predict_single_cropped_img(
    "example_images/image_0000.png",
    class_to_predict=prediction_classes.THREE_LINE_EVN.value,
    advanced_prediction=True,
    doPlot=True
)

# Predict the 1-line EVN from another example image
example_evn_1_line = rail_ocr.predict_single_cropped_img(
    "example_images/image_0001.png",
    class_to_predict=prediction_classes.ONE_LINE_EVN.value,
    advanced_prediction=True,
    doPlot=True
)

# Print the predictions
print(f"The predicted 3-Line EVN: {example_evn_3_line[0]}")
print(f"The predicted 1-Line EVN: {example_evn_1_line[0]}")
