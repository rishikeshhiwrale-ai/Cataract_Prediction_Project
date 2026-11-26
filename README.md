# VisionCheck: Cataract Classification Demo

VisionCheck is an educational desktop application that classifies an eye image as either **Normal** or **Cataract** using a machine-learning model. It includes a Tkinter interface and can generate a PDF report.

> **Important:** This project is for educational demonstration only. Its output is not a medical diagnosis and must not be used to make healthcare decisions.

## How it works

1. Images stored in class-labeled folders are read by the training script.
2. OpenCV converts each image to grayscale and resizes it to 224 × 224 pixels.
3. Histogram of Oriented Gradients (HOG) extracts image features.
4. A scikit-learn pipeline scales the features and trains an RBF-kernel Support Vector Machine (SVM).
5. The trainer evaluates the model on a held-out test set and saves it.
6. The desktop app loads the saved model, predicts a class for a selected image, and can create a PDF report.

This is a traditional computer-vision and machine-learning project. It uses handcrafted HOG features and an SVM; it does **not** use a deep-learning model or a convolutional neural network.

## Project structure

```text
project/
├── app.py
├── train_model.py
├── dataset/
│   ├── normal/
│   └── cataract/
└── saved_model/                 # Created by train_model.py
    ├── cataract_svm.pkl
    └── evaluation_metrics.json
```

Place supported images directly inside the `normal` and `cataract` folders. The folder names provide the labels used during training.

Supported training image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

## Requirements

- 64-bit Python 3.11 or newer recommended
- Python packages:
  - `numpy`
  - `opencv-python`
  - `scikit-image`
  - `scikit-learn`
  - `reportlab`
  - `Pillow`
- Tkinter, normally included with the Windows Python installer

## Installation

From the project folder, create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python scikit-image scikit-learn reportlab pillow
```

If Python 3.11 is not installed, use another installed Python version supported by the packages, such as `py -3.12`.

## Train the model

Ensure the dataset folders contain images, then run:

```powershell
python train_model.py
```

The trainer prints dataset counts and evaluation results, then creates:

- `saved_model/cataract_svm.pkl` — trained model bundle used by the app
- `saved_model/evaluation_metrics.json` — saved evaluation metrics

## Run the application

After training, launch the desktop app:

```powershell
python app.py
```

Choose an image and select **Run prediction**. Fill in the required patient fields if you want to create a PDF report. The report is saved to the location you choose.

## Evaluation

The training script reports accuracy, balanced accuracy, class-specific recall, cataract precision, macro F1, and a confusion matrix.

The confusion matrix uses actual classes as rows and predicted classes as columns:

|  | Predicted normal | Predicted cataract |
|---|---:|---:|
| Actual normal | Correct normal predictions | Normal images predicted as cataract |
| Actual cataract | Cataract images predicted as normal | Correct cataract predictions |

Metrics from a small or unrepresentative dataset may not reflect performance on new images. Similar images, including images from the same patient or source, should not be split across training and test sets when possible.

## Privacy

The app does not use a database or send information to a server. Patient details and the selected image may be included in the PDF report saved on your computer. Store and share reports carefully.

## Limitations

- This is a classroom prototype, not a clinically validated system.
- The model returns a class label, not a calibrated probability or confidence score.
- Incorrect dataset labels, limited data, image quality, lighting, camera differences, and differences between training images and new images can affect predictions.
- Evaluation on a single held-out split can vary, especially with a small dataset.
- Pickle model files should only be loaded from sources you trust.

## Technologies

Python · Tkinter/ttk · OpenCV · NumPy · scikit-image · scikit-learn · Pillow · ReportLab

## License

Add a license before distributing or reusing this project.
