# Facial Expression Recognition — Streamlit deployment

## Important clarification
The supplied notebook builds a **ResNet18 image classifier**. It does not implement
Mask R-CNN instance segmentation. These files preserve the classifier approach.

## 1. Dataset layout

Preferred layout:

```text
Facial Expression DataSet/
  train/
    Angry/
    Happy/
    Sad/
  test/
    Angry/
    Happy/
    Sad/
```

The scripts also support `train/train/<class>` and `test/test/<class>`.

If you only have `test/test/<class>` (as in the original notebook), the training
script makes a random 80/20 training/validation split from that folder. This is
not an independent test evaluation. For a proper final test score, provide
separate train and test folders.

## 2. Train locally or in Google Colab

Install dependencies, then run from the project directory:

```bash
pip install -r requirements.txt
python train_model.py --data-dir "Facial Expression DataSet" --epochs 5
```

Training creates:

```text
models/resnet18_expression_model.pth
models/class_names.json
```

Do not run training inside the Streamlit app.

## 3. Deploy to Streamlit Community Cloud

Commit these files to the root of your GitHub repository:

```text
app.py
requirements.txt
models/resnet18_expression_model.pth
```

Set Streamlit's Main file path to `app.py`.

The dataset is not required on Streamlit Cloud for inference. The model checkpoint
is required. If the `.pth` file is large, use Git LFS and ensure the deployed
repository contains the actual LFS object rather than only an LFS pointer.

## 4. Git LFS (if needed)

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add app.py requirements.txt models/resnet18_expression_model.pth
git commit -m "Add facial expression inference app and model"
git push origin main
```
