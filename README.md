 <div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=230&color=0:020617,45:0F3B55,100:06B6D4&text=VERIFIAGE&fontSize=68&fontColor=FFFFFF&fontAlignY=38&desc=AI-Powered%20Face-Based%20Age%20Estimation&descSize=19&descAlignY=60&animation=fadeIn" width="100%" />

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=600&size=19&duration=2500&pause=700&color=22D3EE&center=true&vCenter=true&width=750&lines=SEE+THE+FACE.+UNDERSTAND+THE+FEATURES.;DEEP+LEARNING+MEETS+COMPUTER+VISION.;DETECT.+ANALYZE.+ESTIMATE." alt="Animated VerifiAge tagline" />

<br/>

<img src="./assets/verifiage-face-scan.svg" width="85%" alt="VerifiAge animated facial scanning visualization" />

<br/>

**A deep learning-powered age estimation system combining face detection, neural networks, real-time inference and explainable AI.**

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-1E293B?style=for-the-badge&logo=brain&logoColor=22D3EE)](#)

</div>

---

## ◈ 01 — Project Vision

**VerifiAge** is an AI-powered facial age estimation system designed to estimate a person's approximate age from facial images.

It combines computer vision-based face detection with a fine-tuned ResNet18 regression model to produce continuous age predictions.

Beyond prediction, VerifiAge incorporates image-quality validation, webcam inference and Grad-CAM visualizations to provide a more interpretable and practical AI experience.

> Not just face detection. Not just prediction. An integrated facial age analysis pipeline.

## ◈ 02 — The Intelligence Behind VerifiAge

<div align="center">

| <img src="https://img.shields.io/badge/01-FACE%20DETECTION-22D3EE?style=for-the-badge" /> | <img src="https://img.shields.io/badge/02-FACE%20VALIDATION-38BDF8?style=for-the-badge" /> |
|---|---|
| OpenCV YuNet detects facial regions. | Face count, confidence and image quality checks. |

| <img src="https://img.shields.io/badge/03-DEEP%20LEARNING-A78BFA?style=for-the-badge" /> | <img src="https://img.shields.io/badge/04-AGE%20ESTIMATION-22D3EE?style=for-the-badge" /> |
|---|---|
| Fine-tuned ResNet18 feature extraction. | Continuous age regression prediction. |

| <img src="https://img.shields.io/badge/05-EXPLAINABLE%20AI-34D399?style=for-the-badge" /> | <img src="https://img.shields.io/badge/06-REAL%20TIME-F472B6?style=for-the-badge" /> |
|---|---|
| Grad-CAM visual explanations. | Webcam inference with prediction smoothing. |

</div>

## ◈ 03 — Model Performance

<div align="center">

<img src="https://img.shields.io/badge/MEAN%20ABSOLUTE%20ERROR-5.9648%20YEARS-06B6D4?style=for-the-badge" />
<img src="https://img.shields.io/badge/RMSE-8.1248%20YEARS-0EA5E9?style=for-the-badge" />
<img src="https://img.shields.io/badge/R%5E2%20SCORE-0.6940-8B5CF6?style=for-the-badge" />
<img src="https://img.shields.io/badge/MEAN%20SIGNED%20ERROR--0.6403-10B981?style=for-the-badge" />

<br/>

| Metric | Evaluated Result |
|:---:|:---:|
| Mean Absolute Error | **5.9648 years** |
| Root Mean Squared Error | **8.1248 years** |
| R² Score | **0.6940** |
| Mean Signed Error | **-0.6403 years** |

</div>

The model uses an ImageNet-pretrained ResNet18 backbone, fine-tuned for continuous age regression.

*Results represent the reported experimental evaluation and do not guarantee individual prediction accuracy.*

## ◈ 04 — System Architecture

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=16&pause=1000&color=22D3EE&center=true&vCenter=true&width=550&lines=FROM+RAW+IMAGE+TO+INTELLIGENT+PREDICTION" alt="Architecture animation" />

</div>

```mermaid
flowchart TD
    A["IMAGE / WEBCAM INPUT"] --> B["OpenCV YuNet"]
    B --> C["Face Validation"]
    C --> D["Quality Assessment"]
    D --> E["Face Crop & Preprocessing"]
    E --> F["ResNet18 Regression"]
    F --> G["Estimated Age"]
    G --> H["Grad-CAM Explanation"]
    H --> I["FastAPI Backend"]
    I --> J["Streamlit Interface"]

    style A fill:#0F172A,stroke:#22D3EE,color:#FFFFFF
    style F fill:#172554,stroke:#38BDF8,color:#FFFFFF
    style G fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    style J fill:#0F172A,stroke:#A78BFA,color:#FFFFFF
```

## ◈ 05 — Technology Arsenal

<div align="center">

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Neural Architecture | ResNet18 |
| Face Detection | OpenCV YuNet |
| Image Processing | OpenCV, PIL |
| Explainability | Grad-CAM |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | MAE, RMSE, R² |

</div>

## ◈ 06 — Experience the Pipeline

<div align="center">

**01 — DETECT**

<img src="https://img.shields.io/badge/FACE-LOCATED-22D3EE?style=for-the-badge&logo=opencv&logoColor=white" />

Identify and validate facial regions from an input image or webcam stream.

<br/>

**02 — ANALYZE**

<img src="https://img.shields.io/badge/FEATURES-EXTRACTED-38BDF8?style=for-the-badge&logo=pytorch&logoColor=white" />

Process the detected face through the trained neural network.

<br/>

**03 — ESTIMATE**

<img src="https://img.shields.io/badge/AGE-PREDICTED-8B5CF6?style=for-the-badge" />

Generate a continuous age estimate and visualization.

</div>

## ◈ 07 — Explainable AI

VerifiAge integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize regions contributing to the prediction.

This helps explore how the model responds to facial image features rather than presenting only a numerical output.

<div align="center">

<img src="./assets/gradcam.png" width="75%" alt="Grad-CAM visualization — add actual project output here" />

<sub>Replace this placeholder with your actual Grad-CAM output.</sub>

</div>

## ◈ 08 — Run VerifiAge Locally

### Clone the repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

### Create an environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

Use the module and application object defined in your FastAPI source:

```bash
uvicorn <module>:<app_object> --reload
```

### Start the interface

```bash
streamlit run <your_streamlit_entrypoint>.py
```

Ensure the trained model checkpoint and YuNet model files are available at the paths configured in the project.

## ◈ 09 — Project Gallery

<div align="center">

| Dashboard | Face Detection |
|---|---|
| ![Dashboard](./assets/dashboard.png) | ![Face Detection](./assets/face-detection.png) |

| Age Estimation | Grad-CAM |
|---|---|
| ![Age Prediction](./assets/age-prediction.png) | ![Grad-CAM](./assets/gradcam.png) |

</div>

*Replace the example image paths with screenshots from your actual application.*

## ◈ 10 — Future Roadmap

- [ ] Explore stronger vision architectures.
- [ ] Improve age prediction across diverse age groups.
- [ ] Expand robustness and bias evaluation.
- [ ] Optimize inference speed.
- [ ] Improve prediction calibration and uncertainty estimation.
- [ ] Enhance real-time visualization and explainability.

## ◈ 11 — Important Disclaimer

VerifiAge is an experimental AI research and development project.

Its predictions are approximate estimates, not verified ages or identity information. Results may be influenced by lighting, image quality, facial appearance and dataset limitations.

It should not be used as the sole basis for legal age verification, identity verification or other high-stakes decisions. Obtain appropriate consent before processing facial images.

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=600&size=21&pause=900&color=22D3EE&center=true&vCenter=true&width=650&lines=THE+FUTURE+OF+FACIAL+AI.;DETECT.+ANALYZE.+ESTIMATE.;WELCOME+TO+VERIFIAGE." alt="Animated closing message" />

<br/>

**Built with Computer Vision · Deep Learning · Explainable AI**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,50:0F3B55,100:020617&height=120&section=footer" width="100%" />

</div>
