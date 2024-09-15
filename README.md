# HackaWu: YOLOv3 Training Resources

This repository contains the code and work made for Hackaton 2024 for the Tecnologico de Monterrey challenge.

## Download Files

### [YOLOv3 Configuration File](https://drive.google.com/file/d/1T80k25Qc56WUSdc9NJeH2regYBW3vfZ_/view?usp=drive_link)
- **Filename:** `yolov3_texting.cfg`
- **Description:** This configuration file contains the architecture settings and parameters for the YOLOv3 model. It is required to initialize the YOLOv3 network for training or inference.

### [YOLOv3 Trained Weights](https://drive.google.com/file/d/1BxQcpbM_RaEDsrhTM9yON2cVBk_Q7Q8D/view?usp=drive_link)
- **Filename:** `yolov3_training_2000.weights`
- **Description:** These are the trained weights for the YOLOv3 model after 2000 iterations of training. Use this file to skip the training process and directly apply the model for object detection tasks.

---

## How to Use

1. **Download the Files:**  
   Use the links above to download both the configuration and trained weights.

2. **Load into YOLOv3 Model:**  
   Incorporate the configuration and weights into your YOLOv3 setup for training or inference. Make sure the paths to these files are correctly set in your project.

3. **Run Object Detection:**  
   Utilize the pre-trained model to detect objects by feeding in your dataset or images. You can also continue training the model from the provided weights.

---

## Additional Information

- Ensure you have a proper setup of YOLOv3, including necessary dependencies such as OpenCV, Darknet, or TensorFlow (depending on your framework).
- You may further fine-tune the model by modifying the `yolov3_texting.cfg` file or continuing the training with additional data.

For more details on YOLOv3 implementation and training, refer to the official [YOLOv3 documentation](https://pjreddie.com/darknet/yolo/).

---
## Project Setup Instructions

Welcome to the project! Below are the necessary steps and commands to get everything up and running smoothly.

## 🛠️ Prerequisites

Before starting, ensure you have **pip** or **pip3** installed on your system. These are Python's package installers, and you'll need them to install all required dependencies.

## 🔧 Installation Guide

Run the following commands in your terminal to install the necessary libraries:

```bash
pip3 install opencv-python
pip3 install mediapipe
pip3 install flask
pip3 install flask_sqlalchemy
pip3 install pytz
pip3 install twilio
pip3 install numpy
pip3 install pymongo
```
---
## 🍏 MongoDB Setup (MacOS)

Add the MongoDB tap:
```bash
brew tap mongodb/brew
```

Install the MongoDB Community Edition:
```bash
brew install mongodb-community
```

Start the MongoDB service:
```bash
brew services start mongodb/brew/mongodb-community
```

---

**Happy Hacking with HackaWu!**
