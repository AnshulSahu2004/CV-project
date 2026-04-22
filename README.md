## CV Project
## Run the main.py file for execution

# Face Anti-Spoofing System using Deep Learning
 BY:-

* Vanshika Sharma (Btech CSE (NSUT))
* Anshul Sahu
  (Btech CSE (NSUT))



##  Overview

This project focuses on **face anti-spoofing (FAS)** — detecting whether a face presented to a system is **real (live)** or **fake (spoof)**.

Spoof attacks include:

* Printed photos
* Replay videos
* Masks and screen displays

The system enhances a deep learning architecture to achieve **high accuracy, robustness, and real-time performance**.



##  Objective

To design a **robust face liveness detection system** that:

* Works under varying lighting and angles
* Detects subtle spoofing artifacts
* Performs in real-time applications


## Core Model Improvements

We build upon the **ConvNeXt architecture** and introduce the following enhancements:

### 1. Group Convolution

* Learns **separate facial feature patterns**
* Preserves **fine-grained details**
* Prevents feature mixing across channels


### 2. Coordinate Attention

* Helps model focus on **important facial regions**
* Captures:

  * Spatial location
  * Directional dependencies
* Improves **feature localization**



### 3. SPPF (Spatial Pyramid Pooling Fast)

* Captures:

  * Local details (micro features)
  * Global structure (face geometry)
* Enables **multi-scale feature extraction**



### 4. Eye Blink Detection (EAR)

* Uses **Eye Aspect Ratio (EAR)** to detect blinking
* Ensures liveness by detecting:

  * Eye open/close patterns

 Limitation:

* Can be spoofed using pre-recorded videos



##  Advanced Enhancements (Proposed)

### 1. Micro-Motion Detection (rPPG)

**Remote Photoplethysmography (rPPG):**

* Extracts **tiny skin color changes** caused by:

  * Blood flow
  * Heartbeat

## Advantages:

* Invisible to human eye
* Extremely hard to spoof
* Strong liveness indicator



### 2. Depth Estimation (Without Depth Camera)

Instead of using hardware sensors:

* CNN predicts a **depth consistency map**

#### Real Face:

* Nose protrusion
* Eye socket depth
* Natural curvature

#### Fake Face:

* Flat surface
* No depth variation



###  3. Multi-Task Learning

The model jointly learns:

* Spoof classification
* Depth estimation

**Loss Function:**
[
Loss = Spoof_Loss + \lambda \times Depth_Loss
]

✔ Improves:

* Generalization
* Feature robustness

---

##  Architecture

## CNN Feature Hierarchy

| Layer        | Feature Type        |
| ------------ | ------------------- |
| Conv Block 1 | Shallow Features    |
| Conv Block 2 | Low-level patterns  |
| Conv Block 3 | Medium Features     |
| Conv Block 4 | High-level patterns |
| Conv Block 5 | Deep Features       |

---

##  Region-Based Learning

| Region    | Purpose                    |
| --------- | -------------------------- |
| Full Face | Structure + Texture        |
| Cheeks    | Strong blood flow (rPPG)   |
| Forehead  | Clean physiological signal |

---

##  Key Observations

### Real Faces:

* Natural skin texture
* Depth variation
* Blood flow signals

### Fake Faces:

* Printing artifacts
* Screen reflections
* Flat depth
* No physiological signals

---

## Performance

* Accuracy: **Up to 99.64%**
* Real-time capable
* Tested on multiple public datasets

---

### Tech Stack
* Python
* PyTorch
* OpenCV
* MediaPipe
* NumPy / SciPy

##  References

1. IEEE Paper:
   [https://ieeexplore.ieee.org/document/10530019](https://ieeexplore.ieee.org/document/10530019)

2. Additional Works:

   * [https://ieeexplore.ieee.org/document/11091314](https://ieeexplore.ieee.org/document/11091314)
   * [https://ieeexplore.ieee.org/document/10850706](https://ieeexplore.ieee.org/document/10850706)

---

## What I learned
 * Liveness detection is not reliable with only image classification
 * Combining multiple cues leads to better performance
 * Dataset preparation can be more challenging than model training
 * Real-time systems require careful balancing between accuracy and speed

## Future Improvements

* Integrate **Transformer-based architectures**
* Improve robustness against **deepfake attacks**
* Optimize for **edge devices (mobile/IoT)**
* Combine with **3D face reconstruction**

---

## Conclusion

This system combines:

* **Spatial features (depth)**
* **Temporal features (rPPG)**
* **Attention mechanisms**

to create a **highly reliable and scalable face anti-spoofing solution** suitable for real-world applications like:

* Face unlock systems
* Banking authentication
* Surveillance systems

## Note
* This project is built for learning and experimentation purposes and serves as a base for further improvements.





