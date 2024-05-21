TECHNICAL REPORT

SOCIAL DISTANCING USING COMPUTER VISION AND 
DEEP LEARNING
TEAM NAME: OMEGA

DATE OF SUBMISSION: 1 4 / 0 7 / 2 0 2 3 
ABSTRACT
Our project focuses on social distancing using computer vision and deep learning. Many deep learning-based algorithms, utilizing two-dimensional images, have become essential tools in various domains, including the health sector. Minimizing the risks of virus spread involves avoiding physical contact among people. Therefore, the purpose of this work is to provide a deep learning platform for social distance tracking from an overhead perspective. The developed project successfully identifies individuals who walk too closely, violating social distancing guidelines. The efficiency of the model is enhanced through the use of a transfer learning approach, specifically by leveraging a pre-trained model called Person-Detection 0202 [2]. We calculate the distance between individuals in the input video to determine whether social distancing is being followed, and classify people into two categories: low risk and high risk. The dataset for this project has been collected from various sources.
INTRODUCTION
Social distancing detection plays a crucial role in various applications. In our project, we spe-cifically focus on the subdomain of mask detection. The World Health Organization has rec-ommended several disease prevention measures, including implementing social distancing measures and increasing physical space between people in society. If social distancing is im-plemented at the initial stages, it can play a pivotal role in overcoming the spread of the virus and preventing the peak of the pandemic disease [1].

To accomplish this, we employ OpenVINO, a powerful toolkit developed by Intel. Combined with the pretrained model Person-Detection 0202, our aim is to effectively detect moving ob-jects and classify them as people in videos. We then determine the Euclidean distance between individuals [3][2]. Most existing methods rely on frontal or side view video sequences, which require proper camera calibration to accurately map pixels to real-world distances (e.g., feet, meters).

Alternatively, by adopting a top-down approach, specifically an overhead view, we can calcu-late distances more accurately and cover a wider scene. By utilizing the Person-Detection 0202 model and analyzing videos, our project contributes to the field of healthcare, specifically in the context of social distancing [2]. Accurate identification and monitoring of the distance between people can help reduce the transmission of various diseases and viruses.




MOTIVATION BEHIND THE PROBLEM

We are delighted to have been selected for this internship. From the very moment we were chosen, we have been highly motivated to work on this project. Furthermore, our collaboration with Intel grants us access to cutting-edge technologies and expertise, enabling us to explore the full potential of deep learning in the field of object detection. Intel's strong reputation in society has also given us an added advantage. The opportunity to work with industry leaders is a dream come true for any undergraduate student.
PRIOR WORK AND BACKGROUND
Due to the virus named "COVID-19," many people have lost their lives rapidly when social distancing is not maintained. Thus, social distancing has become crucial in saving lives from this type of virus [1].

The current state-of-the-art object detectors with deep learning have their pros and cons in terms of accuracy and speed. Objects within an image may have different spatial locations and aspect ratios. In the existing system, image acquisition is carried out by first selecting the video file and splitting it into frames. Then, the images are used for pedestrian detection, with the op-tion to resize them for better results. The pre-trained model 0202 is specifically designed to detect and localize individuals in images or video frames [2]. It has been trained on a large da-taset and optimized to achieve high accuracy in identifying people. This accuracy is crucial for reliable social distancing detection as it forms the foundation for calculating distances and iden-tifying violations.

The Person Detection model is compatible with OpenVINO, a powerful framework that al-lows the deployment of deep learning models across various hardware platforms [3]. This flex-ibility enables the model to be deployed on edge devices such as cameras and IoT devices, as well as on cloud infrastructure, depending on the project requirements. This versatility enables the implementation of social distancing detection in different settings and environments.

OUR APPROACH
Building upon previous work, we present a computer vision technique for detecting people using a roadside-installed camera. By measuring the Euclidean distance between individuals, our application highlights whether there is sufficient social distance in the video. To improve results, image resizing is implemented in our project.
Our approach involves collecting diverse data and pre-processing it to be compatible with the Person-Detection0202 model, which utilizes the MobileNetV2 algorithm. MobileNetV2 is a 53-layer deep convolutional neural network. The model is integrated into the OpenVINO toolkit for optimized inference. Real-time video processing is employed to detect and track individuals. Distance analysis is conducted using Euclidean distance to calculate the distances between people and identify violations based on social distancing guidelines. These violations are categorized as high risk or low risk. The number of violations is also displayed alongside the frames per second (FPS) metric.
The system's results are visualized and presented, and deployment on edge devices is optimized for efficient processing. Testing and performance evaluation ensure accuracy and effectiveness. Iterative improvement based on user feedback is incorporated.
RESULTS:

Output1:
 
                    Fig1
Output2:
 
    Fig2
REFERENCES
1.	https://www.moh.gov.my/index.php/pages/view/2019-ncov-wuhan-guidelines 
social distance guidelines.  
2.	https://docs.openvino.ai/2023.0/omz_models_model_person_detection_0202.html 
to choose a pre-trained model.
3.	https://github.com/openvinotoolkit/openvino documentation link for openvino
4.	https://ieeexplore.ieee.org/abstract/document/9243478

LINK TO SOLUTION
GitHub Link: https://github.com/Msamuelcyrilraj/intelunnati_Omega.git 

