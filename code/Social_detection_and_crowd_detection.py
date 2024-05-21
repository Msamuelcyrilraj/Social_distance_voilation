import openvino
from openvino.inference_engine import IECore, IENetwork
import openvino.runtime as ov
import time
import cv2
import numpy as np

# Load the IR model files
model_xml = "model/person-detection-0202.xml"
model_bin = "model/person-detection-0202.bin"

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)

# Get the input and output layer names
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Load the network to the device (CPU, GPU, etc.)
core = ov.Core()
model = core.compile_model(model_xml, "CPU")


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Define a threshold for the minimum distance between people
distance_threshold = 300  # pixels

# Define crowd density threshold and crowd violation threshold
crowd_density_threshold = 0.5
crowd_violation_threshold = 10

# Read and preprocess the input video
video = cv2.VideoCapture("Input_videos\Park.mp4")
_, frame = video.read()
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

# Give the output address to store the video
writer = cv2.VideoWriter('social_distance_openvino_optimized_version.avi', fourcc, 30, (width, height), True)

# Initialize counts
high_risk_count = 0
low_risk_count = 0
total_persons = 0
crowd_violation_count = 0

# Define the region of interest (ROI) for crowd detection
#crowd_roi = (100, 100, 500, 500)  # (x_min, y_min, x_max, y_max)
crowd_density = 0

# Initialize the set to store previously violated individuals
ignore = set()

while True:
    # Reset counts for each frame
    high_risk_count = 0
    low_risk_count = 0
    total_persons = 0
    crowd_violation_count = 0

    # Read a frame from the video.
    ret, frame = video.read()

    if not ret:
        break  # Exit the loop if end of video or error
    height, width = frame.shape[:2]

    # Preprocess the frame
    image = cv2.resize(frame, (512, 512))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    # Run inference and get the output
    infer_request = model.create_infer_request()
    input_shape = [1, 3, 512, 512]
    input_tensor = ov.Tensor(image.astype(np.float32))
    input_tensor.shape = input_shape
    infer_request.set_tensor(input_blob, input_tensor)
    infer_request.start_async()
    infer_request.wait()
    output_tensor = infer_request.get_tensor(output_blob)
    output = output_tensor.data

    # Parse the output and get the bounding boxes of detected people
    boxes = []
    confidences = []
    class_ids = []

    for detection in output[0][0]:
        # Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max]
        if detection[2] > 0.5:  # Only keep detections with confidence > 0.5
            class_id = int(detection[1])
            if class_id == 0:  # Only keep detections with label 0 (person)
                x_min = int(detection[3] * width)
                y_min = int(detection[4] * height)
                x_max = int(detection[5] * width)
                y_max = int(detection[6] * height)
                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(detection[2]))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    v = 0
    person_centers = []
    persons = []
    violate = set()

    for i in indices:
        box = boxes[i]
        # Draw a bounding box around the person
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Get the center point of the box
        center_a = np.array([box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2])
        persons.append(boxes[i])
        person_centers.append(center_a)

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            distance = euclidean_distance(person_centers[i], person_centers[j])
            if distance < distance_threshold:
                violate.add(tuple(persons[i]))
                violate.add(tuple(persons[j]))

    for (x, y, w, h) in persons:
        if tuple((x, y, w, h)) in violate:
            color = (0, 0, 225)
            v += 1
            # Increment high risk or low risk count based on violation
            if tuple((x, y, w, h)) in violate and tuple((x, y, w, h)) not in ignore:
                high_risk_count += 1
                # Add the violated individual to the ignore set
                ignore.add(tuple((x, y, w, h)))
            else:
                low_risk_count += 1
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (w, h), color, 2)

    # Increment total persons count
    total_persons += len(persons)

    # Compute crowd density
    #crowd_frame = frame[crowd_roi[1]:crowd_roi[3], crowd_roi[0]:crowd_roi[2]]
    crowd_frame = frame
    crowd_gray = cv2.cvtColor(crowd_frame, cv2.COLOR_BGR2GRAY)
    crowd_density = cv2.countNonZero(crowd_gray) / (crowd_frame.shape[0] * crowd_frame.shape[1])

    # Check crowd density threshold and increment crowd violation count if exceeded
    if crowd_density > crowd_density_threshold:
        crowd_violation_count += 1

    cv2.namedWindow("Social Distance Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Social Distance Detector", 800, 600)
    cv2.putText(frame, 'Number of Violations: ' + str(v), (20, frame.shape[0] - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 3)

    cv2.putText(frame, 'High Risk: ' + str(high_risk_count), (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 3)
    cv2.putText(frame, 'Low Risk: ' + str(low_risk_count), (20, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,255), 3)
    cv2.putText(frame, 'Total People: ' + str(total_persons), (600, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,255), 3)

    #cv2.putText(frame, "Crowd Density: {:.2f}".format(crowd_density), (600, frame.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

    # Check crowd violation threshold and increment crowd violation count if exceeded

    #v2.putText(frame, "Crowd Violation: " + str(crowd_violation_count), (600, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    cv2.imshow("Social Distance Detector", frame)
    writer.write(frame)

    # Wait for a key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video and destroy the windows
video.release()
writer.release()
cv2.destroyAllWindows()
