import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = './weightsv1/yolow-l.onnx'
ort_session = ort.InferenceSession(model_path)

# Define a function to preprocess the input image
def preprocess(image):
    # Resize the image
    input_image = cv2.resize(image, (640, 640))
    # Normalize the image
    input_image = input_image.astype(np.float32) / 255.0
    # Convert to CHW format
    input_image = input_image.transpose(2, 0, 1)
    # Add batch dimension
    input_tensor = np.expand_dims(input_image, axis=0)
    return input_tensor

# Define a function to post-process the model output
def postprocess(outputs, confidence_threshold=0.5):
    boxes, scores, class_ids = [], [], []
    for output in outputs:
        print(output)
        for detection in output[0]:  # batch size 1
            print(detection)
            score = detection[4]
            if score > confidence_threshold:
                x, y, w, h = detection[0:4]
                class_id = int(detection[5])
                boxes.append([x, y, w, h])
                scores.append(score)
                class_ids.append(class_id)
    return boxes, scores, class_ids

# Open the input video
input_video_path = 'test1.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_video_path = 'testoutworld.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)

    # Perform inference
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
    print(outputs)
    # Post-process the outputs
    boxes, scores, class_ids = postprocess(outputs)
    print(scores)
    # Draw bounding boxes on the frame
    for (x, y, w, h), score, class_id in zip(boxes, scores, class_ids):
        # Convert from normalized coordinates to image coordinates
        x1, y1 = int(x * frame_width), int(y * frame_height)
        x2, y2 = int((x + w) * frame_width), int((y + h) * frame_height)
        
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label and score
        label = f'Class {class_id}: {score:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
