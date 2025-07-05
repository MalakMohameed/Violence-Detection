import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.applications.mobilenet_v2 import MobileNetV2

# Define the model architecture (same as used during training)
IMG_SIZE = 128  # Update this based on your input size
ColorChannels = 3

def build_model():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, ColorChannels))
    base_model = MobileNetV2(pooling='avg', include_top=False, input_tensor=input_tensor)
    
    head_model = base_model.output
    head_model = Dense(1, activation="sigmoid")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)
    
    return model

# Build the model
model = build_model()

# Load the saved weights
model.load_weights("ModelWeights.weights.h5")
print("Model architecture loaded and weights applied successfully!")

# Compile the model (only necessary for evaluation, not prediction)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

def preprocess_frame(frame):
    # Resize and normalize the frame before passing it to the model
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame for model prediction
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)[0][0]
        
        # Determine the label and probability
        label = "Violence" if prediction < 0.50 else "No Violence"
        probability = prediction if prediction > 0.50 else (1 - prediction)
        
        # Display the prediction percentage (probability) on the frame
        text = f"{label}: {probability * 100:.2f}%"
        
        # Overlay the text on the top-left corner of the frame
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame with the overlayed prediction text
        cv2.imshow('Frame with Prediction', frame)
        
        # Wait for 'q' key to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example video test
video_path = r'C:\Users\malak\OneDrive\Documents\Image Processing Project\SCVD\SCVD_converted\Test\Normal\t_n011_converted.avi'
analyze_video(video_path)
