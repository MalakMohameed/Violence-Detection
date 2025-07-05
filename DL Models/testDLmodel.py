import torch
import torch.nn as nn
from torchvision import models, transforms
model = models.resnet18(pretrained=True)
from PIL import Image
import cv2  # OpenCV for video processing
#Load the model from the respective path where it waas saved after the training was done
# Adjust the output layer for 2 classes (Normal, Violence)
model.fc = nn.Linear(model.fc.in_features, 2)

modeel = torch.load(r'C:\Users\malak\OneDrive\Documents\Image Processing Project\violence_detection_model.pth')
# Load the model weights (if you have a saved model checkpoint, load it here)
model.load_state_dict(torch.load(r'C:\Users\malak\OneDrive\Documents\Image Processing Project\violence_detection_model.pth'))

# Set the model to evaluation mode
model.eval()

# Define the image transform to match ResNet input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to process a video file and predict frame by frame
def process_video(video_path):
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Loop through each frame of the video
    frame_num = 0
    while True:
        ret, frame = cap.read()
        
        # If no more frames are available, exit the loop
        if not ret:
            break

        frame_num += 1
        # Convert frame (BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to PIL Image
        img = Image.fromarray(frame_rgb)
        
        # Apply the transformations
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Perform inference (predict the class)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            pred_prob = probs[0][pred_class].item()

        # Convert the predicted class index to a readable label
        class_names = ['Normal', 'Violence']
        predicted_label = class_names[pred_class.item()]

        # Display the prediction percentage (probability) on the frame
        text = f"{predicted_label}: {pred_prob * 100:.2f}%"

        # Overlay the text on the top-left corner of the frame
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame with the overlaid prediction text
        cv2.imshow('Frame with Prediction', frame)

        # Wait for 'q' key to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Path to the video file
video_path = r'C:\Users\malak\OneDrive\Documents\Image Processing Project\SCVD\SCVD_converted\Test\Normal\t_n001_converted.avi'
# Process the video
process_video(video_path)
