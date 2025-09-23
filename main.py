import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

# load the model architecture
model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, 7)

# Load weights
state_dict = torch.load("saved_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)

# Set eval mode
model.eval()



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
])



# Emotion labels
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('test_video8.mp4')

while True:
    _, frame = cap.read()
    # frame = cv2.imread('test_img1.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]  # this keeps RGB crop (3 channels)
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  
        roi_tensor = transform(roi_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(roi_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = classes[pred]

        # Draw face & emotion
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36,255,12), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


