from transformers import AutoImageProcessor, AutoModelForImageClassification # face emotions classification
import torch
import cv2
import urllib.request
import numpy as np
import threading
# we need threading to run inference in a separate thread, that will allow us to not block the main camera loop

# load the model from HuggingFace for face emotions classification
# the architecture used is the Vision transformer (ViT)
model_name = "trpakov/vit-face-expression"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, use_safetensors=False)
# use_safetensors=False beacause they raise the error with this model

# load emojis, set their size
em_height = 150
em_width = 150
paths = {
    "angry": "",
    "disgust": "",
    "fear": "",
    "happy": "",
    "neutral": "",
    "sad": "",
    "surprise": "",
}
emojis = []
for p in paths.values():
    req = urllib.request.urlopen(p)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(src=img, dsize=(em_width, em_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    emojis.append(img)


# initialize the camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


def run_inference(frame):
    global predicted_class_idx, update_ability
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad(): # no need to compute gradients for inference
        logits = model(**inputs).logits # logits is the raw output
    predicted_class_idx = logits.argmax(-1).item() # item to get the value from the tensor
    
    update_ability = True
    # that shows that the inference is done


def camera_loop():
    global predicted_class_idx, update_ability
    predicted_class_idx = 4 # initial class index: neutral
    update_ability = True
    
    print("press 'q' to quit")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        # Write the frame to the output file
        out.write(frame)
        
        frame[0:em_width, 0:em_height] = emojis[predicted_class_idx] #for top-left corner
        # Display the captured frame with emoji
        cv2.imshow('Camera', frame)
        
        if update_ability: # it will be True when the previous inference is done
            update_ability = False # block the ability to update the emoji until the inference is done
            threading.Thread(target=run_inference, args=(frame.copy(),)).start() # run inference in a separate thread

        # exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    out.release()
    cv2.destroyAllWindows()

camera_loop()