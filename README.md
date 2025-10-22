# Live classification of facial expression

**This program captures the face from your webcam and show a matching emoji live**

- Uses a webcam to capture live video frames via OpenCV.

- Loads a pretrained Vision Transformer (ViT) model from Hugging Face [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)

- Loads emoji images (PNG files) for each emotion and resizes them to a fixed size.

- Runs real-time emotion detection:

- Sends the current frame to the model in a background thread (so the video doesn’t freeze).

- The model predicts the most likely emotion. The matching emoji is selected.

- Overlays the emoji in the top-left corner of the live video to visually show the detected emotion.

- Stops gracefully when you press q.