import os
import cv2
import IPython.display
from IPython.display import Image, display
from transformers import pipeline


def display_image(image_path):
    """
    Display the image using IPython.display.
    """
    display(Image(filename=image_path))


def get_image_caption(image_path, model_name="Salesforce/blip-image-captioning-base"):
    """
    Use a pipeline for image-to-text conversion and get the image caption.

    Parameters:
    - image_path (str): Path to the uploaded image.
    - model_name (str): Name or path of the image captioning model.

    Returns:
    - str: Generated text caption for the image.
    """
    try:
        # Use a pipeline for image-to-text conversion
        caption_pipeline = pipeline("image-to-text", model=model_name)

        # Get completion or caption for the image
        output = caption_pipeline(image_path)
        generated_text = output[0]['generated_text']

        return generated_text

    except Exception as e:
        # Handle exceptions gracefully
        print(f"An error occurred: {e}")
        return None


def get_video_keyframes(video_path, model_name="Salesforce/blip-image-captioning-base"):
    """
    Extract key frames from a video and get captions for each key frame.

    Parameters:
    - video_path (str): Path to the uploaded video.
    - model_name (str): Name or path of the image captioning model.

    Returns:
    - list: List of dictionaries containing 'frame_path' and 'caption' for each key frame.
    """
    try:
        # Create a folder for key frames
        keyframes_folder = "keyframes"
        os.makedirs(keyframes_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Extract key frames
        keyframes = []
        # Extract a key frame every 30 frames (adjust as needed)
        frame_interval = 30
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Save the key frame in the keyframes folder
                keyframe_path = os.path.join(
                    keyframes_folder, f"keyframe_{frame_count}.jpg")
                cv2.imwrite(keyframe_path, frame)

                # Get caption for the key frame
                caption = get_image_caption(keyframe_path, model_name)

                # Append the result to the list
                keyframes.append(
                    {'frame_path': keyframe_path, 'caption': caption})

            frame_count += 1

        cap.release()

        return keyframes

    except Exception as e:
        # Handle exceptions gracefully
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # Replace 'your_uploaded_video.mp4' with the actual file name
    uploaded_video_path = 'final.mp4'

    # Get key frames and captions from the video
    keyframes = get_video_keyframes(uploaded_video_path)

    if keyframes:
        for frame in keyframes:
            print(f"Caption for {frame['frame_path']}:\n{frame['caption']}\n")
    else:
        print("Failed to generate captions for the video.")
