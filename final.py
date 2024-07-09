from transformers import pipeline
from IPython.display import Image, display
import IPython.display
import os
import cv2


def display_image(image_path):
    display(Image(filename=image_path))


def get_image_caption(image_path, model_name="Salesforce/blip-image-captioning-large"):
    try:
        caption_pipeline = pipeline("image-to-text", model=model_name)
        output = caption_pipeline(image_path)
        generated_text = output[0]['generated_text']
        return generated_text
    except Exception as e:
        print(f"Error getting caption: {str(e)}")
        return None


def get_video_keyframes(video_path, model_name="Salesforce/blip-image-captioning-large"):
    try:
        keyframes_folder = "keyframes"
        os.makedirs(keyframes_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        keyframes = []
        frame_interval = 30
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                keyframe_path = os.path.join(
                    keyframes_folder, f"keyframe_{frame_count}.jpg")
                cv2.imwrite(keyframe_path, frame)

                try:
                    caption = get_image_caption(keyframe_path, model_name)
                    keyframes.append(
                        {'frame_path': keyframe_path, 'caption': caption})
                except Exception as e:
                    print(
                        f"Error getting caption for {keyframe_path}: {str(e)}")

            frame_count += 1

        cap.release()

        return keyframes

    except Exception as e:
        print(f"An error occurred: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return None


def main():
    input_video_path = '1.mp4'
    output_video_path = 'final.mp4'

    try:
        video = cv2.VideoCapture(input_video_path)
        if not video.isOpened():
            raise Exception("Error: Could not open video file.")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fgbg = cv2.createBackgroundSubtractorMOG2()
        writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

        total_frames = 0
        detected_frames = 0
        non_detected_frames = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            fgmask = fgbg.apply(frame)
            white_pixels = cv2.countNonZero(fgmask)

            if white_pixels > 2000:
                writer.write(frame)
                detected_frames += 1
            else:
                non_detected_frames += 1

            total_frames += 1

            cv2.imshow('Original Frame', frame)
            cv2.imshow('Foreground Mask', fgmask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Total frames: ", total_frames)
        print("Detected frames: ", detected_frames)
        print("Non-detected frames: ", non_detected_frames)

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        if 'video' in locals() and video.isOpened():
            video.release()
        if 'writer' in locals():
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    uploaded_video_path = 'final.mp4'
    keyframes = get_video_keyframes(uploaded_video_path)

    if keyframes:
        for frame in keyframes:
            print(f"Caption for {frame['frame_path']}:\n{frame['caption']}\n")
            display_image(frame['frame_path'])
    else:
        print("Failed to generate captions for the video.")
