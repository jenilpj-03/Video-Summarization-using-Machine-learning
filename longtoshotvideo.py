import cv2


def main():
    input_video_path = '2.mp4'
    output_video_path = 'final.mp4'

    try:
        # Open the input video file
        video = cv2.VideoCapture(input_video_path)
        if not video.isOpened():
            raise Exception("Error: Could not open video file.")

        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Background subtractor
        fgbg = cv2.createBackgroundSubtractorMOG2()

        # Video writer
        writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

        # Variables for frame counting
        total_frames = 0
        detected_frames = 0
        non_detected_frames = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Apply background subtraction
            fgmask = fgbg.apply(frame)

            # Count white pixels in the foreground mask
            white_pixels = cv2.countNonZero(fgmask)

            # Adjust this threshold based on your video content
            if white_pixels > 2000:
                writer.write(frame)
                detected_frames += 1
            else:
                non_detected_frames += 1

            total_frames += 1

            # Display the original frame
            cv2.imshow('Original Frame', frame)

            # Display the foreground mask (optional)
            cv2.imshow('Foreground Mask', fgmask)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print results
        print("Total frames: ", total_frames)
        print("Detected frames: ", detected_frames)
        print("Non-detected frames: ", non_detected_frames)

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        # Release resources
        if 'video' in locals() and video.isOpened():
            video.release()
        if 'writer' in locals():
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
