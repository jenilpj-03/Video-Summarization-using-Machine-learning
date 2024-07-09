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
        # Display the uploaded image
        display_image(image_path)

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


if __name__ == "__main__":
    # Replace 'your_uploaded_image.jpg' with the actual file name
    uploaded_image_path = 'keyframes\keyframe_270.jpg'

    # Get and print the generated text caption
    generated_text = get_image_caption(uploaded_image_path)

    if generated_text:
        print("Generated Caption:")
        print(generated_text)
    else:
        print("Failed to generate a caption for the image.")
