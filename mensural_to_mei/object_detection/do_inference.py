import numpy as np


def do_inference(session, staff_image: np.ndarray) -> np.ndarray: # type: ignore
    """
    Performs inference on a given staff image using an ONNX model.

    This function takes an ONNX InferenceSession and a staff image as 
    input, preprocesses the image, and performs inference using the 
    ONNX model. The function returns the output of the inference.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        The ONNX InferenceSession to use for performing inference.
    staff_image : np.ndarray
        The staff image on which to perform inference. The image is a 
        numpy array of shape (H, W, 3), where H is the height of the 
        image, W is the width of the image, and 3 represents the three 
        color channels.

    Returns
    -------
    np.ndarray
        The output of the inference. The shape and contents of the output 
        depend on the ONNX model used for inference.
    """
    model_inputs = session.get_inputs()

    # Preprocess the staff image
    # Convert the image data type to float32 for compatibility with the model
    staff_image = staff_image.astype(np.float32)

    # Normalize the image pixel values to the range [0, 1] by dividing by 255
    staff_image = np.array(staff_image) / 255.0

    # Transpose the image from (H, W, C) to (C, H, W)
    # The model expects the color channels to be the first dimension
    staff_image = np.transpose(staff_image, [2, 0, 1])

    # Add an extra dimension to the image array to match the input shape expected by the model
    # The model expects a batch of images as input
    input_data = np.expand_dims(staff_image, axis=0)

    output = session.run(None, {model_inputs[0].name: input_data})
    
    return output