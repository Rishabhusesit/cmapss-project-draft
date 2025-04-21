import tensorflow as tf

def save_model(model, model_path="ml_model/models/lstm_converted.keras"):
    """
    Save the trained model to the specified path.
    
    Args:
        model: The trained TensorFlow model
        model_path: Path where the model should be saved
    """
    try:
        model.save(model_path)
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    # You can load your model here and then save it
    # Example:
    # model = tf.keras.models.load_model("path_to_your_model")
    # save_model(model)
    pass 