# segmentation modules
    - site_packages/segmentation_models/models/unet, line 164, add this line
    ```python
    input_tensor=None,
    ```
    - site_packages/segmentation_models/models/unet, line 222, add this line
    ```python
    input_tensor=input_tensor
    ```
# keras_application
    - site_packages/keras_applications/vgg16.py, line 104, change to 
    ```python
    -    if not backend.is_keras_tensor(input_tensor):
    +    if backend.backend() == 'tensorflow':
    +        from tensorflow.python.keras.backend import is_keras_tensor
    +    else:
    +        is_keras_tensor = backend.is_keras_tensor
    +    if not is_keras_tensor(input_tensor):
    ```