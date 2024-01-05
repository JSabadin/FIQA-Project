# FIQA Perturbations

We introduce a novel Face Image Quality Assessment (FIQA) methodology that focuses on the stability of face embeddings in the embedding space, evaluated through controlled perturbations. Our approach is predicated on the hypothesis that the quality of facial images is directly correlated with the stability of their embeddings; lower-quality images tend to exhibit greater variability upon perturbation.


## Documentation

For detailed information, see our [seminar paper](docs/SBSSeminar.pdf).

## Model Weights

This project uses weights for the ArcFace, AdaFace, and TransFace models. Please download the weights and place them in their respective directories:

- ArcFace weights should be placed in `face_recognition/arcface`
- AdaFace weights should be placed in `face_recognition/adaface`
- TransFace weights should be placed in `face_recognition/transface`

### Important Note

After adding the weights, ensure that the paths to these weight files are correctly set in the `utils` and `main` files. This is crucial for the proper functioning of the face recognition models.

## Experimental Work

To run the experiments, navigate to the `/experimental_work` folder and run the `main.py` script.
