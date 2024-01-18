# Embedding Anonymization

This Python project aims to anonymize embeddings while maintaining high accuracy with a high reconstruction error.

## Original CIFAR10 Embeddings
![Before Anonymization](https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/1da5bc1e-e0fb-4d9f-b83f-a4976925a5b7)

## First Tries
### The Egg
![The Egg](https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/4f288bd4-02eb-4530-af99-8da8cdfbd8c2)

### Modern Art
![Modern Art](https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/cb621191-1809-4a17-8b4c-ca5516c52ca3)

## Measures of Anonymization
- **hasOverlap:** Simple check to see if anonymized embeddings have any overlap with original embeddings.
- **Reconstruction Error:** `torch.mean((normalized_test_embeddings - test_embeddings_anonymized)**2).item()`

## Different Anonymization Techniques Employed
- `anonymize_embeddings_random(embeddings, noise_factor=0.1)`
- `anonymize_embeddings_laplace(embeddings, epsilon=0.1, device="cpu")`
- `anonymize_embeddings_dp(embeddings, epsilon=0.1, device="cpu")`
- `anonymize_embeddings_permutation(embeddings)`
- `anonymize_embeddings_hashing(embeddings, salt="secret_salt")`
- `anonymize_embeddings_pca(embeddings, n_components=2)`
- `anonymize_embeddings_density_based(embeddings, eps=0.5, min_samples=5, noise_scale=0.01, device="cpu")`

## Project Structure
The project is structured as follows:
- **main.py:** The main script to run the anonymization process.
- **anonymization.py:** Contains different functions for anonymizing embeddings using various techniques.
- **model.py:** Defines the PyTorch model used in the project.
- **train_util.py:** Provides utility functions for training and evaluating the model.
- **evaluation.py:** Contains a function to find the best parameters for anonymization.
- **visualization.py:** Provides a visualization function.
- **data_loader.py:** Contains a function to load the data.
- 

## Getting Started
To get started, follow these steps:
1. Clone the repository: `git clone https://github.com/DominicLiebel/EmbeddingAnonymization.git`
2. Navigate to the project directory: `cd EmbeddingAnonymization`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the main script: `python main.py`

Feel free to explore and modify the code based on your specific requirements.
