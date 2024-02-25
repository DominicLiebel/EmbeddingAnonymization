# Embedding Anonymization

This Python project aims to anonymize embeddings while maintaining high utility.

## Hypothesis: Accuracy loss correlates…
 **<font color="green">...positively</font>** with Privacy e.g. <font color="green">Reconstruction Error</font>
 
 **<font color="red">...negatively</font>** with Utility e.g. <font color="red">Variance Retention</font>, <font color="red">Projection Robustness</font>


## Original CIFAR100 Embeddings
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/1da5bc1e-e0fb-4d9f-b83f-a4976925a5b7">


## Measures of Anonymization
- **hasOverlap:** Simple check to see if anonymized embeddings have any overlap with original embeddings.<br>
  e.g.: `No overlap between original and anonymized embeddings.`
- **Reconstruction Error:** Reconstruction error quantifies how well the anonymized embeddings can reconstruct the original embeddings.<br>
  e.g.: `Reconstruction Error: 4.0027`
- **Mean Relative Difference:** Mean relative differences measure the average percentage change between the original and anonymized embeddings for each image.<br>
  e.g.: `Image 1 -> Mean Relative Difference: 69.68424224853516%`
- **Relative Reconstruction Error:** Relative reconstruction error provides a normalized measure of the reconstruction error by dividing the reconstruction error by the total variance of the original data.
- **Variance Retention:** Variance retention represents the proportion of variance preserved after the anonymization process compared to the original embeddings.
- **Projection Robustness:** Projection robustness assesses the stability of the embeddings under different projection methods, providing insights into the robustness of the anonymization technique.

These metrics collectively offer a nuanced understanding of the anonymization process, catering to different goals and considerations depending on the specific context and requirements.



### Density Based Anonymization (CIFAR100)
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/7020740f-63ee-4231-b50e-49fa3cb3ddd6">


<br>
`Epsilon=1.1, Min Samples=3, Noise Scale=1, Accuracy=94.57%,Reconstruction Error=2.0002`



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
- **experiments.ipynb:** Contains experiments with self-generated embeddings.



# Experimental Results Interpretation (CIFAR10)
<img width="454" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/2b761cf2-fb24-49c4-ba3d-91cfcfaf77c3">


These results are part of an experimental optimization process where various parameters are systematically altered, and the resulting accuracy and reconstruction error are documented. Let's delve into the interpretation:

- **Epsilon**: Varied in the range of 1.0 to 1.5 with increments of 0.1.
- **Min Samples**: Kept constant at 3 throughout all iterations.
- **Noise Scale**: Varied from 1.0 to 1.5 in increments of 0.25.

Now, let's analyze the outcomes:

1. **Accuracy**: Represents the percentage of correctly classified instances. Higher accuracy values are generally preferred.

   - Accuracy ranges from approximately 78.56% to 94.65%.
   - Generally, there seems to be a negative correlation between noise and accuracy.
   - Within a specific epsilon value, an increase in noise scale tends to result in decreased accuracy.

2. **Reconstruction Error**: Indicates how well the reconstructed data aligns with the original data. Higher reconstruction error values are desirable for anonymization.

   - Reconstruction error ranges from approximately 2 to 4.5.
   - The reconstruction error does not exhibit a clear trend with epsilon.

It appears there is a discernible trade-off between accuracy and noise, with higher noise values corresponding to lower accuracy. The relationship with epsilon is not as straightforward. This may be due to an implementation error.



## Getting Started
To get started, follow these steps:
1. Clone the repository: `git clone https://github.com/DominicLiebel/EmbeddingAnonymization.git`
2. Navigate to the project directory: `cd EmbeddingAnonymization`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Provide: `test_cifar10.npz, train_cifar10.npz, test_cifar100.npz, train_cifar100.npz`
5. Configure: `config.yaml`
6. Run the main script: `python main.py`

# Experiments
**The Experiments Jupyter notebook demonstrates several techniques for anonymizing embeddings, evaluating the effectiveness of the anonymization, and visualizing the results.** 

## Libraries Used
- Matplotlib: For data visualization.
- mpl_toolkits.mplot3d: For 3D visualization.
- Torch: PyTorch library for machine learning.
- NumPy: For numerical operations.
- Scikit-learn: For machine learning algorithms such as DBSCAN and PCA.
- Sys: For system-specific parameters and functions.
- Random: For generating random numbers.

## Description
1. **Anonymization Techniques**: The notebook presents various anonymization techniques, including DBSCAN clustering and PCA.
2. **Evaluation Metrics**: It calculates and prints several evaluation metrics such as Reconstruction Error, Relative Reconstruction Error, Variance Retention, and Projection Robustness.
3. **Visualization**: The notebook visualizes both original and anonymized embeddings using scatter plots in 3D space.
4. **Functions**: It includes functions for anonymizing embeddings, evaluating metrics, and visualizing embeddings.

## Note
- This notebook assumes familiarity with concepts such as embeddings, clustering, PCA, and evaluation metrics.
- Some parameters such as DBSCAN `eps` and `min_samples` require tuning for optimal performance depending on the dataset.

## Getting Started: Experiments
To get started with the further experiments:
1. Ensure all necessary libraries are installed.
2. Run the notebook cells sequentially to perform anonymization, evaluation, and visualization.
3. Modify parameters as needed, such as DBSCAN parameters, noise scale, and embedding dimensions.
4. Explore the visualization to understand the effectiveness of the anonymization techniques.
Feel free to explore and modify the code based on your specific requirements.

## Contant Information
You can contant me at dominic.liebel@gmail.com

## First Tries
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/4f288bd4-02eb-4530-af99-8da8cdfbd8c2">
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/cb621191-1809-4a17-8b4c-ca5516c52ca3">
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/7474b35a-0ec8-45df-9bde-4224b04af091">
