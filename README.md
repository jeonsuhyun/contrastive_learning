# contrastive_learning

## Process: Contrastive Learning → Motion Planning (Based on This Repository)

This repository implements a pipeline that leverages contrastive learning to improve motion planning for dual-arm robots. The process is as follows:

1. **Data Collection & Preprocessing**
   - Generate or collect a large dataset of robot joint configurations and associated task parameters (e.g., object poses, grasp conditions).
   - Preprocess the data to extract relevant features, such as separating joint angles for each arm and normalizing the data.
   - Example: See `scripts/data_preprocessing.py` for how joint data is extracted and visualized.

2. **Contrastive Representation Learning**
   - Train a neural network encoder using a contrastive loss to map high-dimensional joint configurations (and possibly task conditions) into a lower-dimensional latent space.
   - The contrastive loss encourages configurations that are similar in task space (e.g., similar end-effector poses or successful grasps) to be close in the latent space, and dissimilar ones to be far apart.
   - This step is typically implemented in the `contrastiveik` module (see `contrastiveik/modules/network.py`).

3. **Latent Space Analysis & Clustering**
   - Analyze the learned latent space using dimensionality reduction (e.g., UMAP, t-SNE) and clustering (e.g., KMeans, HDBSCAN) to verify that meaningful structure has been captured.
   - This helps in understanding how well the latent space separates different types of configurations and can inform roadmap construction.
   - Example: See `example/label_cluster_dataset.py` and `scripts/data_preprocessing.py` for clustering and visualization.

4. **Roadmap Construction in Latent Space**
   - Build a roadmap (graph) where nodes are encoded configurations in the latent space, and edges represent feasible transitions (e.g., collision-free, kinematically valid).
   - The roadmap can be constructed using nearest neighbors or sampling-based methods.
   - See `ljcmp/planning/precomputed_roadmap.py` for roadmap utilities.

5. **Motion Planning Using Latent Representations**
   - For a given planning problem (start and goal), encode the start and goal configurations into the latent space.
   - Use a motion planning algorithm (e.g., Constrained BiRRT, Latent BiRRT) to search for a path in the latent space, leveraging the learned representation to guide sampling and connection.
   - The planner can use the latent space to bias sampling or to define distance metrics that reflect task-relevant similarity.
   - See `scripts/benchmark_sh_metric.py` for benchmarking and planning code.

6. **Path Decoding and Post-processing**
   - Map the planned path in latent space back to the original joint configuration space, either by using the decoder (if available) or by retrieving the original configurations.
   - Optionally, apply time parameterization and smoothing to the path for execution.
   - See `ljcmp/utils/time_parameterization.py` for trajectory timing.

7. **Execution and Evaluation**
   - Execute the planned trajectory in simulation or on the real robot.
   - Evaluate the performance using metrics such as path length, planning time, and success rate.
   - See the benchmarking scripts for evaluation routines.

---

**Key Scripts and Modules:**
- `scripts/data_preprocessing.py`: Data extraction and visualization.
- `contrastiveik/modules/network.py`: Contrastive learning model.
- `example/label_cluster_dataset.py`: Clustering and latent space analysis.
- `ljcmp/planning/precomputed_roadmap.py`: Roadmap construction.
- `ljcmp/planning/constrained_bi_rrt_latent_jump.py`: Latent-space motion planning.
- `scripts/benchmark_sh_metric.py`: Benchmarking and evaluation.

---

**Summary Diagram:**

