# Lab 08 — PyTorch vs. Keras Framework Comparison

**Project:** Lakeshore Bike Share — Same Model, Two Frameworks

## Observations and Recommendation

Our team implemented equivalent PyTorch and Keras neural networks to predict hourly bicycle rentals. Both frameworks used the same prepared dataset, 64/32 hidden-layer architecture, ReLU activations, Adam optimizer, batch size of 32, and maximum training budget of 100 epochs. We compared performance across three random seeds: 42, 43, and 44.

**1. Prediction accuracy:** PyTorch achieved an average test mean absolute error (MAE) of **3.0478 rentals**, compared with **3.1927 rentals** for Keras. PyTorch performed better on seeds 42 and 43, while Keras performed slightly better on seed 44. Both frameworks demonstrated comparable predictive performance, with PyTorch achieving a modest advantage across the three runs.

**2. Training efficiency:** PyTorch required approximately **9.58 seconds** per run, compared with **26.03 seconds** for Keras. PyTorch was approximately **2.7 times faster** in our measured environment. For a weekly automated prediction job, this provides greater flexibility for retraining and troubleshooting. These results are specific to our experimental setup.

**3. Implementation and maintainability:** PyTorch provides explicit control over the training loop, including gradient calculations, optimizer updates, and validation procedures. This flexibility supports detailed debugging and future model customization. Keras simplifies training through `compile()`, `fit()`, `EarlyStopping`, and `ModelCheckpoint`, reducing the amount of custom training code.

## Recommendation

I recommend **PyTorch** for Lakeshore Bike Share. Our small team already has a functional PyTorch pipeline with reproducible preprocessing, automated tests, and established reporting. Its measured training efficiency supports the planned weekly job, while its flexibility would be valuable if the team later integrates pretrained models from Hugging Face. Keras achieved better test MAE on all three seeds.

PyTorch trained approximately 2.81 times faster.
PyTorch remains a reasonable choice for an existing weekly production pipeline and potential Hugging Face integration.
Keras may be preferable if predictive accuracy becomes the primary selection criterion.

## Trade-offs

PyTorch requires more explicit training and checkpoint-management code, increasing maintenance responsibilities. Keras provides a simpler high-level workflow that may be preferable for straightforward neural networks.

## Condition for Reconsideration

I would recommend reconsidering Keras if the project remains limited to simple tabular prediction, Hugging Face integration is no longer anticipated, and reducing development and maintenance complexity becomes more important than the observed runtime difference.

Overall, both frameworks satisfy the prediction objective, but PyTorch provides the stronger balance of measured performance, operational efficiency, and future flexibility for this project.