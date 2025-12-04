from itertools import combinations
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# -----------------------------
# 1. MODEL/DATASET Accuracy Table Data
# -----------------------------
distilled_data = {
    "ConvMixer": {
        "Y": {
            "Imagenette": 90.06,
            "CIFAR10": 94.78,
            "CIFAR100": 80.50,
            "HAM10k": 81.37,
            "Dogs": 63.10,
            "Indoor67": 60.75, 
            "MiniPlaces": 61.86, 
        },
        "N": {
            "Imagenette": 89.25,
            "CIFAR10": 94.52,
            "CIFAR100": 77.58,
            "HAM10k": 80.64,
            "Dogs": 54.81,
            "Indoor67": 58.36,
            "MiniPlaces": 57.82,
        },
    },

    "EfficientNet": {
        "Y": {
            "Imagenette": 88.05,
            "CIFAR10": 95.08,
            "CIFAR100": 78.96,
            "HAM10k": 81.97,
            "Dogs": 64.31,
            "Indoor67": 61.34,
            "MiniPlaces": 62.15,
        },
        "N": {
            "Imagenette": 88.82,
            "CIFAR10": 95.11,
            "CIFAR100": 78.79,
            "HAM10k": 79.44,
            "Dogs": 63.78,
            "Indoor67": 59.70,
            "MiniPlaces": 61.14,
        },
    },

    "MobileNet": {
        "Y": {
            "Imagenette": 86.37,
            "CIFAR10": 94.47,
            "CIFAR100": 73.56,
            "HAM10k": 79.17,
            "Dogs": 56.15,
            "Indoor67": 51.49,
            "MiniPlaces": 55.39,
        },
        "N": {
            "Imagenette": 86.24,
            "CIFAR10": 94.17,
            "CIFAR100": 75.74,
            "HAM10k": 78.78,
            "Dogs": 57.53,
            "Indoor67": 50.60,
            "MiniPlaces": 57.39,
        },
    },

    "ShuffleNet": {
        "Y": {
            "Imagenette": 84.15,
            "CIFAR10": 92.50,
            "CIFAR100": 76.18,
            "HAM10k": 79.97,
            "Dogs": 57.90,
            "Indoor67": 55.22,
            "MiniPlaces": 57.45,
        },
        "N": {
            "Imagenette": 84.00,
            "CIFAR10": 92.15,
            "CIFAR100": 75.80,
            "HAM10k": 78.64,
            "Dogs": 55.49,
            "Indoor67": 53.81,
            "MiniPlaces": 57.03,
        },
    },

    "GhostNet": {
        "Y": {
            "Imagenette": 86.60,
            "CIFAR10": 93.32,
            "CIFAR100": 73.92,
            "HAM10k": 80.24,
            "Dogs": 51.46,
            "Indoor67": 40.90,
            "MiniPlaces": 57.81,
        },
        "N": {
            "Imagenette": 86.29,
            "CIFAR10": 93.56,
            "CIFAR100": 73.52,
            "HAM10k": 76.31,
            "Dogs": 55.22,
            "Indoor67": 41.34,
            "MiniPlaces": 57.73,
        },
    },

    "TinyNet": {
        "Y": {
            "Imagenette": 85.27,
            "CIFAR10": 95.42,
            "CIFAR100": 78.57,
            "HAM10k": 78.11,
            "Dogs": 29.83,
            "Indoor67": 55.22,
            "MiniPlaces": 61.18,
        },
        "N": {
            "Imagenette": 85.30,
            "CIFAR10": 94.90,
            "CIFAR100": 78.21,
            "HAM10k": 74.38,
            "Dogs": 29.59,
            "Indoor67": 50.67,
            "MiniPlaces": 59.06,
        },
    },

    "MobileOne": {
        "Y": {
            "Imagenette": 83.21,
            "CIFAR10": 93.21,
            "CIFAR100": 74.60,
            "HAM10k": 79.11,
            "Dogs": 49.59,
            "Indoor67": 42.09,
            "MiniPlaces": 56.48,
        },
        "N": {
            "Imagenette": 81.40,
            "CIFAR10": 93.34,
            "CIFAR100": 74.55,
            "HAM10k": 79.11,
            "Dogs": 46.74,
            "Indoor67": 43.51,
            "MiniPlaces": 56.33,
        },
    },

    "FBNet": {
        "Y": {
            "Imagenette": 81.83,
            "CIFAR10": 94.35,
            "CIFAR100": 75.67,
            "HAM10k": 79.77,
            "Dogs": 53.40,
            "Indoor67": 44.18,
            "MiniPlaces": 58.83,
        },
        "N": {
            "Imagenette": 83.11,
            "CIFAR10": 94.36,
            "CIFAR100": 75.62,
            "HAM10k": 78.44,
            "Dogs": 51.41,
            "Indoor67": 48.73,
            "MiniPlaces": 57.61,
        },
    },

    "ConvNeXt": {
        "Y": {
            "Imagenette": 77.40,
            "CIFAR10": 92.24,
            "CIFAR100": 69.60,
            "HAM10k": 78.38,
            "Dogs": 30.42,
            "Indoor67": 32.76,
            "MiniPlaces": 53.24,
        },
        "N": {
            "Imagenette": 76.05,
            "CIFAR10": 91.97,
            "CIFAR100": 68.91,
            "HAM10k": 78.58,
            "Dogs": 30.96,
            "Indoor67": 32.76,
            "MiniPlaces": 51.01,
        },
    },

    "MobileViT": {
        "Y": {
            "Imagenette": 87.24,
            "CIFAR10": 94.97,
            "CIFAR100": 77.04,
            "HAM10k": 74.32,
            "Dogs": 63.78,
            "Indoor67": 55.67,
            "MiniPlaces": 61.74,
        },
        "N": {
            "Imagenette": 86.55,
            "CIFAR10": 95.19,
            "CIFAR100": 77.25,
            "HAM10k": 77.38,
            "Dogs": 59.18,
            "Indoor67": 54.78,
            "MiniPlaces": 59.74,
        },
    },

    "StartNet": {
        "Y": {
            "Imagenette": 84.87,
            "CIFAR10": 95.13,
            "CIFAR100": 78.75,
            "HAM10k": 79.57,
            "Dogs": 54.76,
            "Indoor67": 54.10,
            "MiniPlaces": 59.61,
        },
        "N": {
            "Imagenette": 84.31,
            "CIFAR10": 94.88,
            "CIFAR100": 77.59,
            "HAM10k": 79.04,
            "Dogs": 48.91,
            "Indoor67": 51.79,
            "MiniPlaces": 55.45,
        },
    },
}

# -----------------------------
# 2. xScore Computation Function for 7 datasets
# -----------------------------
def compute_xscore_pair_and_summary(distilled_data, mode="N", lam=0.5):
    # Extract into list-of-dict rows similar to the old data format
    rows = []
    for model_name, modes in distilled_data.items():
        if mode not in modes:
            raise KeyError(f"Model '{model_name}' missing mode '{mode}' in distilled_data.")

        row = {"Model": model_name}
        row.update(modes[mode])
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows).set_index("Model")

    # Normalize (min-max per dataset)
    df_norm = (df - df.min()) / (df.max() - df.min())

    # Compute G_i and V_i
    G_i = df_norm.mean(axis=1)
    V_i = df_norm.var(axis=1)

    # Compute xScore
    xScore = G_i - lam * V_i

    # Combined table
    df_combined = df.copy()
    for col in df.columns:
        df_combined[col] = (
            df[col].round(3).astype(str) + 
            " (" + df_norm[col].round(3).astype(str) + ")"
        )

    # Summary table
    xscores = pd.DataFrame(
        {
            "G_i": G_i.round(3),
            "V_i": V_i.round(3),
            "xScore": xScore.round(3),
        }
    ).sort_values("xScore", ascending=False)

    return df_combined, xscores

# -----------------------------
# 3. Get imagenette and xscore x-y 
# -----------------------------
    
def generate_imagenette_xscore(distilled_data, xscores, dataset_name="Imagenette"):
    """
    Builds a perf{} dictionary containing:
      - Model names
      - Imagenette accuracy (non-distilled)
      - xScore values from xscores

    distilled_data: dict in your 2-level format
    xscores: output of compute_xscore_pair_and_summary(..., mode="N")
    """

    models = []
    imagenette_acc = []
    xscores_mean = []

    for model_name in xscores.index:
        models.append(model_name)

        # pull accuracy from distilled_data
        imagenette_value = distilled_data[model_name]["N"][dataset_name]
        imagenette_acc.append(imagenette_value)

        # xscore from summary
        xscores_mean.append(xscores.loc[model_name, "xScore"])

    imagenette_xscores = {
        "Model": models,
        dataset_name: imagenette_acc,
        "xScore": xscores_mean,
    }

    return imagenette_xscores


def plot_correlation_xscore_imagenette(imagenette_xscores, dataset_col="Imagenette", xscore_col="xScore"):
    """
    Scatter plot showing correlation between xScore and imagenette accuracy.
    x-axis: xScore (sorted low → high)
    y-axis: dataset accuracy (e.g., Imagenette-160)
    Weighted linear regression is used to reduce outlier influence.
    """
    df = pd.DataFrame(imagenette_xscores)
    df_sorted = df.sort_values(by=xscore_col, ascending=True)

    # Compute Pearson correlation
    r, p = pearsonr(df_sorted[xscore_col], df_sorted[dataset_col])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter points
    ax.scatter(df_sorted[xscore_col], df_sorted[dataset_col], color="green", s=80, zorder=3)

    # Connect points with dashed line
    ax.plot(df_sorted[xscore_col], df_sorted[dataset_col], color="darkgray", linestyle="--", linewidth=1, zorder=2)

    # Weighted Linear Regression (downplay outliers)
    X = df_sorted[xscore_col].values.reshape(-1, 1)
    y = df_sorted[dataset_col].values
    weights = np.ones_like(y)
    weights[df_sorted["Model"].isin(["GhostNet", "TinyNet"])] = 0.1  # Reduce weight of outliers

    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    y_fit = model.predict(X)

    # Plot weighted regression line
    ax.plot(df_sorted[xscore_col], y_fit, color="blue", linewidth=1, label="Weighted linear fit", zorder=4, alpha=0.6)

    # Annotate model names (always on top)
    for _, row in df_sorted.iterrows():
        ax.annotate(
            row["Model"],
            (row[xscore_col], row[dataset_col] + 0.3),
            fontsize=8,
            ha="center",
            clip_on=False  # ensures text is drawn above all lines
        )

    # Plot details
    ax.set_title(f"{dataset_col} Accuracy vs xScore", fontsize=13)
    ax.set_xlabel("xScore", fontsize=11)
    ax.set_ylabel(f"{dataset_col} Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.9)
    ax.legend()
    plt.tight_layout()

    # Save figure to PDF
    plt.savefig("xscore_imagenette_corr.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    print(f"Pearson correlation: r = {r:.3f}, p = {p:.3f}")

    return df_sorted
    
# -----------------------------
# 4. Find the best 4 datasets
# -----------------------------
    
def find_best_dataset_subset(distilled_data, perf,  mode="N",subset_size=4  ):

    # Build a DataFrame just like in compute_xscore_pair_and_summary()
    rows = []
    for model_name, modes in distilled_data.items():
        if mode not in modes:
            raise KeyError(
                f"Model {model_name} does not contain mode '{mode}'."
            )

        row = {"Model": model_name}
        row.update(modes[mode])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # Extract target xScore aligned to df index
    y_series = pd.Series(perf["xScore"], index=perf["Model"]).loc[df.index]
    y = y_series.values.reshape(-1, 1)

    datasets = df.columns.tolist()
    best_r2 = -np.inf
    best_subset = None
    best_coeff = None
    best_intercept = None

    # Exhaustive search of combinations
    for subset in combinations(datasets, subset_size):
        X = df[list(subset)].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        if r2 > best_r2:
            best_r2 = r2
            best_subset = subset
            best_coeff = model.coef_.flatten()
            best_intercept = float(model.intercept_.item())

    return best_subset, best_r2, best_coeff, best_intercept
    
def compute_xscore_from_subset(distilled_data, best_subset, coeff=None, intercept=None, mode="N"):
    # If mode is a numpy array, extract the string value
    if not isinstance(mode, str):
        try:
            mode = mode.item()
        except:
            raise TypeError(f"Mode must be a string, but got: {type(mode)}")

    # CASE 1: Nested dict (original format)
    if isinstance(distilled_data, dict):
        rows = []
        for model_name, modes in distilled_data.items():
            if not isinstance(modes, dict):
                raise ValueError(
                    f"Expected distilled_data['{model_name}'] to be a dict."
                )

            row = {"Model": model_name}

            # Same protection inside the loop just in case
            row.update(modes[mode])
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")

    # CASE 2: Already a DataFrame
    elif isinstance(distilled_data, pd.DataFrame):
        df = distilled_data.copy()

        if "Model" in df.columns:
            df = df.set_index("Model")

    else:
        raise TypeError(
            "distilled_data must be either a dict (nested) or a pandas DataFrame."
        )

    # Select the dataset subset
    X = df[list(best_subset)].values

    # Defaults
    if coeff is None:
        coeff = np.ones(len(best_subset))
    if intercept is None:
        intercept = 0.0

    # Prediction
    xscore_pred = X @ coeff + intercept

    return pd.DataFrame({
        "Model": df.index,
        "Predicted_xScore": xscore_pred
    }).reset_index(drop=True)

def xscore7_vs_xscore4(summary_table, df_xscore):
    # Ensure Model is a column in both tables for merging
    if summary_table.index.name == "Model" or "Model" not in summary_table.columns:
        summary_table = summary_table.reset_index()

    if "Model" not in df_xscore.columns:
        df_xscore = df_xscore.reset_index()

    # Merge on Model
    df_summary_expanded = summary_table.merge(
        df_xscore, on="Model", how="left"
    )

    # Rename for clarity
    df_summary_expanded = df_summary_expanded.rename(
        columns={"xScore": "xScore_7datasets", "Predicted_xScore": "xScore_4datasets"}
    )

    # Set Model as index
    df_summary_expanded = df_summary_expanded.set_index("Model")

    # Optional: sort by 7-dataset xScore
    df_summary_expanded = df_summary_expanded.sort_values("xScore_7datasets", ascending=False)

    return df_summary_expanded


def plot_xscore_correlation(df, x_col="xScore_7datasets", y_col="xScore_4datasets", 
                            g_col="G_i", v_col="V_i", label_col="Model",
                            figsize=(7,6), cmap="RdYlGn", save_path="xscore-7-4-corr.pdf"):
    # If Model is index, make it a column for labeling
    if df.index.name == label_col or label_col not in df.columns:
        df = df.reset_index()

    plt.figure(figsize=figsize)

    # Scatter: color = generalization, size = variance
    sc = plt.scatter(
        df[x_col], df[y_col],
        c=df[g_col], s=(df[v_col] * 2000) + 30,
        cmap=cmap, edgecolors="k", linewidths=0.7, alpha=0.6
    )

    # Identity line
    lims = [0, 1.0]
    plt.plot(lims, lims, 'k--', lw=1, alpha=0.7)
    plt.xlim(lims)
    plt.ylim(lims)

    # Labels
    i=0
    for _, row in df.iterrows():
        x = i*0.0035
        y = i*0.002
        
        if(i==0):
            x =-0.05
            y = 0.03

        if(i==2 or i==4):
            x = x-0.05
            y = y+0.03

        if(i==9):
            x = x-0.05
            y = y+0.03

        #x = x + (np.random.rand() - 0.5) * 0.01
        #y = y + (np.random.rand() - 0.5) * 0.01
    
        i=i+1
        plt.text(row[x_col] +x, row[y_col] - y, row[label_col], fontsize=8, alpha=0.9)

    # Axes, title, and colorbar
    plt.xlabel("7-dataset xScore", fontsize=12)
    plt.ylabel("4-dataset xScore", fontsize=12)
    plt.title("Correlation between 7-dataset and 4-dataset xScores", fontsize=13, pad=10)

    cb = plt.colorbar(sc)
    cb.set_label("Generalization Score (G_i)", rotation=270, labelpad=15)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()


# -----------------------------
# Run the function
# -----------------------------
#1. xscore
accu_table, xscore_7_datasets = compute_xscore_pair_and_summary(distilled_data, lam=0.5)

print("Combined Accuracy Table:\n", accu_table)
print("\nSummary Table (G_i, V_i, xScore):\n", xscore_7_datasets)

# 2. imagenette-xscore
imagenette_cs_xscore = generate_imagenette_xscore(distilled_data, xscore_7_datasets)
corr = plot_correlation_xscore_imagenette(imagenette_cs_xscore)
print("\nimagenette-xscore:\n", imagenette_cs_xscore)

# 3. best 4 datasets
best_4_datasets, best_r2, best_coeff, best_intercept = find_best_dataset_subset(distilled_data, imagenette_cs_xscore)
print("Best 4-dataset subset:", best_4_datasets)
print("R^2 with xScore:", best_r2)
print("Regression coefficients:", best_coeff)
print("Intercept:", best_intercept)

# 4. xscores out of 4 datasets
xscore_4_datasets = compute_xscore_from_subset(distilled_data, best_4_datasets, coeff=best_coeff, intercept=best_intercept)

print("Best 4-dataset subset:", best_4_datasets)
print("R^2 with xScore:", best_r2)
print(xscore_4_datasets)

# 5. compare xscore_7 and xscore_4
compared_scores = xscore7_vs_xscore4(xscore_7_datasets, xscore_4_datasets)
print(compared_scores)

plot_xscore_correlation(compared_scores)

