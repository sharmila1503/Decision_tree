import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Renewable Energy Adoption — Decision Tree",
    page_icon="🌳",
    layout="wide",
)

FEATURES = ["carbon_emissions", "energy_output", "renewability_index", "cost_efficiency"]
TARGET = "adoption"
CLASS_NAMES = ["Non-Adoption", "Adoption"]

FEATURE_BOUNDS = {
    "carbon_emissions": (50.0, 400.0, "Emissions in hypothetical units. Lower generally favors adoption."),
    "energy_output": (100.0, 1000.0, "Energy produced, in hypothetical units."),
    "renewability_index": (0.0, 1.0, "0 = fully non-renewable, 1 = fully renewable."),
    "cost_efficiency": (0.5, 5.0, "Lower score = more cost efficient."),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data(num_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Fallback demo dataset with the same schema as 'Renewable_Energy_Adoption.csv'.
    The original CSV used in the notebook was not provided, so this generates a
    plausible synthetic stand-in: low emissions + high renewability + good cost
    efficiency + decent energy output favor adoption. Replace with your real CSV
    for actual results.
    """
    rng = np.random.RandomState(seed)
    carbon_emissions = rng.uniform(50, 400, num_samples)
    energy_output = rng.uniform(100, 1000, num_samples)
    renewability_index = rng.uniform(0, 1, num_samples)
    cost_efficiency = rng.uniform(0.5, 5, num_samples)

    score = (
        -0.01 * carbon_emissions
        + 0.002 * energy_output
        + 3.0 * renewability_index
        - 0.8 * cost_efficiency
    )
    prob = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-9)))
    adoption = (rng.uniform(0, 1, num_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "carbon_emissions": carbon_emissions,
            "energy_output": energy_output,
            "renewability_index": renewability_index,
            "cost_efficiency": cost_efficiency,
            "adoption": adoption,
        }
    )


@st.cache_resource
def train_model(df: pd.DataFrame, max_depth: int, test_size: float, random_state: int):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else np.zeros(len(X_test))
    return model, X_train, X_test, y_train, y_test, y_pred, y_proba


# ----------------------------------------------------------------------------
# Sidebar — data source & model controls
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Data & Model Settings")

data_source = st.sidebar.radio(
    "Dataset source",
    ["Upload Renewable_Energy_Adoption.csv", "Use built-in demo data"],
)

if data_source == "Upload Renewable_Energy_Adoption.csv":
    uploaded = st.sidebar.file_uploader(
        "CSV with columns: " + ", ".join(FEATURES + [TARGET]), type=["csv"]
    )
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing_cols = set(FEATURES + [TARGET]) - set(data.columns)
        if missing_cols:
            st.sidebar.error(f"Missing required columns: {missing_cols}")
            st.stop()
        data = data.dropna()
        st.sidebar.success(f"Loaded {len(data)} rows.")
    else:
        st.sidebar.info("Upload your CSV, or switch to the demo dataset below.")
        data = generate_demo_data()
        st.sidebar.caption("Currently showing demo data until a CSV is uploaded.")
else:
    num_samples = st.sidebar.slider("Number of demo samples", 50, 1000, 200, step=50)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    data = generate_demo_data(num_samples=num_samples, seed=seed)
    st.sidebar.warning(
        "This is synthetic demo data (the original CSV wasn't provided). "
        "Upload the real file for accurate results."
    )

max_depth = st.sidebar.slider("Max tree depth", 1, 10, 3)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Train/test split random state", value=42, step=1)

model, X_train, X_test, y_train, y_test, y_pred, y_proba = train_model(
    data, max_depth, test_size, random_state
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Reproduces the Decision Tree notebook: load data → train/test split → "
    "DecisionTreeClassifier → evaluation, plus an interactive predictor and tree viewer."
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌳 Renewable Energy Adoption — Decision Tree Classifier")
st.write(
    "A Decision Tree model that predicts whether a renewable-energy scenario will "
    "see **adoption**, based on carbon emissions, energy output, renewability index, "
    "and cost efficiency."
)

tab_predict, tab_explore, tab_tree, tab_performance, tab_about = st.tabs(
    ["🔮 Predict", "📊 Explore Data", "🌲 Tree Viewer", "📈 Model Performance", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# Tab 1: Interactive prediction
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Try it yourself")
    st.write("Adjust the sliders to describe a scenario, then see the model's prediction live.")

    col1, col2 = st.columns([1, 1])

    input_values = {}
    with col1:
        for feat in FEATURES:
            lo, hi, help_text = FEATURE_BOUNDS[feat]
            default = float(data[feat].mean())
            step = (hi - lo) / 100
            input_values[feat] = st.slider(
                feat.replace("_", " ").title(),
                min_value=float(lo),
                max_value=float(hi),
                value=float(np.clip(default, lo, hi)),
                step=float(step),
                help=help_text,
            )

    input_df = pd.DataFrame([input_values])[FEATURES]
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    with col2:
        st.markdown("#### Prediction")
        if prediction == 1:
            st.success("✅ **Adoption**")
        else:
            st.error("❌ **Non-Adoption**")

        if len(proba) > 1:
            st.metric("Probability of Adoption", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

        st.markdown("#### Input summary")
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

        with st.expander("Decision path for this input"):
            node_indicator = model.decision_path(input_df)
            leaf_id = model.apply(input_df)
            feature = model.tree_.feature
            threshold = model.tree_.threshold
            node_index = node_indicator.indices[
                node_indicator.indptr[0] : node_indicator.indptr[1]
            ]
            lines = []
            for node_id in node_index:
                if leaf_id[0] == node_id:
                    lines.append(f"**Leaf node {node_id}** → prediction made here")
                    continue
                fname = FEATURES[feature[node_id]]
                val = input_df.iloc[0][fname]
                direction = "<=" if val <= threshold[node_id] else ">"
                lines.append(f"Node {node_id}: `{fname}` = {val:.2f} {direction} {threshold[node_id]:.2f}")
            st.markdown("\n\n".join(lines))

    st.markdown("---")
    st.subheader("Batch prediction")
    st.write("Upload a CSV with the same feature columns to score many rows at once.")
    batch_file = st.file_uploader(
        "CSV with columns: " + ", ".join(FEATURES), type=["csv"], key="batch_upload"
    )
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = set(FEATURES) - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            batch_df["predicted_adoption"] = model.predict(batch_df[FEATURES])
            if len(model.classes_) > 1:
                batch_df["probability_adoption"] = model.predict_proba(batch_df[FEATURES])[:, 1]
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "⬇️ Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# Tab 2: Explore the data
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"Full dataset: {data.shape[0]} rows × {data.shape[1]} columns")

    st.subheader("Class balance")
    counts = data[TARGET].value_counts().rename({0: "Non-Adoption", 1: "Adoption"})
    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot(kind="bar", color=["#d9534f", "#5cb85c"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.subheader("Feature distributions by class")
    feat_choice = st.selectbox("Choose a feature", FEATURES)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.histplot(data=data, x=feat_choice, hue=TARGET, kde=True, palette=["#d9534f", "#5cb85c"], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation heatmap")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(data[FEATURES + [TARGET]].corr(), annot=True, cmap="Blues", ax=ax3)
    st.pyplot(fig3)

# ----------------------------------------------------------------------------
# Tab 3: Tree viewer
# ----------------------------------------------------------------------------
with tab_tree:
    st.subheader("Visualize the trained tree")
    st.write(f"Current max depth: **{max_depth}**. Adjust it from the sidebar and the tree updates.")

    fig4, ax4 = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        ax=ax4,
    )
    st.pyplot(fig4)

    buf = io.BytesIO()
    fig4.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    st.download_button(
        "⬇️ Download tree as PNG",
        buf.getvalue(),
        file_name="decision_tree.png",
        mime="image/png",
    )

# ----------------------------------------------------------------------------
# Tab 4: Model performance
# ----------------------------------------------------------------------------
with tab_performance:
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Summary metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Train samples", len(X_train))
    m3.metric("Test samples", len(X_test))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig5, ax5 = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax5,
        )
        ax5.set_xlabel("Predicted")
        ax5.set_ylabel("Actual")
        st.pyplot(fig5)

    with col_b:
        st.markdown("#### ROC Curve")
        if y_test.nunique() > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig6, ax6 = plt.subplots(figsize=(4, 4))
            ax6.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#5cb85c")
            ax6.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax6.set_xlabel("False Positive Rate")
            ax6.set_ylabel("True Positive Rate")
            ax6.legend(loc="lower right")
            st.pyplot(fig6)
        else:
            st.info("ROC curve needs both classes present in the test set.")

    st.markdown("#### Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("#### Feature Importance")
    importance = pd.DataFrame(
        {"Feature": FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    fig7, ax7 = plt.subplots(figsize=(6, 3))
    ax7.barh(importance["Feature"], importance["Importance"], color="#5cb85c")
    ax7.set_xlabel("Importance")
    ax7.invert_yaxis()
    st.pyplot(fig7)
    st.dataframe(importance, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 5: About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### About this app

        This Streamlit app is an interactive wrapper around the notebook
        **`4__Decision_Trees.ipynb`**. It reproduces the same pipeline:

        1. Load `Renewable_Energy_Adoption.csv` (upload it in the sidebar) with features
           `carbon_emissions`, `energy_output`, `renewability_index`, `cost_efficiency`
           and target `adoption`.
        2. Split into train/test sets.
        3. Fit a `sklearn.tree.DecisionTreeClassifier` (max depth adjustable, default 3
           as in the notebook).
        4. Evaluate with accuracy, a confusion matrix, ROC/AUC, classification report,
           and feature importances.
        5. Visualize the trained tree, and interactively predict on new inputs — one at a
           time (with the decision path shown) or in batch via CSV upload.

        **Note on demo data** — the original `Renewable_Energy_Adoption.csv` file used in
        the notebook wasn't provided alongside it. If you don't upload it, the app falls
        back to a synthetic dataset with the same column schema so the app is still fully
        functional; upload your real CSV in the sidebar for accurate results.
        """
    )
