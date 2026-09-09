# Renewable Energy Adoption — Decision Tree Classifier (Streamlit)

An interactive Streamlit app built from `4__Decision_Trees.ipynb`. It trains a
`DecisionTreeClassifier` to predict renewable-energy **adoption** and lets you:

- Predict adoption live with sliders, including the exact decision path taken
- Score a batch of scenarios by uploading a CSV
- Explore the dataset (distributions, correlations, class balance)
- Visualize the trained tree (and download it as a PNG)
- Inspect model performance (accuracy, confusion matrix, ROC/AUC, feature importance)

## Files

```
dt_streamlit_app/
├── app.py             # the Streamlit app
├── requirements.txt   # Python dependencies
└── README.md          # this file
```

## ⚠️ About the dataset

The notebook loads `Renewable_Energy_Adoption.csv`, which was **not included** with the
notebook upload. The app therefore lets you:

- **Upload the real CSV** in the sidebar (columns: `carbon_emissions`, `energy_output`,
  `renewability_index`, `cost_efficiency`, `adoption`) — recommended, or
- **Use the built-in demo dataset**, a synthetic stand-in with the same schema, clearly
  labeled in the UI, so the app is fully functional even without the original file.

## 1. Run it locally

```bash
# (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

## 2. Deploy for free — Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (must contain `app.py` and `requirements.txt`).
   Optionally include your `Renewable_Energy_Adoption.csv` in the repo if you want a
   default dataset baked in (then add a small code tweak to auto-load it).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to `app.py`.
4. Click **Deploy**. You'll get a shareable URL like
   `https://<your-app-name>.streamlit.app`.

Any time you push new commits to the repo, the app redeploys automatically.

## 3. Deploy on Hugging Face Spaces (alternative)

1. Create a new Space at https://huggingface.co/new-space, choosing **Streamlit** as the SDK.
2. Upload `app.py` and `requirements.txt` (or push via git — Spaces are git repos).
3. The Space builds and serves the app automatically at
   `https://huggingface.co/spaces/<username>/<space-name>`.

## 4. Deploy with Docker (any cloud VM / container service)

Create a `Dockerfile` alongside `app.py`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then build and run:

```bash
docker build -t adoption-tree-app .
docker run -p 8501:8501 adoption-tree-app
```

Push the image to any container registry (Docker Hub, ECR, GCR) and deploy it on
Render, Railway, AWS App Runner, Google Cloud Run, Azure Container Apps, etc.

## Notes

- Max tree depth, test split size, and random state are all adjustable from the sidebar
  and retrain the model live (cached with `st.cache_resource`).
- The tree viewer renders with `sklearn.tree.plot_tree` and can be downloaded as a PNG.
- The single-prediction view shows the exact decision path (which nodes/thresholds were
  used) for full interpretability.
