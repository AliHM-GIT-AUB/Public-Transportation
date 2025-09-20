
# Lebanon Transport — Streamlit App

This app reproduces and extends two visualizations from the earlier Plotly assignment with **interactive controls** using Streamlit.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure `Public Transportation.csv` sits in the same folder as `app.py`.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, create a new app, and point it to `app.py` in your repo.
3. Add the `Public Transportation.csv` file to the repo.
4. Click **Deploy** and copy the public URL.

## Interactions
- **Road type selector** and **bus-stop filter** recalculate the **Condition Composition** chart.
- **Transport mode multiselect** updates the **Mode Share** pie.

## Files
- `app.py` — Streamlit app.
- `requirements.txt` — Python dependencies.
- `Public Transportation.csv` — data file (provided).
