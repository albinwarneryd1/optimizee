from __future__ import annotations

import pandas as pd
import joblib
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from opimizee.utils.config import DATA_PROCESSED, MODELS_DIR, DataSchema


def _load_data() -> pd.DataFrame:
    p = DATA_PROCESSED / "processed.parquet"
    if not p.exists():
        raise FileNotFoundError("processed.parquet not found. Run: opimizee preprocess")
    return pd.read_parquet(p).sort_values(DataSchema().datetime_col)


def _load_model():
    m = MODELS_DIR / "model.joblib"
    if not m.exists():
        raise FileNotFoundError("model.joblib not found. Run: opimizee train")
    payload = joblib.load(m)
    return payload["model"], payload["schema"]


def run_dashboard() -> None:
    df = _load_data()
    model, schema = _load_model()

    app = Dash(__name__)

    slider_id = "rows_slider"

    app.layout = html.Div(
        style={"maxWidth": "1100px", "margin": "30px auto", "fontFamily": "system-ui"},
        children=[
            html.H1("Opimizee", style={"marginBottom": "0"}),
            html.Div("Electricity insights + baseline ML forecast", style={"opacity": 0.7, "marginBottom": "20px"}),

            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "center"},
                children=[
                    html.Div("Show last N rows:"),
                    dcc.Slider(
                        200,
                        min(2000, len(df)),
                        step=100,
                        value=min(800, len(df)),
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                        id=slider_id,
                    ),
                    html.Div(id="kpi", style={"marginLeft": "auto", "fontWeight": 600}),
                ],
            ),

            dcc.Graph(id="trend", style={"marginTop": "20px"}),
            dcc.Graph(id="pred", style={"marginTop": "10px"}),

            html.H3("Quick takeaways", style={"marginTop": "30px"}),
            html.Ul(id="insights"),
        ],
    )

    @app.callback(
        Output("trend", "figure"),
        Output("pred", "figure"),
        Output("kpi", "children"),
        Output("insights", "children"),
        Input(slider_id, "value"),
    )
    def update(n: int):
        view = df.tail(int(n)).copy()

        fig_trend = px.line(view, x=schema.datetime_col, y=schema.target_col, title="Consumption trend")

        preds = model.predict(view)
        pred_df = view[[schema.datetime_col]].copy()
        pred_df["prediction"] = preds
        fig_pred = px.line(pred_df, x=schema.datetime_col, y="prediction", title="Model prediction (baseline)")

        kpi = f"Rows: {len(view)} | Avg: {view[schema.target_col].mean():.2f} | Max: {view[schema.target_col].max():.2f}"

        hour_avg = view.groupby("hour")[schema.target_col].mean().sort_values(ascending=False)
        peak_hour = int(hour_avg.index[0])

        dow_avg = view.groupby("dayofweek")[schema.target_col].mean().sort_values(ascending=False)
        peak_dow = int(dow_avg.index[0])

        insights = [
            html.Li(f"Peak hour (avg consumption): {peak_hour}:00"),
            html.Li(f"Highest day-of-week (0=Mon): {peak_dow}"),
            html.Li("Baseline model (RandomForest). Improve with weather, spot price, and appliance-level signals."),
        ]

        return fig_trend, fig_pred, kpi, insights

    app.run_server(debug=True)
