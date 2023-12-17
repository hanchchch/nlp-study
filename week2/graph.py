import pandas as pd
import plotly.express as px

df = pd.read_csv("week2/loss.csv")

fig = px.line(
    df, x="epoch", y="loss", title="Loss Per Epoch", color="model", markers=True
)
fig.show()
