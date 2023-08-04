from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd


def rename_hurricanes(x):
    if x in ['florence', 'michael', 'harvey', 'matthew']:
        return "hurricane"
    else:
        return x


df = pd.read_csv("data/umap_embedding.csv")
df = df.rename(columns={"labels": "event_name"})
df["event_type"] = df["event_name"].apply(lambda x: x.split("-")[-1])
df["event_type"] = df["event_type"].apply(rename_hurricanes)

loc_map = {"portugal-wildfire": "portugal",

           "pinery-bushfire": "pinery",

           "nepal-flooding": "nepal",

           "lower-puna-volcano": "lower-puna",
           # South East Asia
           "palu-tsunami": "South East Asia",
           "sunda-tsunami": "South East Asia",
           # California
           "socal-fire": "California",
           "woolsey-fire": "California",
           "santa-rosa-wildfire": "California",
           # US Midwest
           "midwest-flooding": "US Midwest",
           "moore-tornado": "US Midwest",
           "joplin-tornado": "US Midwest",
           "hurricane-harvey": "US Midwest",
           "tuscaloosa-tornado": "US Midwest",
           #Floria and around
           "hurricane-florence": "Floria and around",
           "hurricane-michael": "Floria and around",
           "hurricane-matthew": "Floria and around",
           # Central America
           "guatemala-volcano": "Central America",
           "mexico-earthquake": "Central America"}

df["location"] = df["event_name"].apply(lambda x: loc_map[x])

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='UMap embeddings of disasters',
            style={'textAlign': 'center'}),
    dcc.Dropdown(["event_name", "event_type", "location"],
                 "event_name", id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])


@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    fig = px.scatter(df, x='umap_x', y='umap_y', color=value,
                     hover_data={'event_name': True, "location": True, "umap_x": False, "umap_y": False}, width=1800, height=1200)
    fig.update_traces(marker_size=6)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8023)
