from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table
import dash_design_kit as ddk
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import os
import redis
import pickle
# resampler stuff
# from dash_extensions.enrich import (
#     DashProxy,
#     ServersideOutputTransform,
# )
from trace_updater import TraceUpdater

from plotly_resampler import FigureResampler


app = Dash(__name__)
server = app.server  # expose server variable for Procfile
redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))


app.layout = ddk.App([
    ddk.Header([
        ddk.Title('Sample Resampler Application'),
    ]),
    html.Div(id='kick', style={'visibility':'hidden'}),
    ddk.Graph(id='graph-id'),
    ddk.Row(children=[
        ddk.Card(width=.33, children=[
            ddk.CardHeader(title='.5 std'),
            dcc.Loading(
                dash_table.DataTable(id='half', columns=[{'name':'Time', 'id':'Time'},{'name':'a', 'id':'a'}])
            )
        ]),
        ddk.Card(width=.33, children=[
            ddk.CardHeader(title='.75 std'),
            dcc.Loading(            
                dash_table.DataTable(id='three-quarters', columns=[{'name':'Time', 'id':'Time'},{'name':'a', 'id':'a'}])
            )
        ]),
        ddk.Card(width=.33, children=[
            ddk.CardHeader(title='1 std'),
            dcc.Loading(
                dash_table.DataTable(id='one', columns=[{'name':'Time', 'id':'Time'},{'name':'a', 'id':'a'}])
            )
        ])
    ]),
    TraceUpdater(id="trace-updater", gdID="graph-id"),
])


@app.callback(
    [
        Output("graph-id", "figure"), 
        Output('half', 'data'),
        Output('three-quarters', 'data'),
        Output('one', 'data')
    ],
    Input("kick", "n_clicks"),
    memoize=True,
)
def plot_graph(n_clicks):

    # This random data generation is stolen directly from the datashader timeseries docs: https://datashader.org/user_guide/Timeseries.html
    # Constants
    # don't seed, get different values everytime -----> np.random.seed(2)
    n = int(1 * 10 * 100000)               # Number of points
    cols = list('abcdefg')                   # Column names of samples
    start = datetime.datetime(2010, 10, 1, 0)   # Start time
    # Generate a fake signal
    signal = np.random.normal(0, 0.3, size=n).cumsum() + 50

    # Generate many noisy samples from the signal
    noise = lambda var, bias, n: np.random.normal(bias, var, n)
    data = {c: signal + noise(1, 10*(np.random.random() - 0.5), n) for c in cols}

    # Add some "rogue lines" that differ from the rest 
    cols += ['x'] ; data['x'] = signal + np.random.normal(0, 0.02, size=n).cumsum() # Gradually diverges
    cols += ['y'] ; data['y'] = signal + noise(1, 20*(np.random.random() - 0.5), n) # Much noisier
    cols += ['z'] ; data['z'] = signal # No noise at all

    # Pick a few samples from the first line and really blow them out
    locs = np.random.choice(n, 15)

    # Create a dataframe
    data['Time'] = [start + datetime.timedelta(minutes=1)*i for i in range(n)]

    df = pd.DataFrame(data)

    stats = df.describe()

    a_std = stats['a'].loc['std']

    df['a'].iloc[locs[0:5]] = df['a'].iloc[locs[0:5]] + (np.sign(df['a'].iloc[locs[0:5]]))*(0.5*a_std)
    df['a'].iloc[locs[5:10]] = df['a'].iloc[locs[5:10]] + (np.sign(df['a'].iloc[locs[5:10]]))*(0.75*a_std)
    df['a'].iloc[locs[10:15]] = df['a'].iloc[locs[10:15]] + (np.sign(df['a'].iloc[locs[10:15]]))*(a_std)

    outs = df.iloc[locs]

    figure = px.scatter(df, x='Time', y=['a','b','c'])
    figure.update_traces(marker={'size': 8})
    figure.update_layout(hovermode="x unified")
    figure.update_yaxes(fixedrange=True)
    fig: FigureResampler = FigureResampler(figure)
    fig_pic = pickle.dumps(fig)
    redis_instance.hset('resam', 'fig_cache', fig_pic)

    return [
                fig, 
                # fig, 
                outs[['Time', 'a']].iloc[0:5].sort_values(['Time']).to_dict('records'),  
                outs[['Time', 'a']].iloc[5:10].sort_values(['Time']).to_dict('records'),  
                outs[['Time', 'a']].iloc[10:15].sort_values(['Time']).to_dict('records')
            ]


@app.callback(
    Output("trace-updater", "updateData"),
    Input("graph-id", "relayoutData"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata):
    fig_cache = redis_instance.hget('resam', 'fig_cache')
    if fig_cache is None:
        return no_update
    fig = pickle.loads(fig_cache)
    return fig.construct_update_data(relayoutdata)

if __name__ == '__main__':
    app.run_server(debug=True)
