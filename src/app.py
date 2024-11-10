import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import matplotlib
from scipy.interpolate import griddata

matplotlib.use('agg')

data = pd.read_csv('../universities.csv')
datasal = pd.read_csv('../salary.csv')
datagen = pd.read_csv('../Gender.csv')

def contourPlot():
    fig = go.Figure(
        data=go.Contour(
            z=data['AI Research Funding (millions)'],
            x=data['Ranking Europe'],
            y=data['Enrollment'],
            colorscale="Viridis",
            contours=dict(
                start=data['AI Research Funding (millions)'].min(),
                end=data['AI Research Funding (millions)'].max(),
                size=2
            ),
            colorbar=dict(title="AI Research Funding (millions)")
        )
    )

    fig.update_layout(
        title="Contour Plot",
        xaxis_title="Ranking Europe",
        yaxis_title="Enrollment"
    )

    return fig

def DPlot():
    x = data['Ranking Europe']
    y = data['Enrollment']
    z = data['AI Research Funding (millions)']

    # Create a meshgrid for the surface plot
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))

    # Interpolate the data to fit the meshgrid
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')

    # Plot the 3D surface plot
    fig = go.Figure(data=go.Surface(
        z=z_grid,
        x=x_grid,
        y=y_grid,
        colorscale='Viridis',
        colorbar=dict(title='AI Research Funding (millions)')
    ))

    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='Ranking Europe',
            yaxis_title='Enrollment',
            zaxis_title='AI Research Funding'
        ),
    )
    return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([

    dbc.Navbar([
        dbc.NavItem(dbc.NavLink("Bar Chart", href='#bar-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(
            dbc.NavLink("Scatter Chart", href='#scatter-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(dbc.NavLink("Geo Chart", href='#geo-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(dbc.NavLink("Line Chart", href='#line-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(dbc.NavLink("Pie Chart", href='#pie-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(dbc.NavLink("Error Chart", href='#error-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(
            dbc.NavLink("Contour Chart", href='#contour-chart', external_link=True, className="nav-link-custom")),
        dbc.NavItem(dbc.NavLink("3D Chart", href='#3d-chart', external_link=True, className="nav-link-custom")),
    ],style={'background-color':'green'},sticky='top',dark=True,color='primary',className="mb-3"),

    html.H1("AI and Education in Ireland", className='mb-2 text-primary', style={'textAlign':'center'}),
    html.Div(className="bg-opacity-75 p-2 m-1 bg-primary text-light fw-bold rounded"),

    #Bar Plot
    html.Div([
        dbc.Row([
            html.H5("Top 20 University for AI in Ireland", className='mb-2',style={'textAlign':'center'}),
            html.H6("Choose Category",style={'width':"20%"}),
            dcc.Dropdown(
                id='category',
                value='Ranking Europe',
                clearable=False,
                options=data.columns[1:4],
                style={'width': "50%"},
            )
        ]),
        dbc.Row([
            dcc.Graph(id='bar', figure={})
        ]),
    ],id='bar-chart',className='border rounded border-dark'),

    html.Div(className="bg-opacity-75 p-2 m-1 bg-primary text-light fw-bold rounded"),

    #Scatter Plot and Geo Plot
    html.Div([
        dbc.Row([
            dbc.Row([
                html.P("Choose Top", style={"width": '15%'}),
                dbc.Input(type='number', placeholder='Enter Number', value=10, id='top', style={'width': '15%'}),
            ]),
            dbc.Col([
                html.Div([
                    html.H5("Top AI University Enrollment with European Ranking in Ireland", className='mb-2', style={'textAlign':'center'}),
                    dcc.Graph(id='scatter',figure={}),

                ],id='scatter-chart',className='border rounded border-dark')
            ]),
            dbc.Col([
                html.Div([
                    html.H5("Top AI University in Ireland on Map", className='mb-2', style={'textAlign':'center'}),
                    dcc.Graph(id='geo'),
                ],id='geo-chart',className='border rounded border-dark')
            ]),
        ]),

    ]),

    html.Div(className="bg-opacity-75 p-2 m-1 bg-primary text-light fw-bold rounded"),

    #Line Plot and Pie Plot
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Outcome annually of the student study AI in Ireland", className='mb-2',style={'textAlign': 'center'}),
                    dcc.Graph(id='line',figure={}),
                ], id='line-chart', className='border rounded border-dark'),
            ]),
            dbc.Col([
                html.Div([
                    html.H5("AI Enrollment Gender Ratio in Ireland", className='mb-2', style={'textAlign': 'center'}),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='pie1', figure={})
                        ]),
                        dbc.Col([
                            dcc.Graph(id='pie2', figure={})
                        ])
                    ])
                ],id='pie-chart', className='border rounded border-dark'),
            ]),
            dbc.Row([
                dbc.Label("Select Year"),
                dcc.RangeSlider(
                    min=2015,
                    max=2024,
                    step=1,
                    value=[2015, 2024],
                    marks={i: '{}'.format(i) for i in range(2015, 2025, 1)},
                    id='year',
                    tooltip={"placement": "bottom"},
                )
            ])
        ]),
    ]),

    html.Div(className="bg-opacity-75 p-2 m-1 bg-primary text-light fw-bold rounded"),

    #Error Plot
    html.Div([
        dbc.Row([
            html.H5("University Enrollment with Error Bars", className='mb-2', style={'textAlign': 'center'}),
            html.P("Choose a number of error (%)",style={"width":'20%'}),
            dbc.Input(type='number', placeholder='Choose Error %', value=5, id='nume',style={'width':'15%'}),
            dcc.Graph(id='error',figure={}),
        ]),
    ],id='error-chart', className='border rounded border-dark'),

    html.Div(className="bg-opacity-75 p-2 m-1 bg-primary text-light fw-bold rounded"),

    #Contour Plot and 3D Plot
    html.Div([
        dbc.Row([
            html.H5("AI Research Funding by European Ranking and Enrollment", className='mb-2', style={'textAlign': 'center'}),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=contourPlot())
                ],id='contour-chart', className='border rounded border-dark')
            ]),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=DPlot())
                ], id='3d-chart', className='border rounded border-dark')
            ])
        ]),
    ]),

    html.Div(className="bg-opacity-75 p-2 m-1 bg-primary text-light fw-bold rounded"),

],style={"textAlign":'center',"background-color": "#D3D3D3"},id='container',fluid=True)

#BarPlot
@app.callback(
    Output('bar', 'figure'),
    Input('category', 'value'),
)
def bar_plot(selected_yaxis):
    fig_bar_plotly = px.bar(data, x='Name University', y=selected_yaxis).update_xaxes(tickangle=330)
    return  fig_bar_plotly

#ScatterPlot and GeoPlot
@app.callback(
    Output('scatter','figure'),
    Output('geo','figure'),
    Input('top','value')
)
def scatter_and_geo_plot(top):
    dfscatter = data.head(top)
    scatter = px.scatter(dfscatter, x='Ranking Europe', y='Enrollment', text='Name University')
    scatter.update_traces(textposition='top right')

    geo = px.scatter_geo(dfscatter, lat='latitude',lon='longitude',hover_name='Name University',fitbounds='geojson')
    geo.update_geos(
        showcoastlines = True,
        showcountries = True,
        showframe = True,
        showlakes = True,
        showland = True,
        showocean = True,
        showrivers = True
    )
    return scatter,geo

#LinePlot and PiePlot
@app.callback(
    Output('line','figure'),
    Output('pie1','figure'),
    Output('pie2','figure'),
    Input('year','value')
)
def line_and_pie_plot(year_list):
    dfyear = datasal[(datasal['year']>=year_list[0]) & (datasal['year']<=year_list[1])]
    line = px.line(dfyear, x='year', y=['intern', 'junior', 'senior'], markers=True)
    line.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=35000,
            dtick=10000
        )
    )

    pie1 = px.pie(datagen,values=str(year_list[0]),title=f'Data in {str(year_list[0])}',names='Gender')
    pie2 = px.pie(datagen,values=str(year_list[1]),title=f'Data in {str(year_list[1])}',names='Gender')

    return line ,pie1, pie2

#ErrorPlot
@app.callback(
    Output('error','figure'),
    Input('nume','value')
)
def error_plot(nume):
    data['e'] = data['Enrollment'] * (nume/100)
    error = px.scatter(data,x='Name University',y='Enrollment',error_y='e')
    return error


if __name__ == '__main__':
    app.run_server(debug=True, port=8008)