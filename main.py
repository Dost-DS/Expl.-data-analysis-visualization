import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime

# Load and prepare data
def load_data(file_path):
    """Load and prepare the energy data for visualization."""
    df = pd.read_csv(file_path)
    
    # Remove aggregated regions (like World, continents)
    aggregates = ['World', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
    df = df[~df['country'].isin(aggregates)]
    
    # Fill NaN values for better visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Create derived features for analysis
    df['renewable_to_fossil_ratio'] = np.where(df['fossil_share_energy'] > 0,
                                             df['renewables_share_energy'] / df['fossil_share_energy'],
                                             0)
    
    # Calculate energy transition speed (5-year change in renewable share)
    df_sorted = df.sort_values(['country', 'year'])
    df_sorted['renewables_5yr_change'] = df_sorted.groupby('country')['renewables_share_energy'].diff(5)
    
    # Create region classification for analysis
    regions = {
        'Europe': ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 
                  'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 
                  'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 
                  'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 
                  'United Kingdom', 'Norway', 'Switzerland', 'Iceland'],
        'North America': ['United States', 'Canada', 'Mexico'],
        'Asia': ['China', 'Japan', 'South Korea', 'India', 'Indonesia', 'Malaysia', 'Vietnam', 
                'Thailand', 'Philippines', 'Singapore', 'Pakistan', 'Bangladesh'],
        'Middle East': ['Saudi Arabia', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Oman', 
                       'Israel', 'Turkey', 'Iran', 'Iraq'],
        'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela'],
        'Africa': ['South Africa', 'Nigeria', 'Egypt', 'Algeria', 'Morocco', 'Kenya', 'Ethiopia'],
        'Oceania': ['Australia', 'New Zealand']
    }
    
    # Map countries to regions
    df_sorted['region'] = 'Other'
    for region, countries in regions.items():
        df_sorted.loc[df_sorted['country'].isin(countries), 'region'] = region
    
    return df_sorted

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
app.title = "Global Energy Transition Explorer"

# Define color scales with better accessibility and aesthetic appeal
color_scales = {
    'renewables_share_energy': px.colors.sequential.Greens,
    'fossil_share_energy': px.colors.sequential.Oranges_r,
    'co2': px.colors.sequential.Reds,
    'energy_per_capita': px.colors.sequential.Purples,
    'renewables_5yr_change': px.colors.diverging.RdBu,
    'renewable_to_fossil_ratio': px.colors.sequential.Viridis
}

# Placeholder for data loading
df = pd.DataFrame()

# Define available metrics with improved descriptions
available_metrics = [
    {"label": "Renewable Energy Share (%)", "value": "renewables_share_energy"},
    {"label": "Fossil Fuels Share (%)", "value": "fossil_share_energy"},
    {"label": "CO2 Emissions (kt)", "value": "co2"},
    {"label": "Energy Per Capita (kWh)", "value": "energy_per_capita"},
    {"label": "5-Year Renewable Growth", "value": "renewables_5yr_change"},
    {"label": "Renewable to Fossil Ratio", "value": "renewable_to_fossil_ratio"}
]


metric_descriptions = {
    "renewables_share_energy": "Percentage of total energy consumption from renewable sources",
    "fossil_share_energy": "Percentage of total energy consumption from fossil fuels",
    "co2": "Total carbon dioxide emissions in kilotons",
    "energy_per_capita": "Total energy consumption per person",
    "renewables_5yr_change": "Change in renewable energy share over the past 5 years",
    "renewable_to_fossil_ratio": "Ratio of renewable energy to fossil fuel consumption"
}

# App layout with improved UI/UX
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Global Energy Transition Explorer", className="text-primary"),
                html.P("Interactive visualization of global energy patterns and the transition to renewable energy sources", 
                       className="lead")
            ], className="page-header my-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Dashboard Controls", className="d-flex align-items-center"),
                    html.Span(id="last-update", className="ms-auto text-muted small")
                ], className="d-flex justify-content-between"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Time Period:", className="fw-bold"),
                        dcc.RangeSlider(
                            id="year-range-slider",
                            min=1990,
                            max=2020,
                            step=1,
                            marks={i: str(i) for i in range(1990, 2021, 5)},
                            value=[1990, 2020],
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Map Metric:", className="fw-bold"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=available_metrics,
                            value="renewables_share_energy",
                            clearable=False,
                            className="mb-2"
                        ),
                        html.P(id="metric-description", className="text-muted small"),
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Region Filter:", className="fw-bold"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label": "All Regions", "value": "all"},
                                {"label": "Europe", "value": "Europe"},
                                {"label": "Asia", "value": "Asia"},
                                {"label": "North America", "value": "North America"},
                                {"label": "South America", "value": "South America"},
                                {"label": "Africa", "value": "Africa"},
                                {"label": "Middle East", "value": "Middle East"},
                                {"label": "Oceania", "value": "Oceania"}
                            ],
                            value="all",
                            clearable=False
                        ),
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Selected Country:", className="fw-bold"),
                        html.Div(id="selected-country", className="lead text-primary fw-bold mt-2"),
                        html.Button("Clear Selection", id="clear-selection", 
                                   className="btn btn-outline-secondary mt-2")
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Analysis Mode:", className="fw-bold"),
                        dbc.RadioItems(
                            id="analysis-mode",
                            options=[
                                {"label": "Single Year", "value": "single_year"},
                                {"label": "Time Series", "value": "time_series"},
                                {"label": "Comparison", "value": "comparison"}
                            ],
                            value="single_year",
                            inline=True,
                            className="mt-2"
                        ),
                        html.Div(id="comparison-controls", className="mt-3", style={"display": "none"}, children=[
                            html.Label("Compare With:", className="fw-bold small"),
                            dcc.Dropdown(
                                id="comparison-year",
                                options=[{"label": str(i), "value": i} for i in range(1990, 2021)],
                                value=2010,
                                clearable=False
                            )
                        ])
                    ])
                ])
            ], className="mb-4 sticky-top")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4(id="map-title", className="card-title"),
                    dbc.Button("View Regional Stats", id="toggle-regional-stats", 
                              color="link", size="sm", className="ms-auto")
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    dcc.Loading(
                        id="map-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="choropleth-map", 
                                     style={"height": "500px"}, 
                                     config={"displayModeBar": True, 
                                           "modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                        ]
                    ),
                    dbc.Collapse(
                        id="regional-stats-collapse",
                        is_open=False,
                        children=[
                            html.Div(id="regional-stats", className="mt-3")
                        ]
                    )
                ])
            ], className="mb-4")
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4(id="time-series-title", className="card-title"),
                    dbc.Button("View Trend Analysis", id="toggle-trend-analysis", 
                              color="link", size="sm", className="ms-auto")
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    dcc.Loading(
                        id="time-series-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="time-series", 
                                     style={"height": "400px"}, 
                                     config={"displayModeBar": True,
                                            "modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                        ]
                    ),
                    dbc.Collapse(
                        id="trend-analysis-collapse",
                        is_open=False,
                        children=[
                            html.Div(id="trend-analysis", className="mt-3")
                        ]
                    )
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4(id="energy-mix-title", className="card-title"),
                    dbc.Button("View Energy Breakdown", id="toggle-energy-breakdown", 
                              color="link", size="sm", className="ms-auto")
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    dcc.Loading(
                        id="energy-mix-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="energy-mix", 
                                     style={"height": "400px"}, 
                                     config={"displayModeBar": True,
                                            "modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                        ]
                    ),
                    dbc.Collapse(
                        id="energy-breakdown-collapse",
                        is_open=False,
                        children=[
                            html.Div(id="energy-breakdown", className="mt-3")
                        ]
                    )
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Energy Transition Analysis", className="card-title"),
                    dbc.Button("View Statistical Summary", id="toggle-stats-summary", 
                              color="link", size="sm", className="ms-auto")
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    dcc.Loading(
                        id="scatter-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="scatter-plot", 
                                     style={"height": "400px"}, 
                                     config={"displayModeBar": True,
                                            "modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                        ]
                    ),
                    dbc.Collapse(
                        id="stats-summary-collapse",
                        is_open=False,
                        children=[
                            html.Div(id="stats-summary", className="mt-3")
                        ]
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("About This Dashboard", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Data Story & Key Insights"),
                            html.P([
                                "This interactive dashboard visualizes global energy transition patterns, highlighting the shift ",
                                "from fossil fuels to renewable energy sources. The visualizations reveal how different regions ",
                                "are addressing climate challenges through changes in their energy mix."
                            ]),
                            html.Ul([
                                html.Li("Nordic countries typically lead in renewable energy adoption, with hydro and wind power dominating"),
                                html.Li("Developing economies show varying patterns in their energy transition journey"),
                                html.Li("CO₂ emissions correlate strongly with both economic development and fossil fuel dependency"),
                                html.Li("Recent years show accelerating renewable adoption globally, though regional differences remain significant"),
                                html.Li("Energy transition speed varies widely, influenced by policy, geography, and economic factors")
                            ])
                        ], width=6),
                        
                        dbc.Col([
                            html.H5("Interaction Guide"),
                            html.P([
                                "This dashboard uses interactive linked visualizations to explore different dimensions of the global energy transition. ",
                                "Selections in one chart affect all other charts, enabling multi-dimensional analysis."
                            ]),
                            html.Ul([
                                html.Li("Click any country on the map to select it and update all charts"),
                                html.Li("Use time range sliders to explore temporal patterns and transitions"),
                                html.Li("Switch between metrics to view different aspects of energy consumption"),
                                html.Li("Toggle between single-year, time series, and comparison modes for different analyses"),
                                html.Li("Filter by region to focus on specific geographical areas"),
                                html.Li("Hover over charts for detailed information tooltips")
                            ]),
                            html.P([
                                "Data sourced from global energy statistics covering consumption, production, ",
                                "and emissions metrics from 1990 to 2020 across multiple countries."
                            ], className="text-muted small mt-3")
                        ], width=6)
                    ])
                ])
            ])
        ], width=12)
    ]),
    
    # Modal for detailed country analysis
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="country-modal-title")),
        dbc.ModalBody([
            dbc.Tabs([
                dbc.Tab(label="Energy Profile", tab_id="tab-profile", children=[
                    dcc.Loading(
                        id="modal-profile-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="modal-energy-profile", style={"height": "400px"})
                        ]
                    ),
                    html.Div(id="modal-energy-highlights", className="mt-3")
                ]),
                dbc.Tab(label="Historical Trends", tab_id="tab-trends", children=[
                    dcc.Loading(
                        id="modal-trends-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="modal-historical-trends", style={"height": "400px"})
                        ]
                    )
                ]),
                dbc.Tab(label="Comparison", tab_id="tab-comparison", children=[
                    html.Div([
                        html.Label("Compare With:"),
                        dcc.Dropdown(
                            id="modal-comparison-country",
                            placeholder="Select a country to compare"
                        ),
                    ], className="mb-3"),
                    dcc.Loading(
                        id="modal-comparison-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="modal-comparison-chart", style={"height": "400px"})
                        ]
                    )
                ])
            ], id="country-tabs", active_tab="tab-profile")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-country-modal", className="ms-auto")
        )
    ], id="country-detail-modal", size="xl"),
    
    # Footer with timestamp
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(f"Last Updated: {datetime.now().strftime('%B %d, %Y')}", 
                   className="text-muted text-center small")
        ])
    ])
], fluid=True)

# CALLBACKS

# Update metric description
@app.callback(
    Output("metric-description", "children"),
    Input("metric-dropdown", "value")
)
def update_metric_description(selected_metric):
    return metric_descriptions.get(selected_metric, "")


# Toggle comparison controls visibility
@app.callback(
    Output("comparison-controls", "style"),
    Input("analysis-mode", "value")
)
def toggle_comparison_controls(analysis_mode):
    if analysis_mode == "comparison":
        return {"display": "block"}
    return {"display": "none"}

# Toggle regional stats visibility
@app.callback(
    Output("regional-stats-collapse", "is_open"),
    Input("toggle-regional-stats", "n_clicks"),
    State("regional-stats-collapse", "is_open")
)
def toggle_regional_stats(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Toggle trend analysis visibility
@app.callback(
    Output("trend-analysis-collapse", "is_open"),
    Input("toggle-trend-analysis", "n_clicks"),
    State("trend-analysis-collapse", "is_open")
)
def toggle_trend_analysis(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Toggle energy breakdown visibility
@app.callback(
    Output("energy-breakdown-collapse", "is_open"),
    Input("toggle-energy-breakdown", "n_clicks"),
    State("energy-breakdown-collapse", "is_open")
)
def toggle_energy_breakdown(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Toggle statistical summary visibility
@app.callback(
    Output("stats-summary-collapse", "is_open"),
    Input("toggle-stats-summary", "n_clicks"),
    State("stats-summary-collapse", "is_open")
)
def toggle_stats_summary(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Country modal controls
@app.callback(
    Output("country-detail-modal", "is_open"),
    Input("selected-country", "children"),
    Input("close-country-modal", "n_clicks"),
    State("country-detail-modal", "is_open")
)
def toggle_country_modal(selected_country, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if prop_id == "selected-country" and selected_country:
        return True
    elif prop_id == "close-country-modal":
        return False
    return is_open

# Update country modal title and dropdown options
@app.callback(
    [Output("country-modal-title", "children"),
     Output("modal-comparison-country", "options"),
     Output("modal-comparison-country", "value")],
    [Input("selected-country", "children")]
)
def update_country_modal_title(selected_country):
    if not selected_country:
        return "Country Details", [], None
    
    # Get list of countries for comparison dropdown
    countries = sorted(df["country"].unique())
    country_options = [{"label": country, "value": country} for country in countries if country != selected_country]
    
    # Return the title and dropdown options
    return f"{selected_country} Energy Profile", country_options, None

# Main callback to update all visualizations
@app.callback(
    [Output("choropleth-map", "figure"),
     Output("time-series", "figure"),
     Output("energy-mix", "figure"),
     Output("scatter-plot", "figure"),
     Output("map-title", "children"),
     Output("time-series-title", "children"),
     Output("energy-mix-title", "children"),
     Output("selected-country", "children"),
     Output("regional-stats", "children"),
     Output("trend-analysis", "children"),
     Output("energy-breakdown", "children"),
     Output("stats-summary", "children"),
     Output("last-update", "children")],
    [Input("year-range-slider", "value"),
     Input("metric-dropdown", "value"),
     Input("region-filter", "value"),
     Input("analysis-mode", "value"),
     Input("comparison-year", "value"),
     Input("choropleth-map", "clickData"),
     Input("scatter-plot", "clickData"),
     Input("clear-selection", "n_clicks")]
)
def update_figures(year_range, selected_metric, region_filter, analysis_mode, 
                   comparison_year, map_click, scatter_click, clear_clicks):
    """Update all visualizations based on user interactions."""
    ctx = callback_context
    
    # Determine which input triggered the callback
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    # Initialize selected country
    selected_country = None
    
    # Parse years
    start_year, end_year = year_range
    current_year = end_year
    
    # Update selected country based on triggers
    if triggered_id == "choropleth-map" and map_click:
        selected_country = map_click["points"][0]["customdata"][0]
    elif triggered_id == "scatter-plot" and scatter_click:
        selected_country = scatter_click["points"][0]["customdata"][0]
    elif triggered_id == "clear-selection":
        selected_country = None
    
    # Filter data by region if specified
    if region_filter != "all":
        filtered_df = df[df["region"] == region_filter]
    else:
        filtered_df = df
    
    # Filter data by year range
    filtered_df = filtered_df[(filtered_df["year"] >= start_year) & (filtered_df["year"] <= end_year)]
    
    # Get data for the current/end year
    current_year_data = filtered_df[filtered_df["year"] == current_year]
    
    # Get metric label for titles
    metric_labels = {metric["value"]: metric["label"] for metric in available_metrics}
    metric_label = metric_labels[selected_metric]
    
    # Prepare for comparison mode
    if analysis_mode == "comparison" and comparison_year != current_year:
        comparison_data = filtered_df[filtered_df["year"] == comparison_year]
        
        # Calculate change
        comparison_merged = current_year_data.merge(
            comparison_data[["country", "iso_code", selected_metric]],
            on=["country", "iso_code"],
            suffixes=("", "_prev")
        )
        
        comparison_merged["change"] = comparison_merged[selected_metric] - comparison_merged[f"{selected_metric}_prev"]
        comparison_merged["pct_change"] = np.where(
            comparison_merged[f"{selected_metric}_prev"] != 0,
            (comparison_merged["change"] / comparison_merged[f"{selected_metric}_prev"]) * 100,
            0
        )
        
        map_data = comparison_merged
        map_color_column = "change"
        choropleth_title = f"Change in {metric_label} ({comparison_year} to {current_year})"
        color_scale = px.colors.diverging.RdBu_r
    else:
        map_data = current_year_data
        map_color_column = selected_metric
        choropleth_title = f"{metric_label} ({current_year})"
        color_scale = color_scales[selected_metric]
    
    # Create Choropleth Map with custom hover template
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>" +
        f"{metric_label}: %{{z:,.1f}}<br>" +
        "GDP: $%{customdata[1]:,.0f}<br>" +
        "Population: %{customdata[2]:,.0f}<br>" +
        "<extra></extra>"
    )
    
    # Create Choropleth Map
    map_fig = px.choropleth(
        map_data,
        locations="iso_code",
        color=map_color_column,
        hover_name="country",
        custom_data=["country", "gdp", "population"],
        color_continuous_scale=color_scale,
        projection="natural earth",
    )
    
    map_fig.update_layout(
        title=choropleth_title,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            coastlinecolor="lightgray",
            landcolor="white",
            oceancolor="lightblue"
        ),
        coloraxis_colorbar=dict(
            title=metric_label if analysis_mode != "comparison" else f"Change in {metric_label}"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )
    
    map_fig.update_traces(
        hovertemplate=hovertemplate
    )
    
    # Highlight selected country on map
    if selected_country and selected_country in map_data["country"].values:
        selected_iso = map_data[map_data["country"] == selected_country]["iso_code"].iloc[0]
        map_fig.add_trace(
            go.Choropleth(
                locations=[selected_iso],
                z=[1],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                hoverinfo="skip",
                marker_line_color="black",
                marker_line_width=3
            )
        )
    
    # Create Time Series Chart
    if selected_country and selected_country in filtered_df["country"].values:
        country_data = filtered_df[filtered_df["country"] == selected_country]
        time_series_title = f"{metric_label} Trend for {selected_country}"
        
        # Create multi-line plot with multiple metrics for the selected country
        time_series_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Primary Y-axis: Selected metric
        time_series_fig.add_trace(
            go.Scatter(
                x=country_data["year"],
                y=country_data[selected_metric],
                name=metric_label,
                line=dict(color="#1f77b4", width=3),
                mode="lines+markers"
            ),
            secondary_y=False
        )
        
        # Secondary Y-axis: Energy per capita
        time_series_fig.add_trace(
            go.Scatter(
                x=country_data["year"],
                y=country_data["energy_per_capita"],
                name="Energy per Capita",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                mode="lines"
            ),
            secondary_y=True
        )
        
        time_series_fig.update_layout(
            title=time_series_title,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
        
        time_series_fig.update_xaxes(title_text="Year")
        time_series_fig.update_yaxes(title_text=metric_label, secondary_y=False)
        time_series_fig.update_yaxes(title_text="Energy per Capita (kWh)", secondary_y=True)
        
    else:
        # Global/regional average trend
        if region_filter != "all":
            avg_title = f"{region_filter} Average {metric_label} Trend"
        else:
            avg_title = f"Global Average {metric_label} Trend"
        
        # Group by year and calculate averages
        yearly_avg = filtered_df.groupby("year")[selected_metric].mean().reset_index()
        
        time_series_fig = px.line(
            yearly_avg,
            x="year",
            y=selected_metric,
            title=avg_title
        )
        
        # Add global range (min-max) as a shaded area
        yearly_min = filtered_df.groupby("year")[selected_metric].min().reset_index()
        yearly_max = filtered_df.groupby("year")[selected_metric].max().reset_index()
        
        time_series_fig.add_trace(
            go.Scatter(
                x=yearly_max["year"].tolist() + yearly_min["year"].tolist()[::-1],
                y=yearly_max[selected_metric].tolist() + yearly_min[selected_metric].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0, 114, 178, 0.1)",
                line=dict(width=0),
                name="Min-Max Range",
                showlegend=True
            )
        )
        
        time_series_fig.update_layout(
            title=avg_title,
            xaxis_title="Year",
            yaxis_title=metric_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
    
    time_series_title = time_series_fig.layout.title.text
    
    # Create Energy Mix Chart (Stacked area or pie chart)
    energy_types = ["renewables_share_energy", "fossil_share_energy", "nuclear_share_energy"]
    energy_labels = ["Renewable", "Fossil Fuels", "Nuclear"]
    energy_colors = ["#2ca02c", "#d62728", "#9467bd"]
    
    if selected_country and selected_country in filtered_df["country"].values:
        country_year_data = filtered_df[(filtered_df["country"] == selected_country) & 
                                    (filtered_df["year"] == current_year)]
        energy_mix_title = f"Energy Mix for {selected_country} ({current_year})"
        
        if not country_year_data.empty:
            # Create stacked bar chart for energy mix over time
            country_data = filtered_df[filtered_df["country"] == selected_country]
            
            # Sort by year
            country_data_sorted = country_data.sort_values("year")
            
            # Create a figure with secondary Y axis
            energy_mix_fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add stacked area chart for energy mix
            for i, energy_type in enumerate(energy_types):
                energy_mix_fig.add_trace(
                    go.Scatter(
                        x=country_data_sorted["year"],
                        y=country_data_sorted[energy_type],
                        name=energy_labels[i],
                        mode="lines",
                        stackgroup="one",
                        groupnorm="percent",
                        line=dict(width=0.5, color=energy_colors[i])
                    ),
                    secondary_y=False
                )
            
            # Add line for total energy consumption
            energy_mix_fig.add_trace(
                go.Scatter(
                    x=country_data_sorted["year"],
                    y=country_data_sorted["primary_energy_consumption"],
                    name="Total Energy (TWh)",
                    mode="lines",
                    line=dict(color="black", width=2)
                ),
                secondary_y=True
            )
            
            energy_mix_fig.update_layout(
                title=energy_mix_title,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=40, b=0),
                height=400
            )
            
            energy_mix_fig.update_xaxes(title_text="Year")
            energy_mix_fig.update_yaxes(title_text="Share of Energy Mix (%)", secondary_y=False)
            energy_mix_fig.update_yaxes(title_text="Total Energy (TWh)", secondary_y=True)
            
        else:
            # Default empty pie chart if no data
            values = [0, 0, 0]
            energy_mix_fig = go.Figure(data=[
                go.Pie(
                    labels=energy_labels,
                    values=values,
                    marker_colors=energy_colors,
                    textinfo="label+percent",
                    hole=0.3
                )
            ])
            energy_mix_fig.update_layout(title=energy_mix_title)
    else:
        # Regional/global average
        if region_filter != "all":
            region_data = filtered_df[(filtered_df["region"] == region_filter) & 
                                   (filtered_df["year"] == current_year)]
            energy_mix_title = f"{region_filter} Energy Mix ({current_year})"
        else:
            region_data = filtered_df[filtered_df["year"] == current_year]
            energy_mix_title = f"Global Energy Mix ({current_year})"
        
        # Calculate average values
        values = [region_data[energy_type].mean() for energy_type in energy_types]
        
        energy_mix_fig = go.Figure(data=[
            go.Pie(
                labels=energy_labels,
                values=values,
                marker_colors=energy_colors,
                textinfo="label+percent",
                hole=0.3
            )
        ])
        
        energy_mix_fig.update_layout(
            title=energy_mix_title,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
    
    # Create Scatter Plot (CO2 vs Energy Per Capita)
    scatter_data = filtered_df[filtered_df["year"] == current_year].copy()
    
    # Create a size reference column normalized for better visualization
    scatter_data["size_ref"] = np.sqrt(scatter_data["population"]) / 100
    scatter_data["size_ref"] = np.clip(scatter_data["size_ref"], 5, 50)
    
    scatter_fig = px.scatter(
        scatter_data,
        x="energy_per_capita",
        y="co2",
        size="size_ref",
        color="renewables_share_energy",
        hover_name="country",
        size_max=50,
        color_continuous_scale=color_scales["renewables_share_energy"],
        custom_data=["country", "fossil_share_energy", "renewables_share_energy", "nuclear_share_energy", "population"]
    )
    
    # Create custom hover template
    scatter_fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "Energy per Capita: %{x:,.0f} kWh<br>" +
            "CO2 Emissions: %{y:,.0f} kt<br>" +
            "Renewables: %{customdata[2]:.1f}%<br>" +
            "Fossil Fuels: %{customdata[1]:.1f}%<br>" +
            "Population: %{customdata[4]:,.0f}<br>" +
            "<extra></extra>"
        )
    )
    
    # Highlight selected country in scatter plot
    if selected_country and selected_country in scatter_data["country"].values:
        selected_data = scatter_data[scatter_data["country"] == selected_country]
        scatter_fig.add_trace(
            go.Scatter(
                x=[selected_data["energy_per_capita"].iloc[0]],
                y=[selected_data["co2"].iloc[0]],
                mode="markers",
                marker=dict(
                    size=20,
                    color="rgba(0,0,0,0)",
                    line=dict(color="black", width=2)
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )
    
    # Add regression line
    if not scatter_data.empty:
        # Add trendline
        z = np.polyfit(scatter_data["energy_per_capita"], scatter_data["co2"], 1)
        p = np.poly1d(z)
        
        x_range = [scatter_data["energy_per_capita"].min(), scatter_data["energy_per_capita"].max()]
        y_range = [p(x_range[0]), p(x_range[1])]
        
        scatter_fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
                name="Trend Line",
                hoverinfo="skip"
            )
        )
        
        # Add annotation for correlation coefficient
        correlation = scatter_data["energy_per_capita"].corr(scatter_data["co2"])
        scatter_fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"Correlation: {correlation:.2f}",
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    scatter_fig.update_layout(
        title="CO₂ Emissions vs. Energy Consumption (Size = Population, Color = Renewables %)",
        xaxis_title="Energy Per Capita (kWh)",
        yaxis_title="CO₂ Emissions (kt)",
        coloraxis_colorbar_title="Renewable %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=0),
        height=400
    )
    
    # Create Regional Stats HTML
    if region_filter != "all":
        region_data = filtered_df[(filtered_df["region"] == region_filter) & 
                                (filtered_df["year"] == current_year)]
        stats_title = f"{region_filter} Statistics ({current_year})"
    else:
        region_data = filtered_df[filtered_df["year"] == current_year]
        stats_title = f"Global Statistics ({current_year})"
    
    # Calculate statistics
    total_population = region_data["population"].sum()
    avg_renewable = region_data["renewables_share_energy"].mean()
    avg_fossil = region_data["fossil_share_energy"].mean()
    avg_co2 = region_data["co2"].mean()
    total_energy = region_data["primary_energy_consumption"].sum()
    
    # Format regional stats
    regional_stats = html.Div([
        html.H5(stats_title, className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Population"),
                    html.P(f"{total_population:,.0f}", className="lead text-primary")
                ])
            ]), width=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Avg. Renewable %"),
                    html.P(f"{avg_renewable:.1f}%", className="lead text-success")
                ])
            ]), width=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Avg. Fossil Fuel %"),
                    html.P(f"{avg_fossil:.1f}%", className="lead text-danger")
                ])
            ]), width=4)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Avg. CO₂ Emissions"),
                    html.P(f"{avg_co2:,.0f} kt", className="lead text-warning")
                ])
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total Energy Consumption"),
                    html.P(f"{total_energy:,.0f} TWh", className="lead text-info")
                ])
            ]), width=6)
        ])
    ])
    
    # Create Trend Analysis HTML
    if selected_country and selected_country in filtered_df["country"].values:
        country_data = filtered_df[filtered_df["country"] == selected_country]
        
        # Sort by year
        country_data_sorted = country_data.sort_values("year")
        
        # Calculate trends (comparing oldest and newest available years)
        first_year_data = country_data_sorted.iloc[0]
        last_year_data = country_data_sorted.iloc[-1]
        
        first_year = first_year_data["year"]
        last_year = last_year_data["year"]
        
        renewable_change = last_year_data["renewables_share_energy"] - first_year_data["renewables_share_energy"]
        fossil_change = last_year_data["fossil_share_energy"] - first_year_data["fossil_share_energy"]
        co2_change = (last_year_data["co2"] / first_year_data["co2"] - 1) * 100 if first_year_data["co2"] > 0 else 0
        
        trend_title = f"{selected_country} Trends ({first_year} to {last_year})"
        
        # Determine trend direction
        renewable_trend = "↑ Increasing" if renewable_change > 0 else "↓ Decreasing" if renewable_change < 0 else "→ Stable"
        fossil_trend = "↑ Increasing" if fossil_change > 0 else "↓ Decreasing" if fossil_change < 0 else "→ Stable"
        co2_trend = "↑ Increasing" if co2_change > 0 else "↓ Decreasing" if co2_change < 0 else "→ Stable"
        
        # Trend colors
        renewable_color = "success" if renewable_change > 0 else "danger" if renewable_change < 0 else "secondary"
        fossil_color = "danger" if fossil_change > 0 else "success" if fossil_change < 0 else "secondary"
        co2_color = "danger" if co2_change > 0 else "success" if co2_change < 0 else "secondary"
        
        trend_analysis = html.Div([
            html.H5(trend_title, className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Renewable Energy Trend"),
                        html.P([
                            html.Span(f"{renewable_change:+.1f}%", className=f"lead text-{renewable_color}"),
                            html.Span(f" ({renewable_trend})", className="ms-2 small")
                        ])
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Fossil Fuel Trend"),
                        html.P([
                            html.Span(f"{fossil_change:+.1f}%", className=f"lead text-{fossil_color}"),
                            html.Span(f" ({fossil_trend})", className="ms-2 small")
                        ])
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("CO₂ Emissions Trend"),
                        html.P([
                            html.Span(f"{co2_change:+.1f}%", className=f"lead text-{co2_color}"),
                            html.Span(f" ({co2_trend})", className="ms-2 small")
                        ])
                    ])
                ]), width=4)
            ]),
            html.Div([
                html.P([
                    "Transition Speed Indicator: ",
                    html.Span(
                        "Fast" if renewable_change > 10 else 
                        "Moderate" if renewable_change > 5 else 
                        "Slow" if renewable_change > 0 else 
                        "Negative" if renewable_change < 0 else "Stable",
                        className=f"fw-bold text-{'success' if renewable_change > 0 else 'danger' if renewable_change < 0 else 'secondary'}"
                    )
                ], className="mt-3 small")
            ])
        ])
    else:
        # Global trend analysis
        global_data = filtered_df.groupby("year")[["renewables_share_energy", "fossil_share_energy", "co2"]].mean().reset_index()
        
        # Sort by year
        global_data_sorted = global_data.sort_values("year")
        
        # Calculate trends
        first_year_data = global_data_sorted.iloc[0]
        last_year_data = global_data_sorted.iloc[-1]
        
        first_year = first_year_data["year"]
        last_year = last_year_data["year"]
        
        renewable_change = last_year_data["renewables_share_energy"] - first_year_data["renewables_share_energy"]
        fossil_change = last_year_data["fossil_share_energy"] - first_year_data["fossil_share_energy"]
        co2_change = (last_year_data["co2"] / first_year_data["co2"] - 1) * 100 if first_year_data["co2"] > 0 else 0
        
        trend_title = f"Global Trends ({first_year} to {last_year})"
        
        # Determine trend direction
        renewable_trend = "↑ Increasing" if renewable_change > 0 else "↓ Decreasing" if renewable_change < 0 else "→ Stable"
        fossil_trend = "↑ Increasing" if fossil_change > 0 else "↓ Decreasing" if fossil_change < 0 else "→ Stable"
        co2_trend = "↑ Increasing" if co2_change > 0 else "↓ Decreasing" if co2_change < 0 else "→ Stable"
        
        # Trend colors
        renewable_color = "success" if renewable_change > 0 else "danger" if renewable_change < 0 else "secondary"
        fossil_color = "danger" if fossil_change > 0 else "success" if fossil_change < 0 else "secondary"
        co2_color = "danger" if co2_change > 0 else "success" if co2_change < 0 else "secondary"
        
        trend_analysis = html.Div([
            html.H5(trend_title, className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Renewable Energy Trend"),
                        html.P([
                            html.Span(f"{renewable_change:+.1f}%", className=f"lead text-{renewable_color}"),
                            html.Span(f" ({renewable_trend})", className="ms-2 small")
                        ])
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Fossil Fuel Trend"),
                        html.P([
                            html.Span(f"{fossil_change:+.1f}%", className=f"lead text-{fossil_color}"),
                            html.Span(f" ({fossil_trend})", className="ms-2 small")
                        ])
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("CO₂ Emissions Trend"),
                        html.P([
                            html.Span(f"{co2_change:+.1f}%", className=f"lead text-{co2_color}"),
                            html.Span(f" ({co2_trend})", className="ms-2 small")
                        ])
                    ])
                ]), width=4)
            ])
        ])
    
    # Create Energy Breakdown HTML
    if selected_country and selected_country in filtered_df["country"].values:
        country_year_data = filtered_df[(filtered_df["country"] == selected_country) & 
                                     (filtered_df["year"] == current_year)]
        
        if not country_year_data.empty:
            # Get detailed energy breakdown
            data = country_year_data.iloc[0]
            
            # Extract renewable breakdown
            hydro = data["hydro_share_energy"] if "hydro_share_energy" in data else 0
            wind = data["wind_share_energy"] if "wind_share_energy" in data else 0
            solar = data["solar_share_energy"] if "solar_share_energy" in data else 0
            biofuel = data["biofuel_share_energy"] if "biofuel_share_energy" in data else 0
            other_renewable = data["other_renewables_share_energy"] if "other_renewables_share_energy" in data else 0
            
            # Extract fossil breakdown
            coal = data["coal_share_energy"] if "coal_share_energy" in data else 0
            oil = data["oil_share_energy"] if "oil_share_energy" in data else 0
            gas = data["gas_share_energy"] if "gas_share_energy" in data else 0
            
            # Nuclear
            nuclear = data["nuclear_share_energy"] if "nuclear_share_energy" in data else 0
            
            energy_breakdown = html.Div([
                html.H5(f"{selected_country} Energy Breakdown ({current_year})", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Renewable Sources", className="text-success"),
                        dbc.Progress([
                            dbc.Progress(value=hydro, color="info", bar=True, label=f"Hydro: {hydro:.1f}%"),
                            dbc.Progress(value=wind, color="primary", bar=True, label=f"Wind: {wind:.1f}%"),
                            dbc.Progress(value=solar, color="warning", bar=True, label=f"Solar: {solar:.1f}%"),
                            dbc.Progress(value=biofuel, color="success", bar=True, label=f"Biofuel: {biofuel:.1f}%"),
                            dbc.Progress(value=other_renewable, color="secondary", bar=True, label=f"Other: {other_renewable:.1f}%", className="mb-1")
                        ], className="mb-3")
                    ], width=12),
                    dbc.Col([
                        html.H6("Fossil Fuels", className="text-danger"),
                        dbc.Progress([
                            dbc.Progress(value=coal, color="dark", bar=True, label=f"Coal: {coal:.1f}%"),
                            dbc.Progress(value=oil, color="danger", bar=True, label=f"Oil: {oil:.1f}%"),
                            dbc.Progress(value=gas, color="warning", bar=True, label=f"Gas: {gas:.1f}%")
                        ], className="mb-3")
                    ], width=12),
                    dbc.Col([
                        html.H6("Nuclear", className="text-primary"),
                        dbc.Progress(value=nuclear, color="purple", label=f"Nuclear: {nuclear:.1f}%")
                    ], width=12)
                ])
            ])
        else:
            energy_breakdown = html.Div([
                html.P("No detailed energy breakdown available for this country and year.", 
                       className="text-muted")
            ])
    else:
        energy_breakdown = html.Div([
            html.P("Select a country to view detailed energy breakdown.", 
                   className="text-muted")
        ])
    
    # Create Statistical Summary HTML
    if not scatter_data.empty:
        # Calculate statistics for scatter plot
        corr_co2_energy = scatter_data["co2"].corr(scatter_data["energy_per_capita"])
        corr_co2_renewable = scatter_data["co2"].corr(scatter_data["renewables_share_energy"])
        corr_renewable_fossil = scatter_data["renewables_share_energy"].corr(scatter_data["fossil_share_energy"])
        
        # Top countries by renewables
        top_renewable = scatter_data.nlargest(5, "renewables_share_energy")[["country", "renewables_share_energy"]]
        top_renewable_list = [f"{row['country']} ({row['renewables_share_energy']:.1f}%)" for _, row in top_renewable.iterrows()]
        
        # Top CO2 emitters
        top_co2 = scatter_data.nlargest(5, "co2")[["country", "co2"]]
        top_co2_list = [f"{row['country']} ({row['co2']:,.0f} kt)" for _, row in top_co2.iterrows()]
        
        stats_summary = html.Div([
            html.H5(f"Statistical Summary ({current_year})", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("Correlation Analysis"),
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Variables"),
                                html.Th("Correlation"),
                                html.Th("Strength")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("CO₂ vs Energy Per Capita"),
                                html.Td(f"{corr_co2_energy:.2f}"),
                                html.Td(html.Span(
                                    "Strong Positive" if corr_co2_energy > 0.7 else
                                    "Moderate Positive" if corr_co2_energy > 0.3 else
                                    "Weak Positive" if corr_co2_energy > 0 else
                                    "Strong Negative" if corr_co2_energy < -0.7 else
                                    "Moderate Negative" if corr_co2_energy < -0.3 else
                                    "Weak Negative",
                                    className=f"text-{'success' if corr_co2_energy > 0 else 'danger'} fw-bold"
                                ))
                            ]),
                            html.Tr([
                                html.Td("CO₂ vs Renewable Share"),
                                html.Td(f"{corr_co2_renewable:.2f}"),
                                html.Td(html.Span(
                                    "Strong Positive" if corr_co2_renewable > 0.7 else
                                    "Moderate Positive" if corr_co2_renewable > 0.3 else
                                    "Weak Positive" if corr_co2_renewable > 0 else
                                    "Strong Negative" if corr_co2_renewable < -0.7 else
                                    "Moderate Negative" if corr_co2_renewable < -0.3 else
                                    "Weak Negative",
                                    className=f"text-{'success' if corr_co2_renewable < 0 else 'danger'} fw-bold"
                                ))
                            ]),
                            html.Tr([
                                html.Td("Renewable vs Fossil Share"),
                                html.Td(f"{corr_renewable_fossil:.2f}"),
                                html.Td(html.Span(
                                    "Strong Negative" if corr_renewable_fossil < -0.7 else
                                    "Moderate Negative" if corr_renewable_fossil < -0.3 else
                                    "Weak Negative" if corr_renewable_fossil < 0 else
                                    "Strong Positive" if corr_renewable_fossil > 0.7 else
                                    "Moderate Positive" if corr_renewable_fossil > 0.3 else
                                    "Weak Positive",
                                    className=f"text-{'success' if corr_renewable_fossil < 0 else 'danger'} fw-bold"
                                ))
                            ])
                        ])
                    ], className="table table-sm table-striped")
                ], width=6),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Top Renewable Energy Leaders"),
                            html.Ol([html.Li(country) for country in top_renewable_list])
                        ], width=12, className="mb-3"),
                        dbc.Col([
                            html.H6("Top CO₂ Emitters"),
                            html.Ol([html.Li(country) for country in top_co2_list])
                        ], width=12)
                    ])
                ], width=6)
            ])
        ])
    else:
        stats_summary = html.Div([
            html.P("Insufficient data for statistical analysis.", 
                   className="text-muted")
        ])
    
    # Update last update timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    last_update = f"Last updated: {current_time}"
    
    return (map_fig, time_series_fig, energy_mix_fig, scatter_fig, 
            choropleth_title, time_series_title, energy_mix_title, selected_country,
            regional_stats, trend_analysis, energy_breakdown, stats_summary,
            last_update)

# Callbacks for country detail modal
@app.callback(
    [Output("modal-energy-profile", "figure"),
     Output("modal-energy-highlights", "children")],
    [Input("selected-country", "children")]
)
def update_country_modal_profile(selected_country):
    """Update the energy profile tab in the country detail modal."""
    if not selected_country or selected_country not in df["country"].values:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Country Selected",
            annotations=[dict(
                text="Select a country on the map to view details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return empty_fig, html.P("No country selected.")
    
    # Get the most recent data for the selected country
    country_data = df[df["country"] == selected_country]
    latest_year = country_data["year"].max()
    latest_data = country_data[country_data["year"] == latest_year].iloc[0]
    
    # Create energy profile figure (radar chart)
    energy_metrics = {
        "Renewable Energy": latest_data["renewables_share_energy"],
        "Energy Efficiency": 100 - min(100, latest_data["energy_per_capita"] / 10000 * 100),  # Normalize
        "Low Carbon": latest_data["renewables_share_energy"] + latest_data["nuclear_share_energy"],
        "Energy Independence": 50,  # Placeholder value
        "CO₂ Efficiency": 100 - min(100, latest_data["co2"] / latest_data["energy_per_capita"] * 0.01)  # Normalize
    }
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(energy_metrics.values()),
        theta=list(energy_metrics.keys()),
        fill='toself',
        name=selected_country,
        line_color='rgb(31, 119, 180)'
    ))
    
    # Add global average for comparison
    global_avg_data = df[df["year"] == latest_year]
    global_metrics = {
        "Renewable Energy": global_avg_data["renewables_share_energy"].mean(),
        "Energy Efficiency": 100 - min(100, global_avg_data["energy_per_capita"].mean() / 10000 * 100),
        "Low Carbon": global_avg_data["renewables_share_energy"].mean() + global_avg_data["nuclear_share_energy"].mean(),
        "Energy Independence": 50,  # Placeholder value
        "CO₂ Efficiency": 100 - min(100, global_avg_data["co2"].mean() / global_avg_data["energy_per_capita"].mean() * 0.01)
    }
    
    fig.add_trace(go.Scatterpolar(
        r=list(global_metrics.values()),
        theta=list(global_metrics.keys()),
        fill='toself',
        name='Global Average',
        line_color='rgba(255, 99, 71, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title=f"{selected_country} Energy Profile ({latest_year})"
    )
    
    # Create highlights section
    highlights = html.Div([
        html.H5("Key Highlights", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Renewable Energy", className="card-title"),
                        html.P([
                            f"{latest_data['renewables_share_energy']:.1f}% ",
                            html.Span(
                                "(Above Global Average)" if latest_data['renewables_share_energy'] > global_metrics["Renewable Energy"] else "(Below Global Average)",
                                className=f"text-{'success' if latest_data['renewables_share_energy'] > global_metrics['Renewable Energy'] else 'danger'} small"
                            )
                        ]),
                        html.P([
                            html.Strong("Primary Sources: "),
                            "Hydro, Wind, Solar" if 'hydro_share_energy' in latest_data and latest_data['hydro_share_energy'] > 0 else "Data Unavailable"
                        ], className="small mb-0")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Energy Consumption", className="card-title"),
                        html.P([
                            f"{latest_data['energy_per_capita']:,.0f} kWh per capita ",
                            html.Span(
                                "(Above Global Average)" if latest_data['energy_per_capita'] > global_avg_data['energy_per_capita'].mean() else "(Below Global Average)",
                                className=f"text-{'warning' if latest_data['energy_per_capita'] > global_avg_data['energy_per_capita'].mean() else 'success'} small"
                            )
                        ]),
                        html.P([
                            html.Strong("Efficiency Rating: "),
                            f"{energy_metrics['Energy Efficiency']:.1f}/100"
                        ], className="small mb-0")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("CO₂ Emissions", className="card-title"),
                        html.P([
                            f"{latest_data['co2']:,.0f} kt ",
                            html.Span(
                                "(Above Global Average)" if latest_data['co2'] > global_avg_data['co2'].mean() else "(Below Global Average)",
                                className=f"text-{'danger' if latest_data['co2'] > global_avg_data['co2'].mean() else 'success'} small"
                            )
                        ]),
                        html.P([
                            html.Strong("Carbon Intensity: "),
                            f"{latest_data['co2'] / latest_data['energy_per_capita']:.2f} kt/kWh"
                        ], className="small mb-0")
                    ])
                ])
            ], width=4)
        ])
    ])
    
    return fig, highlights

@app.callback(
    Output("modal-historical-trends", "figure"),
    [Input("selected-country", "children")]
)
def update_country_modal_historical_trends(selected_country):
    """Update the historical trends tab in the country detail modal."""
    if not selected_country or selected_country not in df["country"].values:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Country Selected",
            annotations=[dict(
                text="Select a country on the map to view details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return empty_fig
    
    # Get historical data for the selected country
    country_data = df[df["country"] == selected_country].sort_values("year")
    
    # Create figure with secondary Y axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=country_data["year"],
            y=country_data["renewables_share_energy"],
            name="Renewable Share (%)",
            line=dict(color="#2ca02c", width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=country_data["year"],
            y=country_data["fossil_share_energy"],
            name="Fossil Fuel Share (%)",
            line=dict(color="#d62728", width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=country_data["year"],
            y=country_data["co2"],
            name="CO₂ Emissions (kt)",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ),
        secondary_y=True
    )
    
    # Add figure title
    fig.update_layout(
        title=f"{selected_country} Historical Energy Trends",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        xaxis_title="Year"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Energy Share (%)", secondary_y=False)
    fig.update_yaxes(title_text="CO₂ Emissions (kt)", secondary_y=True)
    
    return fig

@app.callback(
    Output("modal-comparison-chart", "figure"),
    [Input("selected-country", "children"),
     Input("modal-comparison-country", "value")]
)
def update_country_modal_comparison(selected_country, comparison_country):
    """Update the comparison tab in the country detail modal."""
    if not selected_country or not comparison_country:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Select Countries to Compare",
            annotations=[dict(
                text="Select another country to compare with",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return empty_fig
    
    # Get data for both countries
    country1_data = df[df["country"] == selected_country].sort_values("year")
    country2_data = df[df["country"] == comparison_country].sort_values("year")
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Renewable Energy Share (%)",
            "Fossil Fuel Share (%)",
            "CO₂ Emissions (kt)",
            "Energy Per Capita (kWh)"
        )
    )
    
    # Add traces for renewable energy
    fig.add_trace(
        go.Scatter(
            x=country1_data["year"],
            y=country1_data["renewables_share_energy"],
            name=f"{selected_country} Renewables",
            line=dict(color="#1f77b4")
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=country2_data["year"],
            y=country2_data["renewables_share_energy"],
            name=f"{comparison_country} Renewables",
            line=dict(color="#1f77b4", dash="dash")
        ),
        row=1, col=1
    )
    
    # Add traces for fossil fuels
    fig.add_trace(
        go.Scatter(
            x=country1_data["year"],
            y=country1_data["fossil_share_energy"],
            name=f"{selected_country} Fossil",
            line=dict(color="#d62728")
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=country2_data["year"],
            y=country2_data["fossil_share_energy"],
            name=f"{comparison_country} Fossil",
            line=dict(color="#d62728", dash="dash")
        ),
        row=1, col=2
    )
    
    # Add traces for CO2 emissions
    fig.add_trace(
        go.Scatter(
            x=country1_data["year"],
            y=country1_data["co2"],
            name=f"{selected_country} CO₂",
            line=dict(color="#ff7f0e")
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=country2_data["year"],
            y=country2_data["co2"],
            name=f"{comparison_country} CO₂",
            line=dict(color="#ff7f0e", dash="dash")
        ),
        row=2, col=1
    )
    
    # Add traces for energy per capita
    fig.add_trace(
        go.Scatter(
            x=country1_data["year"],
            y=country1_data["energy_per_capita"],
            name=f"{selected_country} Energy",
            line=dict(color="#2ca02c")
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=country2_data["year"],
            y=country2_data["energy_per_capita"],
            name=f"{comparison_country} Energy",
            line=dict(color="#2ca02c", dash="dash")
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Comparison: {selected_country} vs {comparison_country}",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        showlegend=False
    )
    
    return fig

# Run the app
if __name__ == "__main__":
    # Load data
    df = load_data("energy-data.csv")
    app.run(debug=True)