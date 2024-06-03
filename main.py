import ee
import geopandas as gpd
import pandas as pd
import io
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
import base64
from shapely.geometry import Point
from geopandas.tools import sjoin
import plotly.express as px
import dash_leaflet as dl
import shapefile
import json

ee.Initialize(project='ee-my-username121')

app = dash.Dash(__name__, external_stylesheets=['style.css'])

app.layout = html.Div(className='app-container', children=[
    html.H1("Приложение для идентификации гарей и анализа восстановления растительности", className='app-title'),
    dcc.Store(id='capture-date-store'),
    dcc.Store(id='output-data1-store'),
    html.Div(id='output-data-upload', className='upload-output'),
    html.Div(className='date-picker-container', children=[
        html.Label("Выберите диапазон дат:", className='date-picker-label'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date_placeholder_text="Начальная дата",
            end_date_placeholder_text="Конечная дата",
            start_date='',
            end_date='',
            className='date-picker'
        ),
    ]),
    html.Div(className='coordinates-container', children=[
        html.Label("Введите координаты:", className='coordinates-label'),
        dcc.Input(
            id='latitude-input',
            type='number',
            placeholder='Широта',
            className='latitude-input'
        ),
        dcc.Input(
            id='longitude-input',
            type='number',
            placeholder='Долгота',
            className='longitude-input'
        ),
    ]),
    html.Div(className='bands-container', children=[
        html.Label("Выберите каналы:", className='bands-label'),
        dcc.Dropdown(
            id='band-select',
            options=[
                {'label': 'B1', 'value': 'B1'},
                {'label': 'B2', 'value': 'B2'},
                {'label': 'B3', 'value': 'B3'},
                {'label': 'B4', 'value': 'B4'},
                {'label': 'B5', 'value': 'B5'},
                {'label': 'B6', 'value': 'B6'},
                {'label': 'B7', 'value': 'B7'},
                {'label': 'B8', 'value': 'B8'},
                {'label': 'B11', 'value': 'B11'},
                {'label': 'B12', 'value': 'B12'},
            ],
            value=['B4', 'B3', 'B2'],
            multi=True,
            className='band-dropdown'
        ),
    ]),
    html.A(id='download-link-tiff', style={'display': 'none'}),
    html.Div(id='image-output', className='image-container'),
    dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Перетащите или выберите файл GeoJSON '
            ]),
            className='upload-container',
            multiple=False,
            style={'font-size': '16px',
                   'textAlign': 'center'}
    ),
    html.Button('Экспорт изображения (TIFF)', id='export-image-btn', className='export-btn', style={'font-size': '14px'}),
    html.Button('Экспортировать полигоны', id='export-btn', className='export-btn', style={'font-size': '14px'}),
    html.Button('Расчет пожаров', id='process-fire-data-btn', className='process-btn', style={'font-size': '14px'}),
    html.Button('Расчет NDVI', id='calculate-ndvi-btn', className='calculate-btn', style={'font-size': '14px'}),
    html.Button('Визуализация', id='update-map-btn', className='calculate-btn', style={'font-size': '14px'}),
    html.Div(id='output-data1', className='output-container'),
    html.Div(id='percentage-output', className='percentage-container'),
    html.Div(id='export-status', className='export-status'),
    html.Div(id='ndvi-output', className='ndvi-container'),
    html.Div(id='map-output', className='output-container1'),
    dcc.Graph(id='geojson-map', className='geojson-map')
])


@app.callback(
    Output('geojson-map', 'figure'),
    Input('upload-data', 'contents'),
)
def update_map(contents):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file_io = io.BytesIO(decoded)
    gdf = gpd.read_file(file_io)
    largest_polygon = gdf.geometry.area.idxmax()
    gdf = gdf.drop(index=largest_polygon)

    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        mapbox_style="carto-positron",
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        zoom=5,
    )
    return fig


def update_output(contents, filename, collection, start_date, end_date):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        file_io = io.BytesIO(decoded)
        polygons_gdf = gpd.read_file(file_io)

        ndvi_values = []
        for idx, row in polygons_gdf.iterrows():
            try:
                region_of_interest_geojson = row['geometry'].__geo_interface__
                region_of_interest_gee = ee.Geometry.Polygon(region_of_interest_geojson['coordinates'])

                # if start_date == '' or end_date == '':
                #     start_date = '2017-01-01'
                #     end_date = '2017-06-01'

                start_date_dt = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=330)
                end_date_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=360)

                collection3 = ee.ImageCollection(collection) \
                    .filterDate(start_date_dt, end_date_dt) \
                    .filterBounds(region_of_interest_gee)

                mean_ndvi_collection = collection3.map(lambda image: calculate_mean_ndvi(image, region_of_interest_gee))

                ndvi_value = mean_ndvi_collection.first().getInfo()
                ndvi_values.append(ndvi_value)

            except Exception as e:
                print(f"Error processing geometry {idx}: {e}")
                ndvi_values.append(None)

        polygons_gdf['NDVI'] = ndvi_values

        output_file_path = f'{filename}_with_ndvi.geojson'
        polygons_gdf.to_file(output_file_path, driver='GeoJSON')

        return html.Div([
            html.H5(f'Processed file: {filename}'),
            html.A('Download processed file', href=f'/{output_file_path}')
        ])

    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])


def calculate_mean_ndvi(image, region_of_interest_gee):
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    mean_ndvi = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region_of_interest_gee,
        scale=5000
    ).get('NDVI')

    feature = ee.Feature(
        None,
        {
            'system:time_start': image.get('system:time_start'),
            'SUN_ELEVATION': image.get('SUN_ELEVATION'),
            'NDVI': mean_ndvi
        }
    )

    return feature


@app.callback(
    Output('capture-date-store', 'data'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('latitude-input', 'value'),
     Input('longitude-input', 'value'),
     Input('band-select', 'value')]
)
def update_capture_date(start_date, end_date, latitude, longitude, bands):
    if start_date and end_date and latitude is not None and longitude is not None and bands is not None:
        point = ee.Geometry.Point([longitude, latitude])
        date_range = ee.DateRange(start_date, end_date)
        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
            .filterBounds(point) \
            .filterDate(date_range)

        image = ee.Image(collection.sort('system:time_start', False).first())
        capture_date = ee.Date(image.get('system:time_start')).format('yyyy-MM-dd').getInfo()
        return capture_date

    return None

def update_image(start_date, end_date, latitude, longitude, bands):
    if start_date and end_date and latitude is not None and longitude is not None and bands is not None:
        point = ee.Geometry.Point([longitude, latitude])
        date_range = ee.DateRange(start_date, end_date)

        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
            .filterBounds(point) \
            .filterDate(date_range)

        image = ee.Image(collection.sort('system:time_start', False).first())
        capture_date = ee.Date(image.get('system:time_start')).format('yyyy-MM-dd').getInfo()
        print(capture_date)
        image = image.select(bands)
        vis_params = {'min': 0, 'max': 3000}
        image_vis = image.visualize(**vis_params)
        image_url = image_vis.getThumbURL({'dimensions': "500x500"})

        return html.Img(src=image_url, style={'width': '50%'})
    else:
        return "Выберите даты, введите координаты и выберите каналы для просмотра изображения."


def calculate_nbr(latitude, longitude, start_date, end_date):
    if start_date and end_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_before = start_date - timedelta(days=360)
        end_date_before = end_date - timedelta(days=330)

        roi = ee.Geometry.Point([longitude, latitude])
        collection1 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA").filterBounds(roi).filterBounds(roi).filterDate(
            start_date, end_date).limit(5000)
        collection2 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA").filterBounds(roi).filterBounds(roi).filterDate(
            start_date_before, end_date_before).limit(5000)

        first_image1 = collection1.first()
        first_image2 = collection2.first()
        nbr_post = first_image1.normalizedDifference(['B7', 'B5']).rename('nbr_post')
        nbr_pre = first_image2.normalizedDifference(['B7', 'B5']).rename('nbr_pre')
        NBR = nbr_pre.subtract(nbr_post)
        return NBR
    else:
        return None


def export_image_tiff(n_clicks, start_date, end_date, latitude, longitude, bands):
    if n_clicks:
        point = ee.Geometry.Point([longitude, latitude])
        date_range = ee.DateRange(start_date, end_date)

        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
            .filterBounds(point) \
            .filterDate(date_range)

        image = ee.Image(collection.sort('system:time_start', False).first())
        selected_image = image.select(bands)

        output_path = 'satellite_image_export_test'

        task = ee.batch.Export.image.toDrive(
            image=selected_image,
            description='Satellite Image Export',
            fileFormat='GeoTIFF',
            fileNamePrefix='satellite_image',
            folder=output_path
        )

        task.start()

        return None

    return None


def calculate_ndvi_callback(contents, filename, start_date, end_date):
    if contents is None:
        return html.Div("Файл не загружен")

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        file_io = io.BytesIO(decoded)
        polygons_gdf = gpd.read_file(file_io)
        return update_output(contents, filename, 'LANDSAT/LC08/C01/T1_SR', start_date, end_date)

    except Exception as e:
        return html.Div([
            f'Произошел сбой в обработке файла для расчета NDVI: {str(e)}'
        ])

def export_image(n_clicks, start_date, end_date, latitude, longitude, bands):
    if n_clicks:
        nbr = calculate_nbr(latitude, longitude, start_date, end_date)
        if nbr is not None:
            threshold = 0.66
            fires = nbr.gt(threshold)

            fire_polygons = fires.reduceToVectors(
                geometryType='polygon',
                scale=5000,
                labelProperty='fire_id',
                eightConnected=True
            )

            output_path = 'fire_polygons_export_test'

            task = ee.batch.Export.table.toDrive(
                collection=fire_polygons,
                description='Fire Polygons Export',
                fileFormat='GeoJSON',
                fileNamePrefix='fire_polygons',
                folder=output_path
            )

            task.start()

            return html.Div([
                html.P('Полигоны экспортировано в GeoJSON!')
            ])
        else:
            return "Для экспорта изображения необходимо выбрать даты"
    return ""

@app.callback(
    Output('percentage-output', 'children'),
    [Input('process-fire-data-btn', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('capture-date-store', 'data'),
     State('output-data1-store', 'data')]  # Получение данных из Хранилища
)
def process_fire_data(n_clicks, filename, start_date, end_date, capture_date, output_data1_store):
    if n_clicks is not None and filename:
        try:
            num_rows = output_data1_store
            print(num_rows)
            start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y/%m/%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y/%m/%d')
            capture_date = datetime.strptime(capture_date, '%Y-%m-%d').strftime('%Y/%m/%d')
            print(capture_date)
            gdf_geojson = gpd.read_file(filename)
            df_csv = pd.read_csv('final_rosleshoz_index.csv')

            geometry = [Point(xy) for xy in zip(df_csv['lon'], df_csv['lat'])]
            gdf_csv = gpd.GeoDataFrame(df_csv, crs='epsg:4326', geometry=geometry)

            gdf_merged = sjoin(gdf_csv, gdf_geojson, how='left', predicate='intersects')
            gdf_merged.to_csv('final_result3.csv', index=False)
            file_path = 'final_result3.csv'

            df = pd.read_csv(file_path)
            df['date_start'] = pd.to_datetime(df['date_start'])
            df['date_end'] = pd.to_datetime(df['date_end'])
            date_start_minus_10_days = df['date_start'] - pd.DateOffset(days=10)
            date_end_plus_10_days = df['date_end'] + pd.DateOffset(days=10)
            # print(date_start_minus_10_days)
            # print(date_end_plus_10_days)
            # print(df['date_start'])
            filtered_df = df[(df['id_right'].notna()) &
                             (capture_date >= date_start_minus_10_days) &
                             (capture_date <= date_end_plus_10_days)]

            filtered_df2 = df[((df['id_right'].isna()) |(df['id_right'].notna())) &
                              (capture_date >= date_start_minus_10_days) &
                              (capture_date <= date_end_plus_10_days)]

            pd.set_option('display.max_columns', None)
            # print(gdf_merged.head())
            count_filtered_rows = filtered_df.shape[0]
            count_filtered_rows2 = filtered_df2.shape[0]
            print(filtered_df.head())
            if count_filtered_rows2 > 0:
                percentage_filtered = (count_filtered_rows / num_rows ) * 100
            else:
                percentage_filtered = 0.0

            print(f"Количество строк в объединенном DataFrame: {gdf_merged.shape[0]}")
            print(f"Количество определенных пожаров: {num_rows}")
            print(f"Количество всех строк с условиями: {count_filtered_rows}")

            return html.Div([
                html.H4('Результаты обработки данных:'),
                html.P(f"Количество определенных пожаров: {num_rows}"),
                html.P(f"Процент определенных пожаров от общего количества: {percentage_filtered:.2f}%")
            ])

        except Exception as e:
            return html.Div([
                html.H4('Ошибка при обработке данных:'),
                html.P(str(e))
            ])

    return html.Div("Загрузите файл и нажмите кнопку 'Расчет пожаров' для расчета процента определенных пожаров", style={'font-size': '18px'})


@app.callback(
    [Output('output-data1', 'children'),
     Output('output-data1-store', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output1(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file_io = io.BytesIO(decoded)
        polygons_gdf = gpd.read_file(file_io)

        gdf = polygons_gdf.to_crs(epsg=3857)

        gdf['area_sqm'] = gdf['geometry'].area / 1e6

        max_area_index = gdf['area_sqm'].idxmax()

        total_area_excluded = 0
        for idx, row in gdf.iterrows():
            if idx != max_area_index:
                print(f"Polygon {idx + 1} area: {row['area_sqm']} square meters")
                total_area_excluded += row['area_sqm']

        max_area = gdf.loc[max_area_index, 'area_sqm']

        if max_area > 0:
            percentage_excluded = (total_area_excluded / max_area) * 100
        else:
            percentage_excluded = 0

        excluded_area = max_area - total_area_excluded
        print(f"\nTotal area of excluded polygons: {total_area_excluded} square meters")
        print(f"Percentage of excluded area compared to the largest polygon area: {percentage_excluded:.2f}%")

        result_div = html.Div([
            html.H4('Результаты загрузки файла:'),
            html.P(f"Файл '{filename}' загружен успешно."),
            html.P(f"Количество строк в файле: {len(polygons_gdf)}"),
            html.P(f"Процент пожаров к общей площади: {percentage_excluded:.2f}%"),
            html.P(f"Общая площадь пожаров: {total_area_excluded}"),
            html.P(f"Общая площадь территории: {excluded_area:.0f}")
        ])

        return result_div, len(polygons_gdf)
    else:
        return html.Div("Загрузите файл для обработки", style={'font-size': '18px'}), None

@app.callback(
    Output('map-output', 'children'),
    Input('update-map-btn', 'n_clicks'),
)
def update_nbr_image(n_clicks):
    if n_clicks:
        geometry = ee.Geometry.Polygon([
            [[106.8, 59.4], [111.888, 59.4], [111.888, 56.26], [106.8, 56.26]]
        ])

        irk = ee.FeatureCollection('projects/ee-yupest/assets/Kirenskoe')
        fireStart = ee.Date('2018-07-15')
        fireEnd = ee.Date('2019-08-11')
        s2 = ee.ImageCollection("COPERNICUS/S2")
        filtered = s2.filterBounds(geometry).select('B.*')
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        csPlusBands = csPlus.first().bandNames()

        def add_cs_bands(image):
            csImage = csPlus.filter(ee.Filter.eq('system:index', image.get('system:index'))).first()
            return image.addBands(csImage.select(csPlusBands))

        filteredS2WithCs = filtered.map(add_cs_bands)

        def maskLowQA(image):
            qaBand = 'cs'
            clearThreshold = 0.5
            mask = image.select(qaBand).gte(clearThreshold)
            return image.updateMask(mask)

        filteredMasked = filteredS2WithCs.map(maskLowQA)

        def addNBR(image):
            nbr = image.normalizedDifference(['B8', 'B12']).rename(['nbr'])
            return image.addBands(nbr)

        before = filteredMasked.filterDate(fireStart, fireStart.advance(1, 'month')).median()
        after = filteredMasked.filterDate(fireEnd, fireEnd.advance(1, 'month')).median()
        before_nbr = addNBR(before).select('nbr')
        after_nbr = addNBR(after).select('nbr')
        change = before_nbr.subtract(after_nbr).clip(irk)
        severity = change \
            .where(change.lt(0.10), 0) \
            .where(change.gte(0.10).And(change.lt(0.27)), 1) \
            .where(change.gte(0.27).And(change.lt(0.44)), 2) \
            .where(change.gte(0.44).And(change.lt(0.66)), 3) \
            .where(change.gte(0.66), 4)

        aoi_geojson = json.dumps(irk.geometry().getInfo())
        colors = {
            0: (0, 128, 0),  # Green
            1: (255, 255, 0),  # Yellow
            2: (255, 165, 0),  # Orange
            3: (255, 0, 0),  # Red
            4: (255, 0, 255)  # Magenta
        }

        color_map = dl.Colorbar(colorscale=list(colors.values()), min=0.1, max=1)
        visualization_params = {
            'min': 0,
            'max': 4,
            'palette': ['green', 'yellow', 'orange', 'red', 'magenta']
        }

        severity_colored = severity.visualize(**visualization_params)
        severity_colored_url = severity_colored.clip(irk).getMapId()['tile_fetcher'].url_format

        map = html.Div([
            dl.Map([
                dl.TileLayer(),
                dl.GeoJSON(data=aoi_geojson, id="aoi"),
                dl.TileLayer(url=severity_colored_url),
                color_map
            ],
                style={'width': '100%', 'height': '100vh'},
                center=[58.83, 109.15],
                zoom=7)
        ])
        return map


@app.callback(
    Output('output-data-upload', 'children'),
    Output('image-output', 'children'),
    Output('ndvi-output', 'children'),
    Output('export-status', 'children'),
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('latitude-input', 'value'),
     Input('longitude-input', 'value'),
     Input('band-select', 'value'),
     Input('export-btn', 'n_clicks'),
     Input('calculate-ndvi-btn', 'n_clicks'),
     Input('process-fire-data-btn', 'n_clicks'),
     Input('export-image-btn', 'n_clicks')],
    [State('upload-data', 'filename')],
)
def combined_callback(upload_contents, start_date, end_date, latitude, longitude, bands, export_btn_clicks,
                      calculate_ndvi_btn_clicks, process_fire_data_btn_clicks, export_image_btn_clicks, filename):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'export-image-btn':
        if export_image_btn_clicks:
            export_image_tiff(export_image_btn_clicks, start_date, end_date, latitude, longitude, bands)
            return (None, None, None, html.Div("Экспорт изображения в формате GeoTIFF запущен."))
        else:
            print('Не нажата')

    if trigger_id == 'upload-data' and process_fire_data_btn_clicks:
        count_filtered_rows = process_fire_data(filename, 'final_rosleshoz_index.csv', start_date, end_date)
        return (None, None, None, f"Обработано {count_filtered_rows} строк.")

    if trigger_id == 'upload-data' and calculate_ndvi_btn_clicks:
        if upload_contents:
            return (update_output(upload_contents, filename, 'LANDSAT/LC08/C01/T1_SR', start_date, end_date), None, None, None)
        else:
            return (html.Div("Файл не загружен."), None, None, None)

    if trigger_id == 'calculate-ndvi-btn':
        if upload_contents:
            return (update_output(upload_contents, filename, 'LANDSAT/LC08/C01/T1_SR', start_date, end_date), None, None, None)
        else:
            return (html.Div("Файл не загружен."), None, None, None)

    if trigger_id in ['date-picker-range', 'latitude-input', 'longitude-input', 'band-select']:
        return (None, update_image(start_date, end_date, latitude, longitude, bands), None, None)

    if trigger_id == 'export-btn':
        if export_btn_clicks:
            return (None, None, None, export_image(export_btn_clicks, start_date, end_date, latitude, longitude, bands))
        else:
            print('Не нажата')

    return (None, None, None, None)



if __name__ == '__main__':
    app.run_server(debug=True)

