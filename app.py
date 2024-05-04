from flask import Flask, request, render_template, current_app
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') 
import io
import base64

app = Flask(__name__)
app.config['DATAFRAME'] = pd.read_csv('data/Tobaccodataset.csv')  # Load the dataset into the app's configuration

def clean_data(data):
    # Fill NaN values in 'Sample_Size' with the median of the column
    data['Sample_Size'] = data['Sample_Size'].fillna(data['Sample_Size'].median())
    return data

def train_state_specific_models(data, states, genders):
    local_models = {}
    data = clean_data(data)  # Clean data before training
    for state in states:
        for gender in genders:
            state_gender_data = data[(data['LocationAbbr'] == state) & (data['Gender'] == gender)]
            if not state_gender_data.empty:
                X = state_gender_data[['YEAR']].values
                y = state_gender_data['Sample_Size'].values
                model = LinearRegression()
                model.fit(X, y)
                local_models[(state, gender)] = model
    return local_models

def create_bar_chart(x_values, y_values, title):
    plt.figure(figsize=(10, 6))
    plt.bar(x_values.astype(str), y_values, color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Sample Size')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return plot_base64


def create_grouped_bar_chart(df, title='Sample Size by Year'):
    # Set figure size
    plt.figure(figsize=(10, 6))

    # Prepare data: group by 'YEAR' and 'Gender' and sum/average the 'Sample Size'
    grouped = df.groupby(['YEAR', 'Gender'])['Sample_Size'].sum().unstack()
    years = grouped.index
    bar_width = 0.35  # Width of the bars

    # Create bars for each gender
    if 'Male' in grouped.columns:
        male_sizes = grouped['Male']
        plt.bar(years, male_sizes, width=bar_width, label='Male', color='blue', align='center')

    if 'Female' in grouped.columns:
        female_sizes = grouped['Female']
        plt.bar(years + bar_width, female_sizes, width=bar_width, label='Female', color='pink', align='center')

    # Add labels, title, and legend
    plt.xlabel('Year')
    plt.ylabel('Sample Size')
    plt.title(title)
    plt.xticks(years + bar_width / 2, years)  # Positioning x-labels in the center of grouped bars
    plt.legend()  # Add a legend
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot to base64
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return plot_base64


@app.route('/')
def index():
    if 'models' not in app.config:
        df = app.config['DATAFRAME']
        states = df['LocationAbbr'].unique()
        genders = ['Male', 'Female']
        app.config['models'] = train_state_specific_models(df, states, genders)
    locations = app.config['DATAFRAME']['LocationAbbr'].unique().tolist()
    return render_template('index.html', locations=locations)

df = pd.read_csv('data/Tobaccodataset.csv')
@app.route('/results', methods=['POST'])
def results():
    location = request.form.get('location')
    genders = request.form.getlist('gender')
    filtered_data = df[(df['LocationAbbr'] == location) & (df['Gender'].isin(genders))]

    if filtered_data.empty:
        return "No data available for the selected state and/or gender."

    # Create the grouped bar chart
    title = 'Sample Size by Year for ' + ', '.join(genders) if genders else 'All Genders'
    plot_data = create_grouped_bar_chart(filtered_data, title)

    return render_template('results.html', plot_data=plot_data)

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get('state')
    gender = request.form.get('gender')
    model_key = (state, gender)
    models = current_app.config['models']

    # Initialize state_data before the if block to ensure it exists
    state_data = current_app.config['DATAFRAME'][(current_app.config['DATAFRAME']['LocationAbbr'] == state) & (current_app.config['DATAFRAME']['Gender'] == gender)]
    state_data = clean_data(state_data)

    if state_data.empty:
        return "No historical or valid data available for the selected state and gender."

    # Only attempt to access the model if state_data is not empty
    if model_key not in models:
        X = state_data[['YEAR']].values
        y = state_data['Sample_Size'].values
        model = LinearRegression()
        model.fit(X, y)
        models[model_key] = model

    model = models[model_key]
    latest_year = state_data['YEAR'].max()
    future_years = np.arange(latest_year + 1, latest_year + 11).reshape(-1, 1)
    predictions = model.predict(future_years)

    plot_data = create_bar_chart(future_years.flatten(), predictions, f'Predicted Tobacco Usage in {state} for {gender}')
    return render_template('predict.html', plot_data=plot_data)



if __name__ == '__main__':
    app.run(debug=True)



# first one working import matplotlib
# matplotlib.use('agg')  # Set the backend to 'agg' for server-side plotting

# import matplotlib.pyplot as plt
# import io
# import base64
# from flask import Flask, render_template, request
# import pandas as pd
# import threading
# from sklearn.linear_model import LinearRegression
# import pandas as pd
# from flask import Flask, request, render_template
# import matplotlib.pyplot as plt
# import io
# import base64
# import numpy as np

# # Load the dataset
# df = pd.read_csv('data/Tobaccodataset.csv')

# # Extract unique locations for dropdown
# locations_list = df['LocationAbbr'].unique().tolist()

# # Initialize Flask application
# app = Flask(__name__)


# # for prediction
# def prepare_model(data):
#     # Assuming 'YEAR' and 'Sample_Size' need cleaning
#     data = data.dropna(subset=['YEAR', 'Sample_Size'])  # Option to drop NaN values
#     # Or fill NaN values instead
#     data['Sample_Size'] = data['Sample_Size'].fillna(data['Sample_Size'].mean())

#     model = LinearRegression()
#     X = data['YEAR'].values.reshape(-1, 1)  # Ensure YEAR is not NaN
#     y = data['Sample_Size'].values          # Ensure Sample_Size is not NaN
#     model.fit(X, y)
#     return model

# # Global model dictionary to store models for each gender
# models = {
#     'Male': prepare_model(df[df['Gender'] == 'Male']),
#     'Female': prepare_model(df[df['Gender'] == 'Female'])
# }

# # Function to predict future usage
# def predict_future_usage(model, start_year, num_years=10):
#     future_years = np.arange(start_year + 1, start_year + 1 + num_years).reshape(-1, 1)
#     predictions = model.predict(future_years)
#     return future_years.flatten(), predictions

# # Function to generate bar chart for predictions
# def create_bar_chart(x_values, y_values, title):
#     plt.figure(figsize=(10, 6))
#     plt.bar(x_values.astype(str), y_values, color='skyblue')
#     plt.xlabel('Year')
#     plt.ylabel('Sample Size')
#     plt.title(title)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close()
#     return plot_base64


# def create_grouped_bar_chart(df, title='Sample Size by Year'):
#     # Set figure size
#     plt.figure(figsize=(10, 6))

#     # Prepare data: group by 'YEAR' and 'Gender' and sum/average the 'Sample Size'
#     grouped = df.groupby(['YEAR', 'Gender'])['Sample_Size'].sum().unstack()
#     years = grouped.index
#     bar_width = 0.35  # Width of the bars

#     # Create bars for each gender
#     if 'Male' in grouped.columns:
#         male_sizes = grouped['Male']
#         plt.bar(years, male_sizes, width=bar_width, label='Male', color='blue', align='center')

#     if 'Female' in grouped.columns:
#         female_sizes = grouped['Female']
#         plt.bar(years + bar_width, female_sizes, width=bar_width, label='Female', color='pink', align='center')

#     # Add labels, title, and legend
#     plt.xlabel('Year')
#     plt.ylabel('Sample Size')
#     plt.title(title)
#     plt.xticks(years + bar_width / 2, years)  # Positioning x-labels in the center of grouped bars
#     plt.legend()  # Add a legend
#     plt.tight_layout()

#     # Save the plot to a BytesIO object
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)

#     # Encode the plot to base64
#     plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close()

#     return plot_base64




# # Function to run Matplotlib in a separate thread
# def run_matplotlib():
#     app.app_context().push()
#     with app.test_request_context():
#         plt.figure(figsize=(10, 6))
#         plt.plot([1, 2, 3], [1, 2, 3])  # Placeholder plot
#         plt.close()

# # Start Matplotlib thread
# matplotlib_thread = threading.Thread(target=run_matplotlib)
# matplotlib_thread.start()

# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html', locations=locations_list)

# @app.route('/results', methods=['POST'])
# def results():
#     location = request.form.get('location')
#     genders = request.form.getlist('gender')
#     filtered_data = df[(df['LocationAbbr'] == location) & (df['Gender'].isin(genders))]

#     if filtered_data.empty:
#         return "No data available for the selected state and/or gender."

#     # Create the grouped bar chart
#     title = 'Sample Size by Year for ' + ', '.join(genders) if genders else 'All Genders'
#     plot_data = create_grouped_bar_chart(filtered_data, title)

#     return render_template('results.html', plot_data=plot_data)



# @app.route('/predict', methods=['POST'])
# def predict():
#     state = request.form.get('state')
#     gender = request.form.get('gender')
#     data = df[(df['LocationAbbr'] == state) & (df['Gender'] == gender)]
#     if data.empty:
#         return "No historical data available for the selected state and gender."

#     latest_year = data['YEAR'].max()
#     model = models[gender]
#     future_years, predictions = predict_future_usage(model, latest_year)
#     plot_data = create_bar_chart(future_years, predictions, f'Predicted Tobacco Usage in {state} for {gender}')
#     return render_template('predict.html', plot_data=plot_data)  # Use predict.html here

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
