from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

df = pd.read_csv('data/Student_performance_data_.csv')
X = df[['StudyTimeWeekly', 'Absences']]
y = df['GPA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

def save_plot(fig, filename):
    path = os.path.join('static', filename)
    fig.savefig(path)
    plt.close(fig)
    return path

@app.route('/')
def index():
    def save_plot(fig, filename):
        path = os.path.join('static', filename)
        fig.savefig(path)
        plt.close(fig)
        return filename  # Kembalikan hanya nama file

    # Visualizations
    fig1 = plt.figure(figsize=(8, 5))
    sns.histplot(df['GPA'], kde=True, color='skyblue')
    plt.title('Distribusi GPA')
    plot1 = save_plot(fig1, 'gpa_dist.png')

    fig2 = plt.figure(figsize=(8, 5))
    sns.boxplot(x='Gender', y='GPA', data=df)
    plt.title('Distribusi GPA berdasarkan Gender')
    plot2 = save_plot(fig2, 'gpa_gender.png')

    fig3 = plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelasi antar variabel')
    plot3 = save_plot(fig3, 'heatmap.png')

    fig4 = plt.figure(figsize=(8, 5))
    sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df)
    plt.title('Scatter Plot Study Time vs GPA')
    plot4 = save_plot(fig4, 'studytime_gpa.png')

    fig5 = plt.figure(figsize=(8, 5))
    sns.countplot(x='Ethnicity', data=df, palette='viridis')
    plt.title('Jumlah siswa berdasarkan Grade Class')
    plot5 = save_plot(fig5, 'ethnicity_count.png')

    fig6 = plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Nilai Aktual GPA')
    plt.ylabel('Prediksi GPA')
    plt.title('Prediksi GPA berdasarkan Study Time dan Absences')
    plt.grid()
    plot6 = save_plot(fig6, 'prediction.png')

    return render_template('index.html',
                           rmse=round(rmse, 2),
                           r2=round(r2, 2),
                           plot1=plot1, plot2=plot2, plot3=plot3,
                           plot4=plot4, plot5=plot5, plot6=plot6)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        studytime = float(request.form['studytime'])
        absences = float(request.form['absences'])
        gpa = round(model.intercept_ + model.coef_[0]*studytime + model.coef_[1]*absences, 2)
        return render_template('result.html', gpa=gpa)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
