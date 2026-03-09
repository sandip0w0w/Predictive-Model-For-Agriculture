import streamlit as st
import joblib
import pandas as pd
import xgboost
from kpi_files import *
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv('soil_data.csv')
def add_ratios(X):
  X = X.copy()
  X['N_P_ratio'] = X['N'] / (X['P'] + 1e-5) # adding epsilon to avoid division by zero
  X['P_K_ratio'] = X['P'] / (X['K'] + 1e-5)
  X['N_K_ratio'] = X['N'] / (X['K'] + 1e-5)
  return X

#importing model
loaded_model = joblib.load('crop_prediction_model.joblib')
loaded_encoder = joblib.load('encoder.joblib')

df = add_ratios(df)
X = df.drop('crop', axis = 1)
y = df['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
y_test = loaded_encoder.transform(y_test)
y_pred = loaded_model.predict(X_test)

@st.cache_data
def tukey_hsd(df, feature):  
  m_comp = pairwise_tukeyhsd(endog = df[feature], groups = df['crop'], alpha = 0.05)

  tukey_data = pd.DataFrame(data = m_comp._results_table.data[1:],
                              columns = m_comp._results_table.data[0])
    
  #Filter for Statistical Twins
  twins = tukey_data[tukey_data['reject'] == False]

  if not twins.empty:
    st.write(f"Crop pair: {feature}")
    st.write(twins[['group1', 'group2', 'meandiff', 'p-adj']])
  else:
    print(f"All Crops are statistically distinct for {feature}")


problem_statement = """

**Problem Statment**


In agriculture, farmers predominantly rely on traditional, season based crop selection methods, 
which often overlook the critical role of soil nutrients such as Nitrogen (N), Phosphorus (P), Potassium (K),
and pH levels in determining crop suitability and yield potential. This approach leads to suboptimal crop choices,
reduced productivity, and inefficient resource use, especially in varying soil conditions. Our project addresses
this gap by developing a predictive model using a dataset of N, P, K, pH values, and their ratios paired with
corresponding crop outcomes. The model will analyze and recommend the most suitable crops for specific soil
nutrient profiles, enabling precision agriculture that maximizes yield and sustainability.

"""

def main():
  
  st.set_page_config(
    page_title = "Predicitive Model For Agriculutre",
    page_icon = "📈",
)
  
  st.sidebar.title("Navigation")

  menu = ["Problem Statement", "KPI","Dataset Info","Predict Crop", "Findings & Summary"]

  choice = st.sidebar.selectbox("Choose a page", menu)

  if choice == "Problem Statement":
    st.title("🎓Project Problem Statement")
    st.markdown(problem_statement)
    st.markdown("---")

  elif choice == "KPI":
    st.sidebar.header("Crops")
    crops_menu = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    choice = st.sidebar.selectbox("Choose a crop", crops_menu)
    dominant_nutrient, nutrient_value, min_ph, max_ph = nutrient_score(df, choice)
    st.title(choice.upper())

    col1, col2 = st.columns(2)
    col3, col4, col5 = st.columns(3)

    with col1:
      st.metric(
      label = f"Most Important Nutrient",
      value = dominant_nutrient.upper(),
      border = True
    )
    
    with col2:
      st.metric(
        label = "Optimal pH",
        value = f"{min_ph:.1f} - {max_ph:.1f}",
        border = True
      )
    
    ratios, status = npk_balance(df, choice)


    def color_status(status, ratios):
      if status == "Optimal":
          return f":green[{ratios:.2f}]"
      elif status == "High":
          return f":orange[{ratios:.2f}]"
      else:
          return f":red[{ratios:.2f}]"


    with col3:
      st.metric(
        label = "Nitrogen Ratio",
        value = color_status(status["N"], ratios["N"]),
        border = True
        
      )

    with col4:
      st.metric(
        label = "Phosphorous Ratio",
        value = color_status(status["P"], ratios["P"]),
        border = True
        
      )
    
    with col5:
      st.metric(
        label = "Potassium Ratio",
        value = color_status(status["K"], ratios["K"]),
        border = True
    
      )

  elif choice == 'Predict Crop':

    st.title("Predictive Model")
    st.subheader("Choose values from the slider")

    nitrogen_value = st.slider("Nitrogen(N)",
              min_value = 0,
              max_value = 140,
              )
    st.write("Nitrogen(N):", nitrogen_value)

    phosphorus_value = st.slider("Phosphorus(P)",
                                  min_value = 0,
                                  max_value = 145)
    st.write("Phosphorus(P):", phosphorus_value)

    potassium_value = st.slider("Potassium(K)",
                    min_value = 0,
                    max_value = 205
                    )
    st.write("Potassium(K):", potassium_value)

    ph_value = st.slider("Ph",
                        min_value = 0,
                        max_value = 14
                        )
    st.write("Ph:", ph_value)

    input_df = pd.DataFrame({
        'N' : [nitrogen_value],
        'P' : [phosphorus_value],
        'K' : [potassium_value],
        'ph': [ph_value]
    })


    combined_df = add_ratios(input_df)

    prediction = loaded_model.predict(combined_df)
    predicted_crop = loaded_encoder.inverse_transform(prediction)
    st.success(f"The predicted crop is **{predicted_crop[0]}**")
  
  elif choice == 'Dataset Info':
    st.title("📊Dataset Analysis")
    st.markdown("This section provides a statistical overview of the soil nutrient distributions.")
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T)

    st.subheader("⚖️ Dataset Balance")
    class_counts = df['crop'].value_counts().reset_index()
    class_counts.columns = ['Crop', 'Samples']

    fig_balance = px.bar(
    class_counts, 
    x='Crop', 
    y='Samples', 
    color='Samples',
    title="Number of samples available per crop type"
)
    st.plotly_chart(fig_balance, use_container_width=True)

    st.subheader("Distribution & Central Tendency")

    cols_to_plot = ['N', 'P', 'K']

    df_melted = df.melt(value_vars= cols_to_plot, var_name = 'Nutrient', value_name = 'Value')

    fig1 = px.box(
      df_melted,
      x = 'Nutrient',
      y = 'Value',
      color = 'Nutrient',
      points = 'all',
      title = 'Soil Nutrient Spread (Lines Represent the Median)'
    )
    st.plotly_chart(fig1, use_container_width = True)

    st.subheader("Feature Correaltion")
    st.write("This heatmap shows how nutrients relate to one another.")
   

    corr = df.drop(columns = ['crop']).corr()

    fig, ax = plt.subplots(figsize = (10,8))
    sns.heatmap(corr, annot = True, cmap = 'RdYlGn', center = 0, ax = ax)
    st.pyplot(fig)

    st.subheader("Nutrient Requirements by Crops")
    target_nutrient = st.selectbox("Select Nutrient for Range Analysis", ['N', 'P', 'K'])

    fig_range = px.box(
      df, 
      x = 'crop',
      y = target_nutrient,
      color = 'crop',
      title = f"Required range of {target_nutrient} for all crops"
    )
    fig_range.update_layout(xaxis = {'categoryorder': 'total descending'})
    fig_range.update_xaxes(tickangle = 60)
    st.plotly_chart(fig_range, use_container_width= True)
  
  elif choice == "Findings & Summary":

    st.title("Findings & Summary")

    st.subheader("Model Performance & Findings")
    st.markdown("""
    To identify the most reliable recommendation engine, we conducted an extensive **Benchmarking Study**. 
    We evaluated four distinct machine learning architectures using a variety of hyperparameters to optimize 
    for precision and recall.
    """)

    results_data = {
        "Model Algorithm": ["Random Forest", "XGBoost", "Decision Tree", "Logistic Regression"],
        "Accuracy Score": ["78.69%", "78.57%", "77.32%", "70.51%"],
        "Strengths": ["Best at handling non-linear soil data", "Fast execution", "Highly interpretable", "Simple baseline"]
    }
    st.table(results_data)

    st.success("""
    **Key Takeaway:** Random Forest emerged as the top performer with an overall accuracy of **78.69%**. 
    It demonstrated superior ability in distinguishing between crops with overlapping nutrient profiles 
    (like pulses and legumes).
    """)

    st.subheader("Top Confused Crops")

    cm = confusion_matrix(y_test, y_pred)

    def get_top_confusion(cm, labels, n = 3):

      cm_df = pd.DataFrame(cm, index = labels, columns = labels)

      # Reshaping to the long format
      confusion_list = cm_df.stack().reset_index()
      confusion_list.columns = ['Actual', 'Predicted', 'Count']

      # Filtering correct predictions
      errors = confusion_list[confusion_list['Actual'] != confusion_list['Predicted']]

      # Sort by count to find the biggest mistake
      top_errors = errors.sort_values(by = 'Count', ascending = False).head(n)
      return top_errors

    top_3_errors = get_top_confusion(cm, loaded_encoder.classes_)

    def plot_confusion_matrix_error_ratio(df):
        df['Mistake'] = df['Actual'] + " misidentified as " + df['Predicted']
        
        df = df.sort_values(by='Count', ascending=True)

        # Creating the Plotly Horizontal Bar Chart
        fig = px.bar(
            df,
            x='Count',
            y='Mistake',
            orientation='h',
            title = "Top 3 confused Crops",
            color='Count',
            color_continuous_scale='Reds',
        )

        # Styling for Streamlit/Web presentation
        fig.update_layout(
            xaxis_title="Number of Mistakes",
            yaxis_title="Type of Confusion"
        )

        # Use this in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    plot_confusion_matrix_error_ratio(top_3_errors)


    st.subheader("Nutrient Distribution of Confused Crops")

    nutrients_to_check = ['N', 'P', 'K', 'ph', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio']
    crops = ['rice', 'jute', 'grapes', 'apple']

    selected_crops = st.multiselect("Select crops to compare", crops, crops[:2])

    if selected_crops:
      cols = st.columns(2)

      for i, nutrient in enumerate(nutrients_to_check):

        col_index = i % 2

        with cols[col_index]:

          hist_data = [df[df['crop'] == crop][nutrient].dropna() for crop in selected_crops]

          fig = ff.create_distplot(
            hist_data,
            selected_crops,
            show_hist = False,
            show_rug = False
          )

          fig.update_layout(
            title = f"Overlap {nutrient}",
            margin = dict(l = 20, r = 20, t = 40, b = 20),
            height = 350,
            showlegend = False if i < len(nutrients_to_check) - 1 else True
          )
          st.plotly_chart(fig, use_container_width= True)
    else:
      st.warning("Please select crops to visualize distribution overlap")
    
    st.markdown('''
The overlapping KDE curves indicate that the class-conditional densities for these crops are nearly identical
in the N-P-K feature space. This statistical similarity leads to high 'crosstalk' between classes. Enhancing the
model will require moving beyond chemical inputs to include spatial and environmental variables that capture the
unique ecological niches of these crops.
                ''')
    

    st.subheader("Statistical Test")
    st.markdown("**Tukey’s HSD (Honestly Significant Difference)**")
    st.markdown("""
Tukey’s HSD is a statistical test that compares all groups in a dataset to see if the differences
between their averages are truly significant or just due to random chance.

In this project, it acts as a diagnostic tool to identify "statistical twins" pairs of crops whose
soil requirements are so similar that they become mathematically indistinguishable. By highlighting
these specific overlaps (where $p$-values reach 1.0), the test explains that the model’s 78% accuracy is
limited by these identical data profiles rather than an issue with the underlying code.                                
                """)
    
    selected_nutrient = st.selectbox("Select Nutrient:",nutrients_to_check)
    tukey_hsd(df, selected_nutrient)



if __name__ == "__main__":
  main()


