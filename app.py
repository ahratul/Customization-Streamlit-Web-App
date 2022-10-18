import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from time import sleep
import plotly.express as px


def main():
    st.title('Customer Analysis & Segmentation')
    st.write('')
    st.write('Using A K-Means clustering algorithm to segment customers into distinct groups.')
    st.write('')
    st.subheader("Goal and Customer Data")
    st.write('The goal of this tool is to take an example customer dataset, containing commonly collected data,'
             ' and use an unsupervised machine learning model to split customers into segments based '
             'on shared characteristics.')
    st.write('The raw data looks like this: ')
    st.write('')
    df = pd.read_csv('data.csv')
    st.dataframe(df)
    st.markdown('[Data Source](https://www.kaggle.com/dev0914sharma/customer-clustering?select=segmentation+data.csv)')
    st.write('')

    if 'clicked' not in st.session_state:
        st.session_state['clicked'] = 0

    segmented = cluster_data()[0]
    if st.button("Search for Customer Segments"):
        st.session_state['clicked'] = 1

        st.subheader('Customer Segments')
        progress = st.progress(0)

        for i in range(100):
            progress.progress(i)
            sleep(0.1)

        st.write("Segmentation complete... 5 segments discovered!")

    if st.session_state['clicked'] == 1:
        choice = st.sidebar.selectbox("Filter customers by segment:", np.unique(segmented['cust_persona']))
        segmented_filtered = segmented[(segmented['cust_persona'] == choice)]
        st.subheader('Segmented Data')
        st.write('')
        st.dataframe(segmented_filtered)
        st.write('')
        st.subheader('Segment Insights')

        def plot(x, data=segmented_filtered, color='yellow'):
            fig = px.histogram(data, x=x, color_discrete_sequence=[color])
            st.plotly_chart(fig, use_container_width=True)

        def column_display(x1, x2):
            col1, col2 = st.columns(2)
            with col1:
                plot(x1)
            with col2:
                plot(x2)

        if choice == 0:
            st.write("Customers belonging to persona segment 0 are nearly all single and living in a smaller city."
                     " 63% of customers in this segment are Male, with the overwhelming majority of customers being "
                     "between 20 and 41 years old. This segment tends to have less formal education and work in"
                     "more entry level positions. Customers in this segment have a yearly income ranging from "
                     "$60k to $125k.")
            analysis_view = st.selectbox("Select", ['Sociodemographic', 'Professional', 'Geographic'])
            if analysis_view == 'Sociodemographic':
                column_display("Sex", "Marital status")
                plot('Age')
            elif analysis_view == 'Professional':
                plot('Income')
                column_display("Education", "Occupation")
            elif analysis_view == 'Geographic':
                plot('Settlement size')

        elif choice == 1:
            st.write("Customers belonging to persona segment 1 are nearly all married, with less formal education "
                     "and working in entry level positions. 65% of of customers in this segment are Female, with"
                     "an age range of 23 to 30, and belonging to a slightly higher income bracket of $90k to $180k."
                     " Customers in this segment overwhelmingly tend to live in a mid sized or big city.")
            analysis_view = st.selectbox("Select", ['Sociodemographic', 'Professional', 'Geographic'])
            if analysis_view == 'Sociodemographic':
                column_display("Sex", "Marital status")
                plot('Age')
            elif analysis_view == 'Professional':
                plot('Income')
                column_display("Education", "Occupation")
            elif analysis_view == 'Geographic':
                plot('Settlement size')

        elif choice == 2:
            st.write("The defining trait of persona segment 2 is that belonging customers are more highly educated "
                     "and of an older age bracket, between the ages of 40 and 74. Customers in this segment are"
                     "of an even gender split, with 67% being married. Over half of customers in this segment work"
                     "in an entry to mid level position, with incomes ranging from $110k to $220k. There is also "
                     "a small subset of this segment with salaries ranging from $260k to $310k. These customers"
                     "live in cities of all sizes.")
            analysis_view = st.selectbox("Select", ['Sociodemographic', 'Professional', 'Geographic'])
            if analysis_view == 'Sociodemographic':
                column_display("Sex", "Marital status")
                plot('Age')
            elif analysis_view == 'Professional':
                plot('Income')
                column_display("Education", "Occupation")
            elif analysis_view == 'Geographic':
                plot('Settlement size')

        elif choice == 3:
            st.write("Customer segment 3, the Bachelor segment, is entirely Male, Single and living in either "
                     "mid sized or big cities. The vast majority of belonging customers are high school educated, "
                     "working in entry to mid level positions, with a small subset working as senior managers."
                     " The salary range for this group is $90 to $180k, with a very wide distribution of ages"
                     "ranging from 22 to 55 years old.")
            analysis_view = st.selectbox("Select", ['Sociodemographic', 'Professional', 'Geographic'])
            if analysis_view == 'Sociodemographic':
                column_display("Sex", "Marital status")
                plot('Age')
            elif analysis_view == 'Professional':
                plot('Income')
                column_display("Education", "Occupation")
            elif analysis_view == 'Geographic':
                plot('Settlement size')

        elif choice == 4:
            st.write("Customers belonging to segment 4 are all married and overwhelmingly female. Ages range from "
                     "20 to 37, with incomes ranging from $60k to $130k. Customers belonging to this group all "
                     "live in smaller cities, with a high school education, and either unemployed or working in "
                     "an entry level position.")
            analysis_view = st.selectbox("Select", ['Sociodemographic', 'Professional', 'Geographic'])
            if analysis_view == 'Sociodemographic':
                column_display("Sex", "Marital status")
                plot('Age')
            elif analysis_view == 'Professional':
                plot('Income')
                column_display("Education", "Occupation")
            elif analysis_view == 'Geographic':
                plot('Settlement size')


def cluster_data():
    df = pd.read_csv('data.csv')

    working_df = df.drop(['ID'], axis=1)
    scaler = StandardScaler()
    working_df = scaler.fit_transform(working_df)
    n_clusters=st.sidebar.number_input("N_Clusters", 1, 10, step=1, key='N_Clusters')
    max_iter=st.sidebar.number_input("Maximum number of iterations",1,301,step=50,key="Maximum number of iterations")
    tol = st.sidebar.radio("Relative tolerance", {'0.0001','0.0005', '0.001'},key="Relative tolerance")
    algorithm = st.sidebar.radio("Algorithm", {'elkan', 'auto', 'full'}, key='Algorithm')
    init=st.sidebar.radio("Method for initialization", {'k-means++', 'random'}, key='Method for initialization')
    model = KMeans(n_clusters=n_clusters, random_state=123, init=init,max_iter=max_iter, tol=float(tol), verbose=0, copy_x=True, algorithm=algorithm)
    labels = model.fit_predict(working_df)

    df['cust_persona'] = labels

    df['Sex'][(df['Sex'] == 0)] = 'Male'
    df['Sex'][(df['Sex'] == 1)] = 'Female'

    df['Marital status'][(df['Marital status'] == 0)] = 'Single'
    df['Marital status'][(df['Marital status'] == 1)] = 'Married'

    df['Education'][(df['Education'] == 0)] = 'Other/Unknown'
    df['Education'][(df['Education'] == 1)] = 'High School'
    df['Education'][(df['Education'] == 2)] = 'University'
    df['Education'][(df['Education'] == 3)] = 'Graduate School'

    df['Occupation'][(df['Occupation'] == 0)] = 'Unemployed or Unskilled'
    df['Occupation'][(df['Occupation'] == 1)] = 'Entry to Mid Level Employee'
    df['Occupation'][(df['Occupation'] == 2)] = 'Senior Manager, Business Owner, Officer'

    df['Settlement size'][(df['Settlement size'] == 0)] = 'Small City'
    df['Settlement size'][(df['Settlement size'] == 1)] = 'Mid-Sized City'
    df['Settlement size'][(df['Settlement size'] == 2)] = 'Big City'

    return df, model


if __name__ == '__main__':
    main()
