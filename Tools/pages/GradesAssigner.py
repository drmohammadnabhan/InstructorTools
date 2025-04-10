
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For ANOVA
from io import BytesIO # For download button

# ============================================
# Define Core Grading Functions (These could be in a separate .py file in your repo)
# ============================================

def calculate_initial_cutoffs(params):
    # Takes parameters (F_Threshold, C_Start_Threshold, C_Width, B_Width, A_Width)
    # Returns a dictionary of initial cutoffs: {'C_Start': val, 'C_End': val, ... 'A_End': val}
    cutoffs = {}
    cutoffs['D_End'] = params['C_Start_Threshold']
    cutoffs['C_End'] = cutoffs['D_End'] + params['C_Width']
    cutoffs['B_End'] = cutoffs['C_End'] + params['B_Width']
    cutoffs['A_End'] = cutoffs['B_End'] + params['A_Width']
    cutoffs['F_Max'] = params['F_Threshold'] # Or D_End if F_Threshold is start of D
    # Adjust based on exact definition (e.g. F < F_thresh, D is [F_thresh, C_Start_thresh))
    # ... calculate C_Start, B_Start, A_Start based on the Ends ...
    # Need consistent naming (e.g., use boundary values: F_max, D_max, C_max, B_max, A_max)
    # Let's redefine for clarity:
    cutoffs = {}
    cutoffs['F_Upper'] = params['F_Threshold'] # Scores below this are F
    cutoffs['D_Upper'] = params['C_Start_Threshold'] # Scores below this (and >= F_Upper) are D
    cutoffs['C_Upper'] = cutoffs['D_Upper'] + params['C_Width']
    cutoffs['B_Upper'] = cutoffs['C_Upper'] + params['B_Width']
    cutoffs['A_Upper'] = cutoffs['B_Upper'] + params['A_Width'] # This is effectively A+ start
    return cutoffs

def adjust_boundaries(scores, initial_cutoffs, params):
    # Takes scores (Pandas Series), initial cutoffs dict, and adjustment parameters
    # Implements Step 3: Boundary Analysis & Rule-Based Adjustment
    # Checks for gaps and clusters around C_Upper, B_Upper, A_Upper
    # Applies shifts based on rules (Max_Boundary_Shift, Gap_Size, Cluster_Threshold, Policy)
    # Returns adjusted_cutoffs dict and adjustment_log list
    adjusted_cutoffs = initial_cutoffs.copy()
    adjustment_log = []
    boundaries_to_check = ['C_Upper', 'B_Upper', 'A_Upper'] # Order might matter

    for boundary_key in boundaries_to_check:
        cutoff_val = adjusted_cutoffs[boundary_key]
        # --- Gap Check Logic ---
        # Find scores near cutoff +/- max_shift
        # Check for gaps >= gap_size
        # If gap found & condition met:
            # new_cutoff = calculate_shifted_value(...)
            # shift_amount = new_cutoff - cutoff_val
            # if abs(shift_amount) <= params['Max_Boundary_Shift']:
                # adjusted_cutoffs[boundary_key] = new_cutoff
                # adjustment_log.append(f"Shifted {boundary_key} to {new_cutoff:.2f} due to gap.")
                # continue # Skip cluster check if gap adjustment made

        # --- Cluster Check Logic ---
        # Count scores exactly equal to cutoff_val
        # cluster_percent = (count / len(scores)) * 100
        # if cluster_percent >= params['Dense_Cluster_Threshold']:
            # shift_direction = -0.5 if params['Cluster_Shift_Policy'] == 'Shift Down (Include in Higher Grade)' else 0.5
            # # Ensure shift doesn't exceed max_shift (might need more nuanced logic)
            # new_cutoff = cutoff_val + shift_direction # Simple example
            # adjusted_cutoffs[boundary_key] = new_cutoff
            # adjustment_log.append(f"Shifted {boundary_key} to {new_cutoff:.2f} due to cluster.")

    return adjusted_cutoffs, adjustment_log


def assign_letter_grades(scores, final_cutoffs):
    # Takes scores Series and final cutoffs dict
    # Returns a Pandas Series with letter grades
    # Use pd.cut or nested np.where for efficiency
    bins = [-np.inf, final_cutoffs['F_Upper'], final_cutoffs['D_Upper'], final_cutoffs['C_Upper'],
            final_cutoffs['B_Upper'], final_cutoffs['A_Upper'], np.inf]
    # Adjust labels based on whether F_Upper is max F score or min D score
    labels = ['F', 'D', 'C', 'B', 'A', 'A+'] # Adjust if using +/-
    grades = pd.cut(scores, bins=bins, labels=labels, right=False, ordered=True) # right=False means [lower, upper)
    return grades

def perform_sanity_checks(df, grade_col, final_cutoffs, params):
    # Check A/A+ percentile
    a_grades_percent = (df[grade_col].isin(['A', 'A+'])).mean() * 100
    flag = a_grades_percent < params['Min_A_Percentile']
    message = f"A/A+ Percentage: {a_grades_percent:.2f}%."
    if flag:
        message += f" (Warning: Below minimum threshold of {params['Min_A_Percentile']}%)"
    return flag, message

def calculate_stats_and_plots(df, score_col, grade_col, section_col, gpa_map):
    # Calculate distributions, GPAs, run ANOVA, create plots
    # Map grades to GPA points
    df['GPA'] = df[grade_col].map(gpa_map)

    # Overall stats
    overall_dist = df[grade_col].value_counts(normalize=True).sort_index()
    overall_gpa = df['GPA'].mean()

    # Per-section stats
    section_stats = df.groupby(section_col).agg(
        Avg_GPA=('GPA', 'mean'),
        Count=('GPA', 'count')
    )
    section_dist = df.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)

    # ANOVA
    section_groups = [group['GPA'].dropna() for name, group in df.groupby(section_col)]
    anova_result = None
    anova_p_value = None
    if len(section_groups) > 1:
      # Check variance and sample sizes before running ANOVA
      # Simple ANOVA for illustration:
      try:
          f_val, p_val = stats.f_oneway(*section_groups)
          anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"
          anova_p_value = p_val
      except ValueError:
          anova_result = "ANOVA could not be run (e.g., section with only one student)."


    # Plots (using matplotlib or plotly)
    # ... create histogram, grade distribution bar chart, section GPA boxplot ...
    # Return dictionary of results: distributions, stats, anova_result, plots
    results = {
        "overall_dist": overall_dist,
        "overall_gpa": overall_gpa,
        "section_stats": section_stats,
        "section_dist": section_dist,
        "anova_result": anova_result,
        "anova_p_value": anova_p_value,
        # Add plot objects here
    }
    return results

# ============================================
# Streamlit App Layout
# ============================================

st.set_page_config(layout="wide") # Use wider layout
st.title("Systematic Grading Tool (Fixed Widths + Checks)")

# --- Sidebar for Configuration ---
st.sidebar.header("1. Grading Parameters")
params = {}
# Use number_input for thresholds and widths
params['F_Threshold'] = st.sidebar.number_input("F Threshold (< score)", value=60.0, step=0.5)
params['C_Start_Threshold'] = st.sidebar.number_input("C Start Score (>= score)", value=65.0, step=0.5)
params['C_Width'] = st.sidebar.number_input("C Grade Width", value=7.0, step=0.5, min_value=1.0)
params['B_Width'] = st.sidebar.number_input("B Grade Width", value=7.0, step=0.5, min_value=1.0)
params['A_Width'] = st.sidebar.number_input("A Grade Width", value=7.0, step=0.5, min_value=1.0)
# Use number_input for adjustment rules
params['Max_Boundary_Shift'] = st.sidebar.number_input("Max Boundary Shift (+/- points)", value=1.0, step=0.5, min_value=0.0)
params['Significant_Gap_Size'] = st.sidebar.number_input("Significant Gap Size (points)", value=2.0, step=0.5, min_value=0.1)
params['Dense_Cluster_Threshold'] = st.sidebar.number_input("Dense Cluster Threshold (% of students)", value=1.0, step=0.5, min_value=0.0)
params['Cluster_Shift_Policy'] = st.sidebar.selectbox("Cluster Shift Policy", ['Shift Down (Include in Higher Grade)', 'Shift Up (Exclude)'])
# Use number_input for sanity check
params['Min_A_Percentile'] = st.sidebar.number_input("Min Acceptable A/A+ Percentile (%)", value=10.0, step=1.0, min_value=0.0, max_value=100.0)

# Standard GPA Scale (can make this configurable too if needed)
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0}
# Need to adjust the assign_letter_grades function and GPA_SCALE if using +/- grades


# --- Main Area Workflow ---
st.header("2. Upload Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"])

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'grading_results' not in st.session_state:
    st.session_state.grading_results = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        st.subheader("Map Columns")
        cols = st.session_state.df.columns.tolist()
        # Add default empty option
        cols_with_none = ["<Select Column>"] + cols

        col_id = st.selectbox("Select Student ID Column (Optional)", options=cols_with_none, index=0)
        col_score = st.selectbox("Select Score Column", options=cols_with_none, index=0)
        col_section = st.selectbox("Select Section Column", options=cols_with_none, index=0)

        # Basic Validation / Preview
        if col_score != "<Select Column>" and col_section != "<Select Column>":
            st.session_state.df[col_score] = pd.to_numeric(st.session_state.df[col_score], errors='coerce')
            st.session_state.df.dropna(subset=[col_score], inplace=True) # Handle non-numeric scores

            st.subheader("Data Preview & Initial Stats")
            st.dataframe(st.session_state.df.head())
            st.write(st.session_state.df[col_score].describe())

            # Initial Histogram
            st.subheader("Initial Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.df[col_score], kde=True, ax=ax)
            st.pyplot(fig)

            # --- Grading Execution ---
            st.header("3. Run Grading Process")
            if st.button("Calculate Grades"):
                with st.spinner("Calculating grades..."):
                    # 1. Calculate Initial Cutoffs
                    initial_cutoffs = calculate_initial_cutoffs(params)

                    # 2. Adjust Boundaries
                    final_cutoffs, adjustment_log = adjust_boundaries(st.session_state.df[col_score], initial_cutoffs, params)

                    # 3. Assign Letter Grades
                    st.session_state.df['Letter_Grade'] = assign_letter_grades(st.session_state.df[col_score], final_cutoffs)

                    # 4. Sanity Checks
                    sanity_flag, sanity_message = perform_sanity_checks(st.session_state.df, 'Letter_Grade', final_cutoffs, params)

                    # 5. Calculate Stats and Plots
                    stats_results = calculate_stats_and_plots(st.session_state.df, col_score, 'Letter_Grade', col_section, GPA_SCALE)

                    # Store results in session state
                    st.session_state.grading_results = {
                        "final_cutoffs": final_cutoffs,
                        "adjustment_log": adjustment_log,
                        "sanity_flag": sanity_flag,
                        "sanity_message": sanity_message,
                        "stats_results": stats_results,
                        "graded_df": st.session_state.df # Store the df with grades
                    }
                st.success("Grading process complete!")

        else:
            st.warning("Please select the Score and Section columns.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.df = None # Reset dataframe on error
        st.session_state.grading_results = None

# --- Display Results ---
if st.session_state.grading_results:
    results = st.session_state.grading_results
    st.header("4. Grading Results")

    # Display Key Info
    st.subheader("Final Cutoffs Used")
    st.json(results['final_cutoffs']) # Display dict nicely

    if results['adjustment_log']:
        with st.expander("Boundary Adjustment Log"):
            for log_entry in results['adjustment_log']:
                st.write(log_entry)
    else:
        st.write("No boundary adjustments were made.")

    st.subheader("Sanity Check")
    if results['sanity_flag']:
        st.warning(results['sanity_message'])
    else:
        st.success(results['sanity_message'])

    # Display Statistics and Plots
    st.subheader("Grade Distributions & Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Overall Distribution:**")
        st.dataframe(results['stats_results']['overall_dist'])
        st.write(f"**Overall Avg GPA:** {results['stats_results']['overall_gpa']:.2f}")
        # Add overall distribution plot
    with col2:
        st.write("**Per Section Avg GPA:**")
        st.dataframe(results['stats_results']['section_stats'])
        st.write(f"**ANOVA Result:** {results['stats_results']['anova_result']}")
        if results['stats_results']['anova_p_value'] is not None and results['stats_results']['anova_p_value'] < 0.05:
            st.warning("Statistically significant difference found between section GPAs (p < 0.05).")
        # Add section GPA boxplot

    # Display Section Distributions Table (optional)
    with st.expander("Per Section Grade Distributions (%)"):
         st.dataframe(results['stats_results']['section_dist'].style.format("{:.1%}"))

    # Display Final Grades Table
    st.subheader("Assigned Grades")
    st.dataframe(results['graded_df'][[col_id if col_id != "<Select Column>" else cols[0], col_score, col_section, 'Letter_Grade', 'GPA']]) # Show relevant columns

    # Download Button
    @st.cache_data # Cache the conversion to avoid re-running
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(results['graded_df'])
    st.download_button(
        label="Download Grades as CSV",
        data=csv_data,
        file_name='graded_student_scores.csv',
        mime='text/csv',
    )

elif uploaded_file and (col_score == "<Select Column>" or col_section == "<Select Column>"):
     st.info("Select Score and Section columns to proceed.")
