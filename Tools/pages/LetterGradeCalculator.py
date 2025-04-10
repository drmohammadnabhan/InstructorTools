import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For ANOVA (still useful for evaluation)
from io import BytesIO # For download button
import collections # To use OrderedDict for cutoffs

# ============================================
# Helper Functions
# ============================================

def calculate_initial_cutoffs_original(a_plus_start, gap):
    """Calculates initial cutoffs working downwards from A+ start with a uniform gap."""
    cutoffs = collections.OrderedDict()
    grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D'] # Standard grades to define boundaries for
    current_start = a_plus_start
    # Define the start score for each grade
    for i, grade in enumerate(grades):
        # The key will be the start score threshold for this grade
        cutoffs[f'{grade}_Start'] = current_start
        if i < len(grades) - 1: # Don't subtract gap after D
             current_start -= gap
    # The 'F' range is below the 'D_Start'
    cutoffs['F_Max'] = cutoffs['D_Start'] # Scores < D_Start are F
    return cutoffs

def assign_letter_grades_from_starts(scores, start_cutoffs):
    """Assigns letter grades based on start scores (lower bound inclusive)."""
    # Create bins and labels from the start cutoffs dictionary
    # Sort grades based on start score (descending)
    sorted_grades = sorted(start_cutoffs.items(), key=lambda item: item[1], reverse=True)

    bins = [-np.inf]
    labels = ['F'] # Start with F for scores below the lowest threshold (D_Start)

    # Iterate through sorted grades (A+, A, B+, ...) to define bins and labels
    for grade, start_score in sorted_grades:
        if grade == 'F_Max': continue # Skip the F_Max entry
        # The upper bound of the previous grade's bin is the start score of the current grade
        bins.append(start_score)
        # The label corresponds to the grade that starts at this score
        labels.append(grade) # Label for the interval [start_score, next_start_score)

    # Add the top bin edge
    bins.append(np.inf)

    # Need to reverse labels because pd.cut assigns label[i] to bin[i] to bin[i+1]
    # Example: bins=[-inf, 60, 65,...], labels=['F','D','C',...]
    # Interval (-inf, 60) gets label 'F', interval [60, 65) gets 'D' etc.
    # So labels should be in ascending grade order F, D, D+, C...
    labels = ['F'] + [grade for grade, score in sorted(start_cutoffs.items(), key=lambda item: item[1]) if grade != 'F_Max']

    # Ensure correct number of labels for bins
    if len(labels) != len(bins) - 1:
        st.error(f"Mismatch between number of bins ({len(bins)}) and labels ({len(labels)}) generated. Check cutoff logic.")
        # Create a dummy Series to avoid crashing later code
        return pd.Series(['Error'] * len(scores), index=scores.index)


    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels, right=False, ordered=True)
    return grades


def calculate_stats(df, grade_col, section_col, gpa_map):
    """Calculates distributions and statistics based on assigned grades."""
    # Map grades to GPA points
    df['GPA'] = df[grade_col].astype(str).map(gpa_map) # Use astype(str) for robustness
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')

    if df['GPA'].isnull().all() and not df[grade_col].isnull().all():
         st.warning("Could not map assigned grades to GPA values. Check GPA Scale.")
         # Allow stats based on grade counts anyway

    # Overall stats
    overall_dist = df[grade_col].value_counts(normalize=True).sort_index()
    overall_gpa = df['GPA'].mean() # NaNs are ignored

    # Per-section stats
    df[section_col] = df[section_col].astype(str)
    section_stats = df.groupby(section_col).agg(
        Avg_GPA=('GPA', 'mean'),
        Count=('GPA', 'size'),
        Valid_GPA_Count = ('GPA', 'count')
    ).reset_index()

    anova_result = "ANOVA not applicable."
    anova_p_value = None
    section_groups = [group['GPA'].dropna().values for name, group in df.groupby(section_col) if group['GPA'].notna().sum() > 1]
    if len(section_groups) > 1:
        valid_groups = [g for g in section_groups if len(g) > 0]
        if len(valid_groups) > 1:
            try:
                f_val, p_val = stats.f_oneway(*valid_groups)
                anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"
                anova_p_value = p_val
            except ValueError as e:
                anova_result = f"ANOVA Error: {e}"

    results = {
        "overall_dist": overall_dist,
        "overall_gpa": overall_gpa,
        "section_stats": section_stats,
        "anova_result": anova_result,
        "anova_p_value": anova_p_value
    }
    return results


# Standard GPA Scale
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0}


# ============================================
# Streamlit App Layout
# ============================================

st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant")
st.markdown("""
This tool supports an **iterative grading process**. You define initial cutoffs based on an A+ start score and a uniform gap.
Then, you can **visualize** the score distribution and students near boundaries, **manually adjust** the start score for each grade,
and **recalculate** statistics until you are satisfied with the result.
""")

st.info("**Workflow:**\n"
        "1. Set **Initial Parameters** (A+ Start, Uniform Gap) in the sidebar.\n"
        "2. Click **'Calculate Initial Cutoffs'**.\n"
        "3. Upload your **Score File** and **Map Columns**.\n"
        "4. Review the **Initial Cutoffs**, **Visualizations**, and **Students Near Boundaries**.\n"
        "5. **Manually Adjust** the start scores for each grade in the 'Manual Cutoff Adjustment' section if needed.\n"
        "6. Click **'Apply Manual Cutoffs & Recalculate'** to see the impact of your changes.\n"
        "7. Repeat steps 1-2 or 5-6 until satisfied.\n"
        "8. Review the **Final Results** and **Download** the grades.")


# --- Sidebar for Initial Parameters ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score (>= score)", value=95.0, step=0.5, help="The minimum score to get an A+.")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap (points)", value=5.0, step=0.5, min_value=0.5, help="Initial point difference between consecutive grade start scores (e.g., A starts 'gap' points below A+).")
points_near_cutoff = st.sidebar.slider("Show Students Near Cutoff (+/- points)", min_value=0.5, max_value=5.0, value=1.5, step=0.5, help="Range around active cutoffs to highlight students for review.")

# Initialize session state
if 'initial_cutoffs' not in st.session_state: st.session_state.initial_cutoffs = None
if 'active_cutoffs' not in st.session_state: st.session_state.active_cutoffs = None # Stores either initial or manually adjusted
if 'df_graded' not in st.session_state: st.session_state.df_graded = None
if 'stats_results' not in st.session_state: st.session_state.stats_results = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}


# --- Button to Calculate Initial Cutoffs ---
if st.sidebar.button("Calculate Initial Cutoffs"):
    st.session_state.initial_cutoffs = calculate_initial_cutoffs_original(a_plus_start_score, uniform_grade_gap)
    st.session_state.active_cutoffs = st.session_state.initial_cutoffs # Initially, active = initial
    st.session_state.manual_override_values = st.session_state.initial_cutoffs.copy() # Populate manual fields
    # Clear previous results if recalculating initial
    st.session_state.df_graded = None
    st.session_state.stats_results = None
    st.sidebar.success("Initial cutoffs calculated.")


# --- Main Area ---

# Display Initial/Active Cutoffs if calculated
if st.session_state.active_cutoffs:
    st.header("Current Active Cutoffs")
    cutoff_df = pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Grade Start / F Max', 'Score Threshold'])
    st.dataframe(cutoff_df.style.format({"Score Threshold": "{:.2f}"}))
else:
    st.warning("Calculate initial cutoffs using the sidebar button first.")


st.header("2. Upload & Prepare Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"])

df_display = None # Define df_display to hold the dataframe for display/analysis

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.subheader("Map Columns")
        cols = df_upload.columns.tolist()
        cols_with_none = ["<Select Column>"] + cols

        col_id = st.selectbox("Select Student ID Column (Optional)", options=cols_with_none, index=0, key='sel_id_orig')
        col_score = st.selectbox("Select Score Column", options=cols_with_none, index=0, key='sel_score_orig')
        col_section = st.selectbox("Select Section Column", options=cols_with_none, index=0, key='sel_section_orig')

        if col_score != "<Select Column>" and col_section != "<Select Column>":
             # Process the dataframe
             df = df_upload.copy()
             df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
             if col_id != "<Select Column>":
                  df.rename(columns={col_id: 'StudentID'}, inplace=True)
             else:
                  df['StudentID'] = 'Stud_' + df.index.astype(str)

             df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
             initial_rows = len(df)
             df.dropna(subset=['Score'], inplace=True)
             removed_rows = initial_rows - len(df)
             if removed_rows > 0: st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")

             if df.empty:
                  st.error("No valid score data remaining.")
             else:
                  df['Section'] = df['Section'].astype(str)
                  df_display = df # Assign the processed dataframe
                  st.success("Data loaded and columns mapped.")

        else:
             st.warning("Please select Score and Section columns.")

    except Exception as e:
        st.error(f"Error loading or processing file: {e}")


# --- Sections that require data and active cutoffs ---
if df_display is not None and st.session_state.active_cutoffs is not None:

    st.header("3. Manual Cutoff Adjustment")
    st.markdown("Review visualizations below. If needed, adjust the **Start Score** for each grade here and click 'Apply'.")

    manual_cutoffs_input = {}
    # Use columns for better layout if many grades
    col_a_plus, col_a, col_b_plus, col_b = st.columns(4)
    col_c_plus, col_c, col_d_plus, col_d = st.columns(4)

    # Dynamically create inputs based on active cutoffs, populate with current values
    current_manual_vals = st.session_state.manual_override_values
    # This assumes cutoffs dict keys are like 'A+_Start', 'A_Start' etc.
    with col_a_plus:
        manual_cutoffs_input['A+_Start'] = st.number_input("A+ Start", value=current_manual_vals.get('A+_Start', 95.0), step=0.1, key='man_A+', format="%.2f")
    with col_a:
        manual_cutoffs_input['A_Start'] = st.number_input("A Start", value=current_manual_vals.get('A_Start', 90.0), step=0.1, key='man_A', format="%.2f")
    with col_b_plus:
        manual_cutoffs_input['B+_Start'] = st.number_input("B+ Start", value=current_manual_vals.get('B+_Start', 85.0), step=0.1, key='man_B+', format="%.2f")
    with col_b:
        manual_cutoffs_input['B_Start'] = st.number_input("B Start", value=current_manual_vals.get('B_Start', 80.0), step=0.1, key='man_B', format="%.2f")
    with col_c_plus:
        manual_cutoffs_input['C+_Start'] = st.number_input("C+ Start", value=current_manual_vals.get('C+_Start', 75.0), step=0.1, key='man_C+', format="%.2f")
    with col_c:
        manual_cutoffs_input['C_Start'] = st.number_input("C Start", value=current_manual_vals.get('C_Start', 70.0), step=0.1, key='man_C', format="%.2f")
    with col_d_plus:
        manual_cutoffs_input['D+_Start'] = st.number_input("D+ Start", value=current_manual_vals.get('D+_Start', 65.0), step=0.1, key='man_D+', format="%.2f")
    with col_d:
        manual_cutoffs_input['D_Start'] = st.number_input("D Start", value=current_manual_vals.get('D_Start', 60.0), step=0.1, key='man_D', format="%.2f")

    # Add F_Max for consistency, though it shouldn't change unless D_Start changes
    manual_cutoffs_input['F_Max'] = manual_cutoffs_input['D_Start']


    if st.button("Apply Manual Cutoffs & Recalculate"):
        # Basic validation: check if start scores are descending
        scores_list = [s for g, s in manual_cutoffs_input.items() if g != 'F_Max']
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)):
             st.error("Manual Start Scores must be in descending order (A+ >= A >= B+...). Please correct.")
        else:
             # Update active cutoffs and stored manual values
             st.session_state.active_cutoffs = collections.OrderedDict(sorted(manual_cutoffs_input.items(), key=lambda item: item[1], reverse=True))
             st.session_state.manual_override_values = manual_cutoffs_input.copy()
             # Clear previous results to force recalculation
             st.session_state.df_graded = None
             st.session_state.stats_results = None
             st.success("Manual cutoffs applied. Recalculating results...")
             # Rerun the grade assignment and stats calculation implicitly on next page load
             # Or trigger explicitly here:
             df_display['Letter_Grade'] = assign_letter_grades_from_starts(df_display['Score'], st.session_state.active_cutoffs)
             st.session_state.stats_results = calculate_stats(df_display, 'Letter_Grade', 'Section', GPA_SCALE)
             st.session_state.df_graded = df_display # Store df with new grades


    st.header("4. Visualization & Observation")
    st.markdown("Use these visualizations to assess the impact of the **Active Cutoffs**.")

    active_cutoff_values = [s for g, s in st.session_state.active_cutoffs.items() if g != 'F_Max']

    # --- Histogram ---
    st.subheader("Score Distribution with Active Cutoffs")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(df_display['Score'], kde=True, ax=ax_hist, bins=30) # Allow adjusting bins?
    for cutoff in active_cutoff_values:
        ax_hist.axvline(cutoff, color='red', linestyle='--', linewidth=1)
    ax_hist.set_title("Score Distribution")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # --- Strip Plot (Shows individual points) ---
    st.subheader("Individual Scores with Active Cutoffs")
    fig_strip, ax_strip = plt.subplots(figsize=(10, 4))
    sns.stripplot(x=df_display['Score'], ax=ax_strip, jitter=0.2, size=3, alpha=0.7)
    for cutoff in active_cutoff_values:
        ax_strip.axvline(cutoff, color='red', linestyle='--', linewidth=1)
    ax_strip.set_title("Individual Scores")
    st.pyplot(fig_strip)
    plt.close(fig_strip)

    # --- Students Near Cutoffs ---
    st.subheader("Students Near Active Cutoffs")
    students_near_cutoff_list = []
    df_temp_graded = df_display.copy() # Use a copy for temporary grade assignment based on active cutoffs
    df_temp_graded['Temp_Grade'] = assign_letter_grades_from_starts(df_temp_graded['Score'], st.session_state.active_cutoffs)

    for grade_start_key, cutoff_score in st.session_state.active_cutoffs.items():
        if grade_start_key == 'F_Max': continue
        min_score = cutoff_score - points_near_cutoff
        max_score = cutoff_score + points_near_cutoff
        nearby_df = df_temp_graded[(df_temp_graded['Score'] >= min_score) & (df_temp_graded['Score'] < max_score)].copy() # Use < max_score for range
        if not nearby_df.empty:
             nearby_df['Near_Boundary'] = f"{grade_start_key} ({cutoff_score:.2f})"
             students_near_cutoff_list.append(nearby_df[['StudentID', 'Score', 'Section', 'Near_Boundary', 'Temp_Grade']])

    if students_near_cutoff_list:
        students_near_df = pd.concat(students_near_cutoff_list).sort_values('Score')
        st.dataframe(students_near_df)
    else:
        st.write("No students found within the specified range of active cutoffs.")


    # --- Trigger Calculation if not already done by button ---
    # This section ensures results are shown if data/cutoffs are ready
    if st.session_state.df_graded is None or st.session_state.stats_results is None:
         df_display['Letter_Grade'] = assign_letter_grades_from_starts(df_display['Score'], st.session_state.active_cutoffs)
         st.session_state.stats_results = calculate_stats(df_display, 'Letter_Grade', 'Section', GPA_SCALE)
         st.session_state.df_graded = df_display

    # --- Display Final Results ---
    st.header("5. Final Results (Based on Active Cutoffs)")
    if st.session_state.stats_results:
        results = st.session_state.stats_results
        col_res1, col_res2 = st.columns(2)
        with col_res1:
             st.write("**Overall Distribution:**")
             st.dataframe(results['overall_dist'].apply("{:.1%}".format))
             st.write(f"**Overall Avg GPA:** {results.get('overall_gpa', 'N/A'):.2f}")
        with col_res2:
             st.write("**Per Section Avg GPA:**")
             st.dataframe(results['section_stats'][['Section', 'Avg_GPA']].style.format({"Avg_GPA": "{:.2f}"}))
             st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
             anova_p = results.get('anova_p_value')
             if anova_p is not None and anova_p < 0.05: st.warning("Significant difference in section GPAs detected.")

        st.subheader("Final Assigned Grades")
        if st.session_state.df_graded is not None:
             display_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
             display_cols = [col for col in display_cols if col in st.session_state.df_graded.columns]
             st.dataframe(st.session_state.df_graded[display_cols])

             # Download Button
             @st.cache_data
             def convert_df_to_csv_orig(df_to_convert):
                 dl_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
                 dl_cols = [col for col in dl_cols if col in df_to_convert.columns]
                 return df_to_convert[dl_cols].to_csv(index=False).encode('utf-8')

             try:
                csv_data = convert_df_to_csv_orig(st.session_state.df_graded)
                st.download_button(label="Download Final Grades as CSV", data=csv_data, file_name='final_grades.csv', mime='text/csv')
             except Exception as e:
                st.error(f"Could not prepare download file: {e}")

        else:
             st.warning("Final grade assignments not yet calculated.")
    else:
         st.warning("Statistics could not be calculated. Check data and cutoffs.")


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Iterative Grading Tool v0.9")
