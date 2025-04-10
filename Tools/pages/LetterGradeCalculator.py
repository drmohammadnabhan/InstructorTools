import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For ANOVA
from io import BytesIO # For download button
import collections # To use OrderedDict for cutoffs
import random # For assigning random colors to sections

# ============================================
# Helper Functions (with corrections and enhancements)
# ============================================

def calculate_initial_cutoffs_original(a_plus_start, gap):
    """Calculates initial cutoffs working downwards from A+ start with a uniform gap."""
    cutoffs = collections.OrderedDict()
    grades_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']
    current_start = a_plus_start
    for i, grade in enumerate(grades_order):
        cutoffs[f'{grade}_Start'] = current_start
        if i < len(grades_order) - 1:
             current_start -= gap
    cutoffs['F_Max'] = cutoffs['D_Start']
    return cutoffs

def assign_letter_grades_from_starts(scores, start_cutoffs):
    """Assigns letter grades based on start scores (lower bound inclusive). Handles non-monotonic checks."""
    lower_bounds_map = {grade: score for grade, score in start_cutoffs.items() if grade != 'F_Max'}
    boundary_scores = sorted(list(set(lower_bounds_map.values())))

    if not all(boundary_scores[i] < boundary_scores[i+1] for i in range(len(boundary_scores)-1)):
        # Check if there are only duplicate scores, which is acceptable if minimal
        unique_scores_count = len(boundary_scores)
        total_scores_count = len([s for g,s in lower_bounds_map.items()])
        if unique_scores_count < total_scores_count - 1 : # Allow one potential duplicate pair maybe? More robust check needed.
            st.error("Grade boundary scores must increase monotonically. Check manual inputs.")
            return pd.Series(['Error - Non-monotonic Cutoffs'] * len(scores), index=scores.index, dtype='object')
        else:
            # If only minor duplicates, proceed but warn
            st.warning("Duplicate start scores detected; using unique boundaries.")


    bins = [-np.inf] + boundary_scores + [np.inf]
    grades_sorted_by_score = sorted(lower_bounds_map.keys(), key=lambda grade: lower_bounds_map[grade])
    labels = ['F'] + grades_sorted_by_score

    if len(labels) != len(bins) - 1:
        st.error(f"CRITICAL: Bin/Label mismatch. Bins: {len(bins)}, Labels: {len(labels)}. Check logic.")
        return pd.Series(['Error - Bin/Label Mismatch'] * len(scores), index=scores.index, dtype='object')

    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels, right=False, ordered=True)
    grades = grades.astype('object').fillna('Invalid Score')
    return grades


def calculate_stats(df, grade_col, section_col, gpa_map):
    """Calculates distributions and statistics based on assigned grades. Corrected GPA handling."""
    if grade_col not in df.columns or df[grade_col].astype(str).str.contains('Error', na=False).any():
        st.warning("Cannot calculate stats due to errors in grade assignment.")
        return {"error": "Grade assignment failed."}

    # --- GPA Calculation ---
    # Map grades to GPA points (Removed .astype(str) - map works on category)
    df['GPA'] = df[grade_col].map(gpa_map)
    # Ensure the GPA column is numeric, coercing errors to NaN
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
    # --- End GPA Calculation ---

    if df['GPA'].isnull().all() and not df[grade_col].isnull().all():
         st.warning("Could not map assigned grades to GPA values. Check GPA Scale.")
         # Proceed with grade-based stats

    # Overall stats
    overall_dist = df[grade_col].value_counts(normalize=True).sort_index()
    # Calculate mean GPA only if there are valid numbers, otherwise report N/A
    overall_gpa = df['GPA'].mean() if df['GPA'].notna().any() else np.nan

    # Per-section stats
    df[section_col] = df[section_col].astype(str)
    # Calculate mean GPA per section, handling cases where a section might have all NaNs
    section_stats = df.groupby(section_col).agg(
        Avg_GPA=('GPA', lambda x: x.mean(skipna=True) if x.notna().any() else np.nan), # Calculate mean only if non-NaN values exist
        Count=('GPA', 'size'),
        Valid_GPA_Count = ('GPA', 'count') # count ignores NaNs
    ).reset_index()

    # ANOVA
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
            except ValueError as e: anova_result = f"ANOVA Error: {e}"

    results = {
        "overall_dist": overall_dist,
        "overall_gpa": overall_gpa,
        "section_stats": section_stats,
        "anova_result": anova_result,
        "anova_p_value": anova_p_value,
        "error": None
    }
    return results


# Standard GPA Scale - includes mapping for potential error values
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0,
             'Invalid Score': np.nan, 'Error - Non-monotonic Cutoffs': np.nan, 'Error - Bin/Label Mismatch': np.nan}

# --- Styling Functions for Final Table ---
GRADE_COLORS = { # Example Colors - Adjust as needed
    'A+': '#d1f7c4', 'A': '#e1f7d5',
    'B+': '#ffffcc', 'B': '#ffffdd',
    'C+': '#fde4d6', 'C': '#fdeae0',
    'D+': '#ffdddd', 'D': '#ffe5e5',
    'F': '#ffcccc',
    'default': '#ffffff' # White for errors or others
}

# Generate section colors dynamically
SECTION_COLORS = {}
def get_section_color(section_name):
    if section_name not in SECTION_COLORS:
        # Generate a random light pastel color
        r = random.randint(210, 255)
        g = random.randint(210, 255)
        b = random.randint(210, 255)
        SECTION_COLORS[section_name] = f'rgb({r},{g},{b})'
    return SECTION_COLORS[section_name]

def highlight_grade(s):
    '''Applies background color based on grade column.'''
    return [f'background-color: {GRADE_COLORS.get(grade, GRADE_COLORS["default"])}' for grade in s]

def highlight_section(s):
    '''Applies background color based on section column.'''
    return [f'background-color: {get_section_color(section)}' for section in s]


# ============================================
# Streamlit App Layout
# ============================================

st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant v1.1") # Version update
# ... (Instructions markdown remains largely the same) ...
st.info("**Workflow:**\n"
        "1. Set **Initial Parameters** (A+ Start, Uniform Gap) in the sidebar.\n"
        "2. Click **'Calculate Initial Cutoffs'**.\n"
        "3. Upload your **Score File** and **Map Columns** (including optional Name columns).\n"
        "4. Review the **Initial Cutoffs**, **Visualizations**, and **Students Near Boundaries** (now shows Points to Upgrade).\n"
        "5. **Manually Adjust** the start scores for each grade in the 'Manual Cutoff Adjustment' section if needed.\n"
        "6. Click **'Apply Manual Cutoffs & Recalculate'** to see the impact of your changes.\n"
        "7. Repeat steps 1-2 or 5-6 until satisfied.\n"
        "8. Review the **Final Results**, including **Failing Student Analysis**, and **Download** the grades (full or per section).")


# --- Sidebar for Initial Parameters ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score (>= score)", value=95.0, step=0.1, format="%.2f", help="The minimum score to get an A+.")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap (points)", value=5.0, step=0.1, min_value=0.1, format="%.2f", help="Initial point difference between consecutive grade start scores.")
points_near_cutoff = st.sidebar.slider("Show Students Near Cutoff (+/- points)", min_value=0.1, max_value=5.0, value=1.5, step=0.1, format="%.1f", help="Range around active cutoffs to highlight students.")

# Initialize session state
if 'initial_cutoffs' not in st.session_state: st.session_state.initial_cutoffs = None
if 'active_cutoffs' not in st.session_state: st.session_state.active_cutoffs = None
if 'df_graded' not in st.session_state: st.session_state.df_graded = None
if 'stats_results' not in st.session_state: st.session_state.stats_results = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'processed_df' not in st.session_state: st.session_state.processed_df = None # Store processed data


# --- Button to Calculate Initial Cutoffs ---
if st.sidebar.button("Calculate Initial Cutoffs"):
    st.session_state.initial_cutoffs = calculate_initial_cutoffs_original(a_plus_start_score, uniform_grade_gap)
    st.session_state.active_cutoffs = st.session_state.initial_cutoffs
    # Populate manual fields based on calculation result
    manual_vals = {grade: score for grade, score in st.session_state.initial_cutoffs.items()}
    # Ensure all expected keys exist for the manual input section
    for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']:
        if f'{grade}_Start' not in manual_vals: manual_vals[f'{grade}_Start'] = 0.0 # Default if missing
    if 'F_Max' not in manual_vals: manual_vals['F_Max'] = manual_vals.get('D_Start', 0.0)
    st.session_state.manual_override_values = manual_vals

    st.session_state.df_graded = None
    st.session_state.stats_results = None
    st.sidebar.success("Initial cutoffs calculated.")
    if st.session_state.data_loaded: st.experimental_rerun()


# --- Main Area ---
cutoff_display_area = st.empty()
if st.session_state.active_cutoffs:
    cutoff_display_area.header("Current Active Cutoffs")
    cutoff_df = pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Grade Start / F Max', 'Score Threshold'])
    cutoff_display_area.dataframe(cutoff_df.style.format({"Score Threshold": "{:.2f}"}))
else:
    cutoff_display_area.warning("Calculate initial cutoffs using the sidebar button first.")


st.header("2. Upload & Prepare Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"], key="file_uploader_v2")


col_id, col_score, col_section, col_first, col_last = None, None, None, None, None # Define variables

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")

        st.subheader("Map Columns")
        cols = df_upload.columns.tolist()
        cols_with_none = ["<Select Column>"] + cols

        # --- Name Columns Added ---
        col_first = st.selectbox("Select First Name Column (Optional)", options=cols_with_none, index=0, key='sel_first_v2')
        col_last = st.selectbox("Select Last Name Column (Optional)", options=cols_with_none, index=0, key='sel_last_v2')
        col_id = st.selectbox("Select Student ID Column (Optional)", options=cols_with_none, index=0, key='sel_id_orig_v3')
        col_score = st.selectbox("Select Score Column*", options=cols_with_none, index=0, key='sel_score_orig_v3')
        col_section = st.selectbox("Select Section Column*", options=cols_with_none, index=0, key='sel_section_orig_v3')
        st.caption("*Mandatory columns")

        if col_score != "<Select Column>" and col_section != "<Select Column>":
             df = df_upload.copy()
             # Rename mandatory columns
             df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
             # Handle Optional Columns
             if col_id != "<Select Column>": df.rename(columns={col_id: 'StudentID'}, inplace=True)
             if col_first != "<Select Column>": df.rename(columns={col_first: 'FirstName'}, inplace=True)
             if col_last != "<Select Column>": df.rename(columns={col_last: 'LastName'}, inplace=True)

             # Create default ID if needed
             if 'StudentID' not in df.columns: df['StudentID'] = 'Stud_' + df.index.astype(str)
             # Create Full Name if possible
             if 'FirstName' in df.columns and 'LastName' in df.columns:
                  df['Name'] = df['FirstName'].astype(str) + ' ' + df['LastName'].astype(str)
             elif 'FirstName' in df.columns: df['Name'] = df['FirstName']
             elif 'LastName' in df.columns: df['Name'] = df['LastName']
             else: df['Name'] = df['StudentID'] # Fallback to ID if no name cols


             # Data Cleaning
             essential_cols = ['Score', 'Section', 'StudentID', 'Name'] # Keep Name now
             df = df[[col for col in essential_cols if col in df.columns]] # Keep only needed existing cols

             df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
             initial_rows = len(df)
             df.dropna(subset=['Score'], inplace=True)
             removed_rows = initial_rows - len(df)
             if removed_rows > 0: st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")

             if df.empty:
                  st.error("No valid score data remaining.")
                  st.session_state.data_loaded = False
                  st.session_state.processed_df = None
             else:
                  df['Section'] = df['Section'].astype(str)
                  st.session_state.processed_df = df
                  st.session_state.data_loaded = True
                  st.success("Data loaded and columns mapped.")

                  # Display Preview only after successful processing
                  st.subheader("Data Preview")
                  st.dataframe(st.session_state.processed_df[['Name', 'Score', 'Section']].head())
                  st.write(st.session_state.processed_df['Score'].describe())

                  # Initial Histogram
                  st.subheader("Initial Score Distribution")
                  fig_hist, ax_hist = plt.subplots()
                  sns.histplot(st.session_state.processed_df['Score'], kde=True, ax=ax_hist, bins=30)
                  ax_hist.set_title("Score Distribution")
                  st.pyplot(fig_hist)
                  plt.close(fig_hist)

                  # Reset results display if data is reloaded
                  st.session_state.df_graded = None
                  st.session_state.stats_results = None

        else:
             st.warning("Please select the Score and Section columns.")
             st.session_state.data_loaded = False
             st.session_state.processed_df = None

    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        st.session_state.data_loaded = False
        st.session_state.processed_df = None


# --- Sections that require data and active cutoffs ---
if st.session_state.data_loaded and st.session_state.processed_df is not None and st.session_state.active_cutoffs is not None:
    df_display = st.session_state.processed_df # Use the processed data

    # --- Manual Cutoff Adjustment ---
    st.header("3. Manual Cutoff Adjustment")
    # ... (Manual cutoff input section remains the same as previous version) ...
    st.markdown("Review visualizations below. If needed, adjust the **Start Score** for each grade here and click 'Apply'.")
    manual_cutoffs_input = {}
    col_a_plus, col_a, col_b_plus, col_b = st.columns(4)
    col_c_plus, col_c, col_d_plus, col_d = st.columns(4)
    current_manual_vals = st.session_state.manual_override_values
    # Grade Order for inputs (match common expectation)
    grade_keys_in_order = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    cols_map = {0:col_a_plus, 1:col_a, 2:col_b_plus, 3:col_b, 4:col_c_plus, 5:col_c, 6:col_d_plus, 7:col_d}

    for i, key in enumerate(grade_keys_in_order):
        with cols_map[i]:
             grade_label = key.replace('_Start', ' Start')
             # Ensure key exists in current_manual_vals before accessing
             default_val = current_manual_vals.get(key, 0.0)
             manual_cutoffs_input[key] = st.number_input(
                 grade_label, value=float(default_val), step=0.1, key=f'man_{key}_v3', format="%.2f"
             )

    manual_cutoffs_input['F_Max'] = manual_cutoffs_input['D_Start']

    if st.button("Apply Manual Cutoffs & Recalculate"):
        scores_list = [manual_cutoffs_input[key] for key in grade_keys_in_order]
        # Allow non-decreasing scores (e.g., B+ and B starting at same point is technically okay)
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)):
             st.error("Manual Start Scores must be in descending or equal order (A+ >= A >= B+...). Please correct.")
        else:
             st.session_state.active_cutoffs = collections.OrderedDict(sorted(manual_cutoffs_input.items(), key=lambda item: item[1], reverse=True))
             st.session_state.manual_override_values = manual_cutoffs_input.copy()
             st.session_state.df_graded = None
             st.session_state.stats_results = None
             st.success("Manual cutoffs applied. Recalculating results...")
             st.experimental_rerun() # Trigger rerun to apply changes


    # Perform grade assignment and stats calculation using active cutoffs
    # This needs to run on each script run after cutoffs & data are ready
    try:
        df_calc = df_display.copy() # Use the processed data
        df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)

        if not df_calc['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
            st.session_state.stats_results = calculate_stats(df_calc, 'Letter_Grade', 'Section', GPA_SCALE)
            st.session_state.df_graded = df_calc # Store successfully graded df
        else:
            st.error("Could not calculate statistics due to errors in grade assignment (check cutoffs/logic).")
            # Reset dependent states
            st.session_state.stats_results = None
            st.session_state.df_graded = None

    except Exception as e:
        st.error(f"An error occurred during grade assignment or stats calculation: {e}")
        st.session_state.stats_results = None
        st.session_state.df_graded = None


    # --- Visualization & Observation ---
    st.header("4. Visualization & Observation")
    st.markdown("Use these visualizations to assess the impact of the **Active Cutoffs**.")

    active_cutoff_values_map = {grade: score for grade, score in st.session_state.active_cutoffs.items() if grade != 'F_Max'}
    active_cutoff_scores_asc = sorted(list(set(active_cutoff_values_map.values()))) # Unique ascending boundaries

    # --- Histogram ---
    st.subheader("Score Distribution with Active Cutoffs")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(df_display['Score'], kde=False, ax=ax_hist, bins=30)
    sns.kdeplot(df_display['Score'], ax=ax_hist, color='orange', warn_singular=False)
    for cutoff in active_cutoff_scores_asc:
        ax_hist.axvline(cutoff, color='red', linestyle='--', linewidth=1)
    ax_hist.set_title("Score Distribution")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # --- Strip Plot ---
    st.subheader("Individual Scores with Active Cutoffs")
    fig_strip, ax_strip = plt.subplots(figsize=(10, 4))
    sns.stripplot(x=df_display['Score'], ax=ax_strip, jitter=0.3, size=3.5, alpha=0.6)
    for cutoff in active_cutoff_scores_asc:
        ax_strip.axvline(cutoff, color='red', linestyle='--', linewidth=1)
    ax_strip.set_title("Individual Scores")
    ax_strip.set_xlabel("Score")
    st.pyplot(fig_strip)
    plt.close(fig_strip)

    # --- Students Near Cutoffs (with Points to Upgrade) ---
    st.subheader(f"Students Near Active Cutoffs (+/- {points_near_cutoff} points)")
    students_near_cutoff_list = []
    df_temp_graded = st.session_state.df_graded if st.session_state.df_graded is not None else df_calc # Use currently graded df

    # Create a mapping from score boundary to the grade it starts
    boundary_to_grade_start = {score: grade for grade, score in active_cutoff_values_map.items()}
    # Create a mapping from score boundary to the NEXT higher score boundary
    next_boundary_map = {score: active_cutoff_scores_asc[i+1] for i, score in enumerate(active_cutoff_scores_asc) if i+1 < len(active_cutoff_scores_asc)}

    if 'Letter_Grade' in df_temp_graded.columns and not df_temp_graded['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
        for boundary_score in active_cutoff_scores_asc:
            grade_starts_here = boundary_to_grade_start.get(boundary_score, "N/A")
            min_score_near = boundary_score - points_near_cutoff
            max_score_near = boundary_score + points_near_cutoff

            nearby_df = df_temp_graded[
                (df_temp_graded['Score'] >= min_score_near) &
                (df_temp_graded['Score'] < max_score_near)
            ].copy()

            if not nearby_df.empty:
                 nearby_df['Near_Boundary_Score'] = boundary_score
                 # Calculate points to *next* boundary if possible
                 next_higher_boundary = next_boundary_map.get(boundary_score, None)
                 if next_higher_boundary is not None:
                     nearby_df['Points_to_Upgrade'] = next_higher_boundary - nearby_df['Score']
                 else: # Likely near A+ boundary
                     nearby_df['Points_to_Upgrade'] = np.nan # Or 0 or N/A

                 students_near_cutoff_list.append(nearby_df)

        if students_near_cutoff_list:
            students_near_df = pd.concat(students_near_cutoff_list).sort_values(['Near_Boundary_Score', 'Score'])
            # Define columns to show, including Name
            cols_near = ['Name', 'Score', 'Section', 'Letter_Grade', 'Near_Boundary_Score', 'Points_to_Upgrade']
            cols_near_exist = [col for col in cols_near if col in students_near_df.columns]
            st.dataframe(students_near_df[cols_near_exist].style.format({
                "Score": "{:.2f}", "Near_Boundary_Score": "{:.2f}", "Points_to_Upgrade": "{:.2f}"
            }))
        else:
            st.write("No students found within the specified range of active cutoffs.")
    else:
        st.warning("Grade calculation needed or failed; cannot show students near cutoffs.")


    # --- Display Final Results ---
    st.header("5. Final Results (Based on Active Cutoffs)")
    if st.session_state.stats_results and st.session_state.stats_results.get("error") is None:
        results = st.session_state.stats_results
        col_res1, col_res2 = st.columns(2)
        with col_res1: # Overall Stats & Plot
             st.write("**Overall Distribution:**")
             if "overall_dist" in results and not results['overall_dist'].empty:
                  st.dataframe(results['overall_dist'].apply("{:.1%}".format))
                  # Format overall GPA correctly, handling NaN
                  overall_gpa_val = results.get('overall_gpa', np.nan)
                  st.write(f"**Overall Avg GPA:** {overall_gpa_val:.2f}" if pd.notna(overall_gpa_val) else "**Overall Avg GPA:** N/A")

                  try: # Overall Dist Plot
                       fig_dist, ax_dist = plt.subplots()
                       results['overall_dist'].sort_index().plot(kind='bar', ax=ax_dist)
                       ax_dist.set_ylabel("Proportion"); ax_dist.set_title("Overall Grade Distribution")
                       plt.xticks(rotation=45); st.pyplot(fig_dist); plt.close(fig_dist)
                  except Exception as e: st.warning(f"Could not plot overall dist: {e}")
             else: st.write("N/A")

        with col_res2: # Section Stats & Plot
             st.write("**Per Section Avg GPA:**")
             if "section_stats" in results and not results['section_stats'].empty:
                   # Format NaN Avg_GPA as N/A explicitly
                   results['section_stats']['Avg_GPA_Formatted'] = results['section_stats']['Avg_GPA'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                   st.dataframe(results['section_stats'][['Section', 'Avg_GPA_Formatted', 'Count']]) # Show count too
             else: st.write("N/A")

             st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
             anova_p = results.get('anova_p_value')
             if anova_p is not None and anova_p < 0.05: st.warning("Significant difference in section GPAs detected.")

             try: # Section GPA Boxplot
                 if st.session_state.df_graded is not None and 'GPA' in st.session_state.df_graded.columns and st.session_state.df_graded['GPA'].notna().any():
                     fig_box, ax_box = plt.subplots()
                     sns.boxplot(data=st.session_state.df_graded, x='Section', y='GPA', ax=ax_box)
                     ax_box.set_title("GPA Distribution by Section")
                     plt.xticks(rotation=45, ha='right'); st.pyplot(fig_box); plt.close(fig_box)
                 else: st.warning("GPA data not available for boxplot.")
             except Exception as e: st.warning(f"Could not generate section GPA boxplot: {e}")

        # --- Failing Students Analysis (with Names) ---
        st.subheader("Failing Students Analysis")
        if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns:
            passing_score = st.session_state.active_cutoffs.get('D_Start', None)
            if passing_score is not None:
                failing_students = st.session_state.df_graded[st.session_state.df_graded['Letter_Grade'] == 'F'].copy()
                if not failing_students.empty:
                    failing_students['Points_Below_Pass'] = passing_score - failing_students['Score']
                    failing_students.sort_values('Points_Below_Pass', ascending=True, inplace=True)
                    st.write(f"Passing Score (D Start): {passing_score:.2f}")
                    cols_fail = ['Name', 'Score', 'Section', 'Points_Below_Pass'] # Added Name
                    cols_fail_exist = [col for col in cols_fail if col in failing_students.columns]
                    st.dataframe(failing_students[cols_fail_exist].style.format({"Score": "{:.2f}", "Points_Below_Pass": "{:.2f}"}))
                else:
                    st.success("No students received an 'F' grade based on active cutoffs.")
            else: st.warning("Could not determine passing score ('D_Start') from active cutoffs.")
        else: st.warning("Final grades needed to analyze failing students.")
        # --- End Failing Students Section ---


        st.subheader("Final Assigned Grades Table")
        if st.session_state.df_graded is not None:
             df_final = st.session_state.df_graded
             display_cols = ['Name', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA'] # Added Name
             display_cols_exist = [col for col in display_cols if col in df_final.columns]
             df_to_display = df_final[display_cols_exist]

             # Apply Styling - Use subset to avoid issues if optional columns don't exist
             styler = df_to_display.style
             if 'Letter_Grade' in df_to_display:
                  styler = styler.apply(highlight_grade, subset=['Letter_Grade'], axis=0)
             if 'Section' in df_to_display:
                   # Pre-generate section colors
                   unique_sections = df_to_display['Section'].unique()
                   for section in unique_sections: get_section_color(section) # Ensure colors exist
                   styler = styler.apply(highlight_section, subset=['Section'], axis=0)

             # Display styled dataframe using HTML
             st.markdown(styler.format({"Score": "{:.2f}", "GPA": "{:.2f}"}).to_html(), unsafe_allow_html=True)


             # --- Download Section ---
             st.subheader("Download Grades")
             sections = ["All Sections"] + sorted(df_final['Section'].unique().tolist())
             selected_section = st.selectbox("Select section to download:", options=sections)

             @st.cache_data
             def convert_df_to_csv_orig(df_to_convert, section_filter):
                 if section_filter != "All Sections":
                     df_filtered = df_to_convert[df_to_convert['Section'] == section_filter].copy()
                 else:
                     df_filtered = df_to_convert.copy()

                 dl_cols = ['Name', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
                 dl_cols_exist = [col for col in dl_cols if col in df_filtered.columns]
                 if not dl_cols_exist: return None
                 return df_filtered[dl_cols_exist].to_csv(index=False).encode('utf-8')

             try:
                csv_data = convert_df_to_csv_orig(df_final, selected_section)
                if csv_data:
                    file_name = f"final_grades_{selected_section.replace(' ', '_')}.csv" if selected_section != "All Sections" else "final_grades_all.csv"
                    st.download_button(label=f"Download Grades for {selected_section}", data=csv_data, file_name=file_name, mime='text/csv')
             except Exception as e: st.error(f"Could not prepare download file: {e}")
             # --- End Download Section ---

        else: st.warning("Final grade assignments not yet calculated.")
    elif st.session_state.active_cutoffs:
         st.warning("Statistics could not be calculated. Check data, grade assignment, and cutoffs.")


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Iterative Grading Tool v1.1")
