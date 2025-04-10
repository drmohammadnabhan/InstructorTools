import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For ANOVA (still useful for evaluation)
from io import BytesIO # For download button
import collections # To use OrderedDict for cutoffs

# ============================================
# Helper Functions (with corrections)
# ============================================

def calculate_initial_cutoffs_original(a_plus_start, gap):
    """Calculates initial cutoffs working downwards from A+ start with a uniform gap."""
    cutoffs = collections.OrderedDict()
    # Define grades in descending order for calculation logic
    grades_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']
    current_start = a_plus_start

    for i, grade in enumerate(grades_order):
        cutoffs[f'{grade}_Start'] = current_start
        if i < len(grades_order) - 1: # Don't subtract gap after D
             current_start -= gap

    # F is implicitly below the D_Start threshold
    cutoffs['F_Max'] = cutoffs['D_Start'] # Use D_Start as the upper boundary for F
    # Sort final dictionary by score descending for potential display consistency elsewhere if needed
    # cutoffs = collections.OrderedDict(sorted(cutoffs.items(), key=lambda item: item[1], reverse=True))
    return cutoffs

# <<< CORRECTED FUNCTION >>>
def assign_letter_grades_from_starts(scores, start_cutoffs):
    """Assigns letter grades based on start scores (lower bound inclusive)."""
    # start_cutoffs format: {'A+_Start': 95, 'A_Start': 90, ..., 'D_Start': 60, 'F_Max': 60}

    # Extract the start scores for each grade (excluding F_Max as it's just D_Start)
    lower_bounds_map = {grade: score for grade, score in start_cutoffs.items() if grade != 'F_Max'}

    # Get the unique start scores and sort them ASCENDING. These are the boundaries.
    boundary_scores = sorted(list(set(lower_bounds_map.values())))

    # Check for monotonicity (this should catch issues from manual input)
    if not all(boundary_scores[i] < boundary_scores[i+1] for i in range(len(boundary_scores)-1)):
        st.error("Grade boundary scores must increase monotonically. Check manual inputs.")
        # Return a Series indicating error
        return pd.Series(['Error - Non-monotonic Cutoffs'] * len(scores), index=scores.index, dtype='object')

    # Create bins for pd.cut. Needs -inf at start, boundaries, +inf at end.
    bins = [-np.inf] + boundary_scores + [np.inf]

    # Create labels corresponding to the bins.
    # Need grades sorted by their start scores (ascending)
    grades_sorted_by_score = sorted(lower_bounds_map.keys(), key=lambda grade: lower_bounds_map[grade])
    # Labels = F, then the grades corresponding to the ascending boundaries
    labels = ['F'] + grades_sorted_by_score

    # Final check for length mismatch between bins and labels
    if len(labels) != len(bins) - 1:
        st.error(f"CRITICAL: Bin/Label mismatch detected. Bins: {len(bins)}, Labels: {len(labels)}. This indicates an internal logic error.")
        return pd.Series(['Error - Bin/Label Mismatch'] * len(scores), index=scores.index, dtype='object')

    numeric_scores = pd.to_numeric(scores, errors='coerce')
    # right=False means intervals are [lower, upper)
    grades = pd.cut(numeric_scores, bins=bins, labels=labels, right=False, ordered=True)
    # Handle potential NaNs from coercion if any scores were non-numeric originally
    grades = grades.astype('object').fillna('Invalid Score') # Convert to object to allow fillna

    return grades
# <<< END CORRECTION >>>


def calculate_stats(df, grade_col, section_col, gpa_map):
    """Calculates distributions and statistics based on assigned grades."""
    # Ensure grade column exists and handle potential error strings from assign_letter_grades
    if grade_col not in df.columns or df[grade_col].str.contains('Error', na=False).any():
        st.warning("Cannot calculate stats due to errors in grade assignment.")
        return {"error": "Grade assignment failed."}

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
        "anova_p_value": anova_p_value,
        "error": None # Indicate success
    }
    return results


# Standard GPA Scale
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0, 'Invalid Score': np.nan, 'Error': np.nan}


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
        "8. Review the **Final Results**, including **Failing Student Analysis**, and **Download** the grades.")


# --- Sidebar for Initial Parameters ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score (>= score)", value=95.0, step=0.1, format="%.2f", help="The minimum score to get an A+.")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap (points)", value=5.0, step=0.1, min_value=0.1, format="%.2f", help="Initial point difference between consecutive grade start scores (e.g., A starts 'gap' points below A+).")
points_near_cutoff = st.sidebar.slider("Show Students Near Cutoff (+/- points)", min_value=0.1, max_value=5.0, value=1.5, step=0.1, format="%.1f", help="Range around active cutoffs to highlight students for review.")

# Initialize session state
if 'initial_cutoffs' not in st.session_state: st.session_state.initial_cutoffs = None
if 'active_cutoffs' not in st.session_state: st.session_state.active_cutoffs = None # Stores either initial or manually adjusted
if 'df_graded' not in st.session_state: st.session_state.df_graded = None
if 'stats_results' not in st.session_state: st.session_state.stats_results = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False


# --- Button to Calculate Initial Cutoffs ---
if st.sidebar.button("Calculate Initial Cutoffs"):
    st.session_state.initial_cutoffs = calculate_initial_cutoffs_original(a_plus_start_score, uniform_grade_gap)
    st.session_state.active_cutoffs = st.session_state.initial_cutoffs # Initially, active = initial
    # Populate manual fields with initial values, ensuring keys exist even if calculation is simple
    st.session_state.manual_override_values = {
        'A+_Start': st.session_state.initial_cutoffs.get('A+_Start', 95.0),
        'A_Start': st.session_state.initial_cutoffs.get('A_Start', 90.0),
        'B+_Start': st.session_state.initial_cutoffs.get('B+_Start', 85.0),
        'B_Start': st.session_state.initial_cutoffs.get('B_Start', 80.0),
        'C+_Start': st.session_state.initial_cutoffs.get('C+_Start', 75.0),
        'C_Start': st.session_state.initial_cutoffs.get('C_Start', 70.0),
        'D+_Start': st.session_state.initial_cutoffs.get('D+_Start', 65.0),
        'D_Start': st.session_state.initial_cutoffs.get('D_Start', 60.0),
        'F_Max': st.session_state.initial_cutoffs.get('F_Max', 60.0)
    }
    # Clear previous results if recalculating initial
    st.session_state.df_graded = None
    st.session_state.stats_results = None
    st.sidebar.success("Initial cutoffs calculated.")
    # Force rerun/update of main panel if data is already loaded
    if st.session_state.data_loaded: st.experimental_rerun()


# --- Main Area ---

# Display Initial/Active Cutoffs if calculated
cutoff_display_area = st.empty() # Placeholder for cutoffs display
if st.session_state.active_cutoffs:
    cutoff_display_area.header("Current Active Cutoffs")
    cutoff_df = pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Grade Start / F Max', 'Score Threshold'])
    cutoff_display_area.dataframe(cutoff_df.style.format({"Score Threshold": "{:.2f}"}))
else:
    cutoff_display_area.warning("Calculate initial cutoffs using the sidebar button first.")


st.header("2. Upload & Prepare Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"], key="file_uploader")

df_display = None # Define df_display to hold the dataframe for display/analysis
if 'processed_df' not in st.session_state: st.session_state.processed_df = None

if uploaded_file:
    # Reset flags if a new file is uploaded
    # Check if file content is different? Simple check by name/size for now.
    # This logic might need refinement based on how session state persists across uploads.
    # A button "Load/Process File" might be more explicit.

    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.subheader("Map Columns")
        cols = df_upload.columns.tolist()
        cols_with_none = ["<Select Column>"] + cols

        col_id = st.selectbox("Select Student ID Column (Optional)", options=cols_with_none, index=0, key='sel_id_orig_v2')
        col_score = st.selectbox("Select Score Column", options=cols_with_none, index=0, key='sel_score_orig_v2')
        col_section = st.selectbox("Select Section Column", options=cols_with_none, index=0, key='sel_section_orig_v2')

        if col_score != "<Select Column>" and col_section != "<Select Column>":
             # Process the dataframe
             df = df_upload.copy()
             df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
             if col_id != "<Select Column>":
                  df.rename(columns={col_id: 'StudentID'}, inplace=True)
             else:
                  # Create default ID only if StudentID doesn't already exist from renaming
                  if 'StudentID' not in df.columns:
                       df['StudentID'] = 'Stud_' + df.index.astype(str)

             # Data Cleaning (only keep essential columns + ID for processing)
             essential_cols = ['Score', 'Section']
             if 'StudentID' in df.columns: essential_cols.append('StudentID')
             df = df[essential_cols] # Keep only needed cols

             df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
             initial_rows = len(df)
             df.dropna(subset=['Score'], inplace=True) # Remove rows with non-numeric scores
             removed_rows = initial_rows - len(df)
             if removed_rows > 0: st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")

             if df.empty:
                  st.error("No valid score data remaining.")
                  st.session_state.data_loaded = False
                  st.session_state.processed_df = None
             else:
                  df['Section'] = df['Section'].astype(str)
                  st.session_state.processed_df = df # Store processed df
                  st.session_state.data_loaded = True
                  st.success("Data loaded and columns mapped.")


        else:
             st.warning("Please select Score and Section columns.")
             st.session_state.data_loaded = False
             st.session_state.processed_df = None

    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        st.session_state.data_loaded = False
        st.session_state.processed_df = None

# Assign df_display if data is loaded
if st.session_state.data_loaded and st.session_state.processed_df is not None:
    df_display = st.session_state.processed_df


# --- Sections that require data and active cutoffs ---
if st.session_state.data_loaded and df_display is not None and st.session_state.active_cutoffs is not None:

    # --- Manual Cutoff Adjustment ---
    st.header("3. Manual Cutoff Adjustment")
    st.markdown("Review visualizations below. If needed, adjust the **Start Score** for each grade here and click 'Apply'.")

    manual_cutoffs_input = {}
    # Use columns for better layout
    col_a_plus, col_a, col_b_plus, col_b = st.columns(4)
    col_c_plus, col_c, col_d_plus, col_d = st.columns(4)

    # Populate inputs with values stored in session state (initially from calculation, then from user)
    # Use unique keys for each input widget
    current_manual_vals = st.session_state.manual_override_values
    with col_a_plus:
        manual_cutoffs_input['A+_Start'] = st.number_input("A+ Start", value=current_manual_vals.get('A+_Start', 95.0), step=0.1, key='man_A+_v2', format="%.2f")
    with col_a:
        manual_cutoffs_input['A_Start'] = st.number_input("A Start", value=current_manual_vals.get('A_Start', 90.0), step=0.1, key='man_A_v2', format="%.2f")
    with col_b_plus:
        manual_cutoffs_input['B+_Start'] = st.number_input("B+ Start", value=current_manual_vals.get('B+_Start', 85.0), step=0.1, key='man_B+_v2', format="%.2f")
    with col_b:
        manual_cutoffs_input['B_Start'] = st.number_input("B Start", value=current_manual_vals.get('B_Start', 80.0), step=0.1, key='man_B_v2', format="%.2f")
    with col_c_plus:
        manual_cutoffs_input['C+_Start'] = st.number_input("C+ Start", value=current_manual_vals.get('C+_Start', 75.0), step=0.1, key='man_C+_v2', format="%.2f")
    with col_c:
        manual_cutoffs_input['C_Start'] = st.number_input("C Start", value=current_manual_vals.get('C_Start', 70.0), step=0.1, key='man_C_v2', format="%.2f")
    with col_d_plus:
        manual_cutoffs_input['D+_Start'] = st.number_input("D+ Start", value=current_manual_vals.get('D+_Start', 65.0), step=0.1, key='man_D+_v2', format="%.2f")
    with col_d:
        manual_cutoffs_input['D_Start'] = st.number_input("D Start", value=current_manual_vals.get('D_Start', 60.0), step=0.1, key='man_D_v2', format="%.2f")

    # Update F_Max based on D_Start input
    manual_cutoffs_input['F_Max'] = manual_cutoffs_input['D_Start']


    if st.button("Apply Manual Cutoffs & Recalculate"):
        # Basic validation: check if start scores are descending (ignoring F_Max)
        scores_list = [s for g, s in manual_cutoffs_input.items() if g != 'F_Max']
        # Check >= allowing boundaries to be equal
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)):
             st.error("Manual Start Scores must be in descending or equal order (A+ >= A >= B+...). Please correct.")
        else:
             # Update active cutoffs and stored manual values
             # Sort the final dict by score to ensure consistency if needed later
             st.session_state.active_cutoffs = collections.OrderedDict(sorted(manual_cutoffs_input.items(), key=lambda item: item[1], reverse=True))
             st.session_state.manual_override_values = manual_cutoffs_input.copy()
             # Clear previous results to force recalculation
             st.session_state.df_graded = None
             st.session_state.stats_results = None
             st.success("Manual cutoffs applied. Recalculating results...")
             # Use experimental_rerun to ensure the rest of the script uses the new state
             st.experimental_rerun()

    # Perform grade assignment and stats calculation based on ACTIVE cutoffs
    # This block runs whenever the script reruns after data is loaded and cutoffs are active
    # Ensure df_display is the processed dataframe from session state
    df_calc = st.session_state.processed_df.copy()
    df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)
    # Check if grade assignment produced errors before calculating stats
    if not df_calc['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
        st.session_state.stats_results = calculate_stats(df_calc, 'Letter_Grade', 'Section', GPA_SCALE)
        st.session_state.df_graded = df_calc # Store df with new grades
    else:
        st.error("Could not calculate statistics due to errors in grade assignment (check cutoffs).")
        st.session_state.stats_results = None
        st.session_state.df_graded = None


    # --- Visualization & Observation ---
    st.header("4. Visualization & Observation")
    st.markdown("Use these visualizations to assess the impact of the **Active Cutoffs**.")

    active_cutoff_values = [s for g, s in st.session_state.active_cutoffs.items() if g != 'F_Max']
    active_cutoff_values = sorted(list(set(active_cutoff_values))) # Unique ascending boundaries

    # --- Histogram ---
    st.subheader("Score Distribution with Active Cutoffs")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(df_display['Score'], kde=False, ax=ax_hist, bins=30) # Allow adjusting bins?
    sns.kdeplot(df_display['Score'], ax=ax_hist, color='orange', warn_singular=False) # Overlay KDE
    for cutoff in active_cutoff_values:
        ax_hist.axvline(cutoff, color='red', linestyle='--', linewidth=1)
    ax_hist.set_title("Score Distribution")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # --- Strip Plot (Shows individual points) ---
    st.subheader("Individual Scores with Active Cutoffs")
    fig_strip, ax_strip = plt.subplots(figsize=(10, 4))
    sns.stripplot(x=df_display['Score'], ax=ax_strip, jitter=0.3, size=3.5, alpha=0.6) # Adjusted jitter/size
    for cutoff in active_cutoff_values:
        ax_strip.axvline(cutoff, color='red', linestyle='--', linewidth=1)
    ax_strip.set_title("Individual Scores")
    ax_strip.set_xlabel("Score")
    st.pyplot(fig_strip)
    plt.close(fig_strip)

    # --- Students Near Cutoffs ---
    st.subheader(f"Students Near Active Cutoffs (+/- {points_near_cutoff} points)")
    students_near_cutoff_list = []
    # Use the graded df from session state if available, otherwise use temp calc
    df_temp_graded = st.session_state.df_graded if st.session_state.df_graded is not None else df_calc

    # Check if 'Letter_Grade' exists before proceeding
    if 'Letter_Grade' in df_temp_graded.columns:
        for grade_start_key, cutoff_score in st.session_state.active_cutoffs.items():
            if grade_start_key == 'F_Max': continue # Don't show near F boundary
            min_score = cutoff_score - points_near_cutoff
            max_score = cutoff_score + points_near_cutoff
            # Ensure score column is numeric before comparison
            nearby_df = df_temp_graded[
                (pd.to_numeric(df_temp_graded['Score'], errors='coerce') >= min_score) &
                (pd.to_numeric(df_temp_graded['Score'], errors='coerce') < max_score)
            ].copy()

            if not nearby_df.empty:
                 nearby_df['Near_Boundary'] = f"{grade_start_key} ({cutoff_score:.2f})"
                 # Select columns, ensure they exist
                 cols_to_show = ['StudentID', 'Score', 'Section', 'Near_Boundary', 'Letter_Grade']
                 cols_exist = [col for col in cols_to_show if col in nearby_df.columns]
                 students_near_cutoff_list.append(nearby_df[cols_exist])

        if students_near_cutoff_list:
            students_near_df = pd.concat(students_near_cutoff_list).sort_values('Score')
            st.dataframe(students_near_df)
        else:
            st.write("No students found within the specified range of active cutoffs.")
    else:
        st.warning("Grade calculation needed to show students near cutoffs.")


    # --- Display Final Results ---
    st.header("5. Final Results (Based on Active Cutoffs)")
    if st.session_state.stats_results and st.session_state.stats_results.get("error") is None:
        results = st.session_state.stats_results
        col_res1, col_res2 = st.columns(2)
        with col_res1:
             st.write("**Overall Distribution:**")
             if "overall_dist" in results and not results['overall_dist'].empty:
                  st.dataframe(results['overall_dist'].apply("{:.1%}".format))
                  st.write(f"**Overall Avg GPA:** {results.get('overall_gpa', 'N/A'):.2f}")

                  # Overall Dist Plot
                  try:
                       fig_dist, ax_dist = plt.subplots()
                       results['overall_dist'].sort_index().plot(kind='bar', ax=ax_dist)
                       ax_dist.set_ylabel("Proportion")
                       ax_dist.set_title("Overall Grade Distribution")
                       plt.xticks(rotation=45)
                       st.pyplot(fig_dist)
                       plt.close(fig_dist)
                  except Exception as e: st.warning(f"Could not plot overall dist: {e}")

             else: st.write("N/A")
        with col_res2:
             st.write("**Per Section Avg GPA:**")
             if "section_stats" in results and not results['section_stats'].empty:
                  st.dataframe(results['section_stats'][['Section', 'Avg_GPA']].style.format({"Avg_GPA": "{:.2f}"}))
             else: st.write("N/A")

             st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
             anova_p = results.get('anova_p_value')
             if anova_p is not None and anova_p < 0.05: st.warning("Significant difference in section GPAs detected.")

        # --- Failing Students Section ---
        st.subheader("Failing Students Analysis")
        if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns:
            passing_score = st.session_state.active_cutoffs.get('D_Start', None) # Get the start score for D
            if passing_score is not None:
                failing_students = st.session_state.df_graded[st.session_state.df_graded['Letter_Grade'] == 'F'].copy()
                if not failing_students.empty:
                    failing_students['Points_Below_Pass'] = passing_score - failing_students['Score']
                    failing_students.sort_values('Points_Below_Pass', ascending=True, inplace=True) # Show closest first
                    st.write(f"Passing Score (D Start): {passing_score:.2f}")
                    cols_fail = ['StudentID', 'Score', 'Section', 'Points_Below_Pass']
                    cols_fail_exist = [col for col in cols_fail if col in failing_students.columns]
                    st.dataframe(failing_students[cols_fail_exist].style.format({"Score": "{:.2f}", "Points_Below_Pass": "{:.2f}"}))
                else:
                    st.success("No students received an 'F' grade based on active cutoffs.")
            else:
                st.warning("Could not determine passing score ('D_Start') from active cutoffs.")
        else:
            st.warning("Final grades needed to analyze failing students.")
        # --- End Failing Students Section ---


        st.subheader("Final Assigned Grades Table")
        if st.session_state.df_graded is not None:
             display_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
             display_cols = [col for col in display_cols if col in st.session_state.df_graded.columns]
             st.dataframe(st.session_state.df_graded[display_cols])

             # Download Button
             @st.cache_data # Cache the conversion
             def convert_df_to_csv_orig(df_to_convert):
                 dl_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
                 dl_cols = [col for col in dl_cols if col in df_to_convert.columns]
                 return df_to_convert[dl_cols].to_csv(index=False).encode('utf-8')

             try:
                csv_data = convert_df_to_csv_orig(st.session_state.df_graded)
                st.download_button(label="Download Final Grades as CSV", data=csv_data, file_name='final_grades.csv', mime='text/csv')
             except Exception as e: st.error(f"Could not prepare download file: {e}")

        else:
             st.warning("Final grade assignments not yet calculated.")
    elif st.session_state.active_cutoffs: # Check if cutoffs exist but stats failed
         st.warning("Statistics could not be calculated. Check data, grade assignment, and cutoffs.")


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Iterative Grading Tool v1.0")
