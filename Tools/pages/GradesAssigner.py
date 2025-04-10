import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For ANOVA
from io import BytesIO # For download button
import collections # To check for ordered dict type if needed

# ============================================
# Define Core Grading Functions (with corrections)
# ============================================

def calculate_initial_cutoffs(params):
    """Calculates initial grade boundaries based on fixed widths."""
    cutoffs = collections.OrderedDict() # Use OrderedDict to maintain order
    cutoffs['F_Upper'] = params['F_Threshold']       # Scores < this are F
    cutoffs['D_Upper'] = params['C_Start_Threshold'] # Scores < this (and >= F_Upper) are D
    cutoffs['C_Upper'] = cutoffs['D_Upper'] + params['C_Width']
    cutoffs['B_Upper'] = cutoffs['C_Upper'] + params['B_Width']
    cutoffs['A_Upper'] = cutoffs['B_Upper'] + params['A_Width'] # Scores >= this (and < 100/max) are A+
    # Add A+_Upper boundary for clarity in pd.cut, assuming 100 is max practical score
    cutoffs['A+_Upper'] = 100.1 # Use a value slightly above max possible score
    return cutoffs

def adjust_boundaries(scores, initial_cutoffs, params):
    """Applies rule-based adjustments to boundaries based on gaps/clusters."""
    adjusted_cutoffs = initial_cutoffs.copy()
    adjustment_log = []
    # Adjust boundaries that define the *start* of C, B, A (which are the upper bounds of D, C, B)
    boundaries_to_check = ['D_Upper', 'C_Upper', 'B_Upper', 'A_Upper']

    for boundary_key in boundaries_to_check:
        # Get the value defining the *start* of the next grade
        cutoff_val = adjusted_cutoffs[boundary_key]
        adjusted = False # Flag to prevent applying both gap and cluster logic

        # --- Gap Check Logic ---
        min_bound = cutoff_val - params['Max_Boundary_Shift']
        max_bound = cutoff_val + params['Max_Boundary_Shift']
        scores_near_cutoff = scores[(scores >= min_bound) & (scores <= max_bound)]
        sorted_unique_near = np.unique(scores_near_cutoff)
        diffs = np.diff(sorted_unique_near)
        # Find if any gap within the check range contains the cutoff_val
        for i, diff in enumerate(diffs):
            if diff >= params['Significant_Gap_Size']:
                gap_start = sorted_unique_near[i]
                gap_end = sorted_unique_near[i+1]
                # Check if the original cutoff falls within this specific gap
                if cutoff_val > gap_start and cutoff_val < gap_end:
                    # Shift to middle of the gap, respecting Max_Boundary_Shift if needed
                    new_cutoff = (gap_start + gap_end) / 2.0
                    # Ensure shift doesn't exceed max allowed magnitude
                    actual_shift = new_cutoff - cutoff_val
                    if abs(actual_shift) > params['Max_Boundary_Shift']:
                       # Limit shift (e.g., shift by max_shift towards gap middle)
                       new_cutoff = cutoff_val + np.sign(actual_shift) * params['Max_Boundary_Shift']

                    adjusted_cutoffs[boundary_key] = new_cutoff
                    adjustment_log.append(f"Shifted {boundary_key} from {cutoff_val:.2f} to {new_cutoff:.2f} due to gap ({gap_start:.1f}-{gap_end:.1f}).")
                    adjusted = True
                    break # Apply only first found gap adjustment for this boundary
        if adjusted: continue # Move to next boundary if adjusted

        # --- Cluster Check Logic ---
        cluster_count = np.sum(scores == cutoff_val)
        cluster_percent = (cluster_count / len(scores)) * 100
        if cluster_percent >= params['Dense_Cluster_Threshold']:
            # Shift slightly based on policy to include/exclude cluster
            # Shift Down includes cluster in the higher grade category
            shift_amount = -0.01 if params['Cluster_Shift_Policy'] == 'Shift Down (Include in Higher Grade)' else 0.01
            # Check if shift exceeds max (unlikely with 0.01 but good practice)
            if abs(shift_amount) <= params['Max_Boundary_Shift']:
                 new_cutoff = cutoff_val + shift_amount
                 adjusted_cutoffs[boundary_key] = new_cutoff
                 adjustment_log.append(f"Shifted {boundary_key} from {cutoff_val:.2f} to {new_cutoff:.2f} due to cluster ({cluster_count} students).")
            else:
                 adjustment_log.append(f"INFO: Cluster found at {boundary_key}={cutoff_val:.2f} but required shift > Max_Boundary_Shift.")


    return adjusted_cutoffs, adjustment_log


def assign_letter_grades(scores, final_cutoffs):
    """Assigns letter grades based on final cutoff boundaries."""
    # Define bins using the upper boundary values from the ordered dict
    bins = [-np.inf] + list(final_cutoffs.values())
    # Define labels corresponding to the intervals created by the bins
    labels = ['F', 'D', 'C', 'B', 'A', 'A+'] # Ensure this matches number of intervals (len(bins)-1)

    grades = pd.cut(scores, bins=bins, labels=labels, right=False, ordered=True) # right=False means [lower, upper)
    return grades

def perform_sanity_checks(df, grade_col, final_cutoffs, params):
    """Performs sanity checks like minimum A/A+ percentage."""
    a_grades_percent = (df[grade_col].isin(['A', 'A+'])).mean() * 100
    flag = a_grades_percent < params['Min_A_Percentile']
    message = f"A/A+ Percentage: {a_grades_percent:.2f}%."
    if flag:
        message += f" (Warning: Below minimum threshold of {params['Min_A_Percentile']}%)"
    return flag, message

def calculate_stats_and_plots(df, score_col, grade_col, section_col, gpa_map):
    """Calculates statistics, runs ANOVA, and prepares data for plots."""
    # --- CRITICAL FIX for the Error ---
    # Map grades to GPA points
    df['GPA'] = df[grade_col].map(gpa_map)
    # Ensure the GPA column is numeric, coercing errors to NaN
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
    # --- End Fix ---

    # Check if GPA mapping resulted in all NaNs (e.g., bad GPA map or grade values)
    if df['GPA'].isnull().all():
        st.error("GPA calculation resulted in all non-numeric values. Check GPA Scale and Assigned Grades.")
        # Return empty or error results
        return { "error": "GPA calculation failed." }


    # Overall stats (handle potential NaNs in GPA)
    overall_dist = df[grade_col].value_counts(normalize=True).sort_index()
    overall_gpa = df['GPA'].mean() # mean() ignores NaNs by default

    # Per-section stats
    section_stats = df.groupby(section_col).agg(
        Avg_GPA=('GPA', 'mean'), # mean ignores NaNs
        Count=('GPA', 'size'), # Use size to count all rows per group
        Valid_GPA_Count = ('GPA', 'count') # count ignores NaNs
    )
    # Calculate section distributions robustly
    section_dist = df.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)

    # ANOVA (only run if there are multiple sections with valid numeric GPA data)
    section_groups = [group['GPA'].dropna() for name, group in df.groupby(section_col) if group['GPA'].notna().sum() > 1] # Need >1 datapoint per group
    anova_result = "ANOVA not applicable (e.g., single section or insufficient data)."
    anova_p_value = None
    if len(section_groups) > 1:
      try:
          # Optional: Add variance check (Levene's test) before ANOVA if desired
          f_val, p_val = stats.f_oneway(*section_groups)
          anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"
          anova_p_value = p_val
      except ValueError as e:
          anova_result = f"ANOVA could not be run. Error: {e}"

    # Prepare results dictionary (Plots will be generated in the main app body for simplicity here)
    results = {
        "overall_dist": overall_dist,
        "overall_gpa": overall_gpa,
        "section_stats": section_stats,
        "section_dist": section_dist,
        "anova_result": anova_result,
        "anova_p_value": anova_p_value,
        "error": None # Indicate success
    }
    return results

# ============================================
# Streamlit App Layout
# ============================================

st.set_page_config(layout="wide")
st.title("Systematic Grading Tool")
st.markdown("""
This tool implements a systematic grading approach based primarily on **fixed score widths** for letter grades,
while incorporating **rule-based adjustments** for score gaps or clusters near boundaries.
It aims for consistency and transparency in the grading process.

**Methodology:**
1.  **Absolute Floors:** Grades 'F' and 'D' are based on fixed score thresholds.
2.  **Fixed Widths:** Grades 'C', 'B', and 'A' ranges are initially set to have a standard point width defined by you.
3.  **Boundary Adjustments:** The tool checks near the initial C/B/A boundaries:
    * If a significant *gap* in scores is found, the boundary may be shifted slightly into the gap.
    * If a *dense cluster* of students sits exactly on a boundary, it may be shifted slightly to keep the cluster together.
    * These shifts are limited by the 'Max Boundary Shift' parameter.
4.  **Sanity Check:** It verifies if the percentage of students receiving 'A' or 'A+' meets a minimum threshold you set.
5.  **Reporting:** Provides grade distributions, GPA statistics per section, ANOVA test for section fairness, and the final grade list.
""")

st.info("**Instructions:**\n"
        "1. Set the desired grading parameters in the sidebar (hover over labels for details).\n"
        "2. Upload your score file (CSV or Excel).\n"
        "3. Select the columns containing Scores and Section identifiers.\n"
        "4. Review the data preview and initial distribution.\n"
        "5. Click 'Calculate Grades'.\n"
        "6. Review the results, including cutoffs, adjustments, sanity checks, statistics, and assigned grades.\n"
        "7. Download the final grades using the button at the bottom.")


# --- Sidebar for Configuration ---
st.sidebar.header("1. Grading Parameters")
params = {}
# Use number_input for thresholds and widths
params['F_Threshold'] = st.sidebar.number_input(
    "F Threshold (< score)", value=60.0, step=0.5,
    help="Scores BELOW this value are assigned 'F'. Defines the start of the 'D' range.")
params['C_Start_Threshold'] = st.sidebar.number_input(
    "C Start Score (>= score)", value=65.0, step=0.5,
    help="Scores AT or ABOVE this value (and below C_End) start the 'C' range. Defines the end of the 'D' range.")
params['C_Width'] = st.sidebar.number_input(
    "C Grade Width", value=7.0, step=0.5, min_value=1.0,
    help="The score point range for the 'C' grade category (e.g., if C starts at 65 and width is 7, C ends below 72).")
params['B_Width'] = st.sidebar.number_input(
    "B Grade Width", value=7.0, step=0.5, min_value=1.0,
    help="The score point range for the 'B' grade category.")
params['A_Width'] = st.sidebar.number_input(
    "A Grade Width", value=7.0, step=0.5, min_value=1.0,
    help="The score point range for the 'A' grade category. Scores above A's range are A+.")
# Use number_input for adjustment rules
st.sidebar.markdown("---")
st.sidebar.markdown("**Boundary Adjustment Rules:**")
params['Max_Boundary_Shift'] = st.sidebar.number_input(
    "Max Boundary Shift (+/- points)", value=1.0, step=0.5, min_value=0.0,
    help="Maximum points a boundary can be shifted due to gaps or clusters.")
params['Significant_Gap_Size'] = st.sidebar.number_input(
    "Significant Gap Size (points)", value=2.0, step=0.5, min_value=0.1,
    help="A score gap must be at least this wide to trigger a potential boundary shift.")
params['Dense_Cluster_Threshold'] = st.sidebar.number_input(
    "Dense Cluster Threshold (% of students)", value=1.0, step=0.5, min_value=0.0,
    help="Percentage of students at a specific score needed to be considered a 'dense cluster' for adjustment.")
params['Cluster_Shift_Policy'] = st.sidebar.selectbox(
    "Cluster Shift Policy", ['Shift Down (Include in Higher Grade)', 'Shift Up (Exclude)'],
    help="If a boundary hits a dense cluster, shift slightly down (giving them the higher grade) or up (giving them the lower grade).")
# Use number_input for sanity check
st.sidebar.markdown("---")
st.sidebar.markdown("**Sanity Check:**")
params['Min_A_Percentile'] = st.sidebar.number_input(
    "Min Acceptable A/A+ Percentile (%)", value=10.0, step=1.0, min_value=0.0, max_value=100.0,
    help="Flags a warning if the final percentage of A/A+ grades is below this value.")

# Standard GPA Scale (Adjust if your scale or +/- grades differ)
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0}
# Make sure the 'labels' in assign_letter_grades match the keys in GPA_SCALE if not using +/-


# --- Main Area Workflow ---
st.header("2. Upload & Prepare Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"])

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'grading_results' not in st.session_state:
    st.session_state.grading_results = None
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False

col_id, col_score, col_section = None, None, None # Define variables before the if block

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        st.subheader("Map Columns")
        cols = df_upload.columns.tolist()
        cols_with_none = ["<Select Column>"] + cols # Add default empty option

        col_id = st.selectbox("Select Student ID Column (Optional)", options=cols_with_none, index=0)
        col_score = st.selectbox("Select Score Column", options=cols_with_none, index=0)
        col_section = st.selectbox("Select Section Column", options=cols_with_none, index=0)

        # Proceed only if essential columns are selected
        if col_score != "<Select Column>" and col_section != "<Select Column>":
            # Make a copy to avoid modifying original upload if user re-selects cols
            st.session_state.df = df_upload.copy()
            # Rename columns for internal consistency
            st.session_state.df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
            if col_id != "<Select Column>":
                 st.session_state.df.rename(columns={col_id: 'StudentID'}, inplace=True)
            else:
                 st.session_state.df['StudentID'] = 'Stud_' + st.session_state.df.index.astype(str) # Create default ID


            # --- Data Cleaning & Validation ---
            st.session_state.df['Score'] = pd.to_numeric(st.session_state.df['Score'], errors='coerce')
            initial_rows = len(st.session_state.df)
            st.session_state.df.dropna(subset=['Score'], inplace=True) # Remove rows with non-numeric scores
            removed_rows = initial_rows - len(st.session_state.df)
            if removed_rows > 0:
                st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")

            if st.session_state.df.empty:
                 st.error("No valid score data remaining after cleaning.")
                 st.session_state.df = None # Reset
            else:
                st.subheader("Data Preview & Initial Stats")
                st.dataframe(st.session_state.df[['StudentID', 'Score', 'Section']].head())
                st.write(st.session_state.df['Score'].describe())

                # Initial Histogram
                st.subheader("Initial Score Distribution")
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(st.session_state.df['Score'], kde=True, ax=ax_hist)
                ax_hist.set_title("Score Distribution")
                st.pyplot(fig_hist)

                # --- Reset results if data/params change before calculation ---
                st.session_state.calculation_done = False
                st.session_state.grading_results = None

                # --- Grading Execution Button ---
                st.header("3. Run Grading Process")
                if st.button("Calculate Grades"):
                    st.session_state.calculation_done = True # Mark calculation as done
                    with st.spinner("Calculating grades..."):
                        # Make a fresh copy for calculations if needed
                        df_calc = st.session_state.df.copy()

                        # 1. Calculate Initial Cutoffs
                        initial_cutoffs = calculate_initial_cutoffs(params)

                        # 2. Adjust Boundaries
                        final_cutoffs, adjustment_log = adjust_boundaries(df_calc['Score'], initial_cutoffs, params)

                        # 3. Assign Letter Grades
                        df_calc['Letter_Grade'] = assign_letter_grades(df_calc['Score'], final_cutoffs)

                        # 4. Sanity Checks
                        sanity_flag, sanity_message = perform_sanity_checks(df_calc, 'Letter_Grade', final_cutoffs, params)

                        # 5. Calculate Stats and Plots Data
                        stats_results = calculate_stats_and_plots(df_calc, 'Score', 'Letter_Grade', 'Section', GPA_SCALE)

                        # Store results in session state if no error during stats calc
                        if stats_results.get("error") is None:
                           st.session_state.grading_results = {
                               "final_cutoffs": final_cutoffs,
                               "adjustment_log": adjustment_log,
                               "sanity_flag": sanity_flag,
                               "sanity_message": sanity_message,
                               "stats_results": stats_results,
                               "graded_df": df_calc # Store the df with grades
                           }
                           st.success("Grading process complete!")
                        else:
                            st.error(f"Failed during statistics calculation: {stats_results.get('error')}")
                            st.session_state.grading_results = None # Clear results on error
                            st.session_state.calculation_done = False


        # Display column selection warning if not ready
        elif uploaded_file: # Check if file uploaded but columns not selected
            st.warning("Please select the Score and Section columns.")

    except Exception as e:
        st.error(f"An error occurred during file processing or column mapping: {e}")
        st.session_state.df = None # Reset dataframe on error
        st.session_state.grading_results = None
        st.session_state.calculation_done = False


# --- Display Results (only if calculation was done and results exist) ---
if st.session_state.calculation_done and st.session_state.grading_results:
    results = st.session_state.grading_results
    st.header("4. Grading Results")

    # Display Key Info
    st.subheader("Final Cutoffs Used")
    # Display cutoffs more readably
    cutoff_df = pd.DataFrame(list(results['final_cutoffs'].items()), columns=['Boundary', 'Score Threshold'])
    st.dataframe(cutoff_df)


    st.subheader("Adjustments & Checks")
    col_adj, col_chk = st.columns(2)
    with col_adj:
        if results['adjustment_log']:
            with st.expander("Boundary Adjustment Log"):
                for log_entry in results['adjustment_log']:
                    st.write(log_entry)
        else:
            st.write("No boundary adjustments were made.")
    with col_chk:
        st.write("**Sanity Check:**")
        if results['sanity_flag']:
            st.warning(results['sanity_message'])
        else:
            st.success(results['sanity_message'])


    # Display Statistics and Plots
    st.subheader("Grade Distributions & Statistics")
    col1, col2 = st.columns(2)
    stats_res = results['stats_results'] # Shortcut

    with col1:
        st.write("**Overall Distribution:**")
        st.dataframe(stats_res['overall_dist'].apply("{:.1%}".format)) # Format as percentage
        st.write(f"**Overall Avg GPA:** {stats_res['overall_gpa']:.2f}")

        # Overall Distribution Plot
        fig_dist, ax_dist = plt.subplots()
        stats_res['overall_dist'].plot(kind='bar', ax=ax_dist)
        ax_dist.set_ylabel("Proportion")
        ax_dist.set_title("Overall Grade Distribution")
        st.pyplot(fig_dist)

    with col2:
        st.write("**Per Section Avg GPA:**")
        st.dataframe(stats_res['section_stats'][['Avg_GPA']].style.format("{:.2f}")) # Show only Avg_GPA formatted
        st.write(f"**ANOVA Result:** {stats_res['anova_result']}")
        if stats_res['anova_p_value'] is not None and stats_res['anova_p_value'] < 0.05:
            st.warning("Statistically significant difference found between section GPAs (p < 0.05). Review Section Stats.")

        # Section GPA Boxplot
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=results['graded_df'], x='Section', y='GPA', ax=ax_box)
        ax_box.set_title("GPA Distribution by Section")
        st.pyplot(fig_box)

    # Display Section Distributions Table (optional)
    with st.expander("Per Section Grade Distributions (%)"):
         st.dataframe(stats_res['section_dist'].style.format("{:.1%}"))

    # Display Final Grades Table
    st.subheader("Assigned Grades")
    display_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
    # Ensure columns exist before trying to display them
    display_cols = [col for col in display_cols if col in results['graded_df'].columns]
    st.dataframe(results['graded_df'][display_cols])

    # Download Button
    @st.cache_data # Cache the conversion to avoid re-running
    def convert_df_to_csv(df):
        # Select only relevant columns for download
        dl_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
        dl_cols = [col for col in dl_cols if col in df.columns]
        return df[dl_cols].to_csv(index=False).encode('utf-8')

    try:
        csv_data = convert_df_to_csv(results['graded_df'])
        st.download_button(
            label="Download Grades as CSV",
            data=csv_data,
            file_name='graded_student_scores.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Could not prepare download file: {e}")


elif uploaded_file and (col_score == "<Select Column>" or col_section == "<Select Column>"):
     st.info("Select Score and Section columns using the dropdowns above to proceed.")

# Add footer or contact info if desired
st.sidebar.markdown("---")
st.sidebar.info("Grading Tool v1.1")
