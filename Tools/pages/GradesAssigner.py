import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For ANOVA
from io import BytesIO # For download button
import collections # To use OrderedDict

# ============================================
# Define Core Grading Functions (with corrections)
# ============================================

def calculate_initial_cutoffs(params):
    """Calculates initial grade boundaries based on fixed widths."""
    cutoffs = collections.OrderedDict() # Use OrderedDict to maintain order
    cutoffs['F_Upper'] = params['F_Threshold']       # Scores < this are F
    # Use the corrected parameter key here:
    cutoffs['D_Upper'] = params['D_C_Boundary']      # Scores < this (and >= F_Upper) are D
    cutoffs['C_Upper'] = cutoffs['D_Upper'] + params['C_Width']
    cutoffs['B_Upper'] = cutoffs['C_Upper'] + params['B_Width']
    cutoffs['A_Upper'] = cutoffs['B_Upper'] + params['A_Width'] # Upper bound for A / Start for A+
    cutoffs['A+_Upper'] = 100.1 # Use a value slightly above max possible score (adjust if max score differs)
    return cutoffs

def adjust_boundaries(scores, initial_cutoffs, params):
    """Applies rule-based adjustments to boundaries based on gaps/clusters."""
    adjusted_cutoffs = initial_cutoffs.copy()
    adjustment_log = []
    # Adjust boundaries that define the *start* of C, B, A, A+ (which are the upper bounds of D, C, B, A)
    # Order matters if adjustments affect subsequent boundaries
    boundaries_to_check = ['D_Upper', 'C_Upper', 'B_Upper', 'A_Upper']

    for boundary_key in boundaries_to_check:
        # Get the value defining the upper bound of the current grade / start of the next
        cutoff_val = adjusted_cutoffs[boundary_key]
        adjusted = False # Flag to prevent applying both gap and cluster logic

        # --- Gap Check Logic ---
        # Define search range around cutoff based on max_shift
        min_bound_search = cutoff_val - params['Max_Boundary_Shift']
        max_bound_search = cutoff_val + params['Max_Boundary_Shift']
        scores_near_cutoff = scores[(scores >= min_bound_search) & (scores <= max_bound_search)]

        if not scores_near_cutoff.empty:
            sorted_unique_near = np.unique(scores_near_cutoff)
            if len(sorted_unique_near) > 1: # Need at least two points to have a gap
                diffs = np.diff(sorted_unique_near)
                # Find if any gap within the check range contains the cutoff_val
                for i, diff in enumerate(diffs):
                    if diff >= params['Significant_Gap_Size']:
                        gap_start = sorted_unique_near[i]
                        gap_end = sorted_unique_near[i+1]
                        # Check if the original cutoff falls strictly within this specific gap
                        if cutoff_val > gap_start and cutoff_val < gap_end:
                            # Shift to middle of the gap
                            new_cutoff = (gap_start + gap_end) / 2.0
                            # Ensure shift doesn't exceed max allowed magnitude from original cutoff
                            actual_shift = new_cutoff - initial_cutoffs[boundary_key] # Check against original
                            if abs(actual_shift) > params['Max_Boundary_Shift']:
                               # Limit shift (e.g., shift by max_shift towards gap middle from original)
                               new_cutoff = initial_cutoffs[boundary_key] + np.sign(actual_shift) * params['Max_Boundary_Shift']

                            adjusted_cutoffs[boundary_key] = new_cutoff
                            adjustment_log.append(f"Shifted {boundary_key} from {initial_cutoffs[boundary_key]:.2f} to {new_cutoff:.2f} due to gap ({gap_start:.1f}-{gap_end:.1f}).")
                            adjusted = True
                            break # Apply only first found gap adjustment for this boundary
        if adjusted:
            # If a boundary is adjusted, we might need to recalculate subsequent boundaries
            # if they depend on the adjusted value (e.g., C_Upper depends on D_Upper + C_Width).
            # Re-calculate subsequent boundaries based on the adjusted one:
            if boundary_key == 'D_Upper':
                 adjusted_cutoffs['C_Upper'] = adjusted_cutoffs['D_Upper'] + params['C_Width']
                 adjusted_cutoffs['B_Upper'] = adjusted_cutoffs['C_Upper'] + params['B_Width']
                 adjusted_cutoffs['A_Upper'] = adjusted_cutoffs['B_Upper'] + params['A_Width']
            elif boundary_key == 'C_Upper':
                 adjusted_cutoffs['B_Upper'] = adjusted_cutoffs['C_Upper'] + params['B_Width']
                 adjusted_cutoffs['A_Upper'] = adjusted_cutoffs['B_Upper'] + params['A_Width']
            elif boundary_key == 'B_Upper':
                 adjusted_cutoffs['A_Upper'] = adjusted_cutoffs['B_Upper'] + params['A_Width']
            continue # Move to next boundary

        # --- Cluster Check Logic ---
        # Check cluster only if no gap adjustment was made
        cluster_count = np.sum(np.isclose(scores, cutoff_val)) # Use isclose for potential float issues
        if len(scores)>0:
             cluster_percent = (cluster_count / len(scores)) * 100
        else:
             cluster_percent = 0

        if cluster_percent >= params['Dense_Cluster_Threshold']:
            # Shift slightly based on policy to include/exclude cluster
            # Shift Down includes cluster in the higher grade category by lowering upper bound slightly
            shift_amount = -0.01 if params['Cluster_Shift_Policy'] == 'Shift Down (Include in Higher Grade)' else 0.01
            # Ensure shift amount itself is within allowed boundary shift (usually true for 0.01)
            if abs(shift_amount) <= params['Max_Boundary_Shift']:
                 new_cutoff = cutoff_val + shift_amount
                 adjusted_cutoffs[boundary_key] = new_cutoff
                 adjustment_log.append(f"Shifted {boundary_key} from {cutoff_val:.2f} to {new_cutoff:.2f} due to cluster ({cluster_count} students).")
                 # Recalculate subsequent boundaries similar to gap adjustment
                 if boundary_key == 'D_Upper':
                      adjusted_cutoffs['C_Upper'] = adjusted_cutoffs['D_Upper'] + params['C_Width']
                      adjusted_cutoffs['B_Upper'] = adjusted_cutoffs['C_Upper'] + params['B_Width']
                      adjusted_cutoffs['A_Upper'] = adjusted_cutoffs['B_Upper'] + params['A_Width']
                 elif boundary_key == 'C_Upper':
                      adjusted_cutoffs['B_Upper'] = adjusted_cutoffs['C_Upper'] + params['B_Width']
                      adjusted_cutoffs['A_Upper'] = adjusted_cutoffs['B_Upper'] + params['A_Width']
                 elif boundary_key == 'B_Upper':
                      adjusted_cutoffs['A_Upper'] = adjusted_cutoffs['B_Upper'] + params['A_Width']

            else:
                 adjustment_log.append(f"INFO: Cluster found at {boundary_key}={cutoff_val:.2f} but required shift > Max_Boundary_Shift.")

    return adjusted_cutoffs, adjustment_log


def assign_letter_grades(scores, final_cutoffs):
    """Assigns letter grades based on final cutoff boundaries."""
    # Define bins using the upper boundary values from the ordered dict
    # Ensure bins start low enough and end high enough
    bins = [-np.inf] + list(final_cutoffs.values())
    # Define labels corresponding to the intervals created by the bins
    # Ensure number of labels = number of bins - 1
    labels = ['F', 'D', 'C', 'B', 'A', 'A+'] # Matches 7 bins -> 6 intervals

    # Use pd.cut. Ensure scores are numeric.
    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels, right=False, ordered=True) # right=False means [lower, upper)
    return grades

def perform_sanity_checks(df, grade_col, final_cutoffs, params):
    """Performs sanity checks like minimum A/A+ percentage."""
    if grade_col not in df.columns or df[grade_col].isnull().all():
         return True, "Warning: Could not perform sanity check due to missing grade assignments."

    a_grades_percent = (df[grade_col].astype(str).isin(['A', 'A+'])).mean() * 100 # Use astype(str) for robustness
    flag = a_grades_percent < params['Min_A_Percentile']
    message = f"A/A+ Percentage: {a_grades_percent:.2f}%."
    if flag:
        message += f" (Warning: Below minimum threshold of {params['Min_A_Percentile']}%)"
    return flag, message

def calculate_stats_and_plots(df, score_col, grade_col, section_col, gpa_map):
    """Calculates statistics, runs ANOVA, and prepares data for plots."""
    # Ensure necessary columns exist
    if grade_col not in df.columns:
        st.error(f"Grade column '{grade_col}' not found in DataFrame.")
        return { "error": f"Grade column '{grade_col}' missing." }
    if section_col not in df.columns:
        st.error(f"Section column '{section_col}' not found in DataFrame.")
        return { "error": f"Section column '{section_col}' missing." }

    # --- CRITICAL FIX for the Error ---
    # Map grades to GPA points
    df['GPA'] = df[grade_col].astype(str).map(gpa_map) # Use astype(str) for robustness before mapping
    # Ensure the GPA column is numeric, coercing errors to NaN
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
    # --- End Fix ---

    # Check if GPA mapping resulted in all NaNs (e.g., bad GPA map or grade values)
    if df['GPA'].isnull().all():
        st.error("GPA calculation resulted in all non-numeric values. Check GPA Scale and Assigned Grades.")
        # Return empty or error results
        return { "error": "GPA calculation failed (all NaNs)." }

    # Overall stats (handle potential NaNs in GPA)
    overall_dist = df[grade_col].value_counts(normalize=True).sort_index()
    overall_gpa = df['GPA'].mean() # mean() ignores NaNs by default

    # Per-section stats
    # Ensure section column is suitable for grouping
    df[section_col] = df[section_col].astype(str) # Convert section to string to be safe

    section_stats = df.groupby(section_col).agg(
        Avg_GPA=('GPA', 'mean'), # mean ignores NaNs
        Count=('GPA', 'size'), # Use size to count all rows per group
        Valid_GPA_Count = ('GPA', 'count') # count ignores NaNs
    ).reset_index() # Reset index to make section a column again

    # Calculate section distributions robustly
    try:
        section_dist = df.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)
    except Exception as e:
        st.warning(f"Could not calculate section distribution table: {e}")
        section_dist = pd.DataFrame() # Return empty dataframe


    # ANOVA (only run if there are multiple sections with valid numeric GPA data)
    # Ensure groups are Series for f_oneway
    section_groups = [group['GPA'].dropna().values for name, group in df.groupby(section_col) if group['GPA'].notna().sum() > 1] # Need >1 datapoint per group as numpy arrays
    anova_result = "ANOVA not applicable (e.g., single section or insufficient data)."
    anova_p_value = None
    if len(section_groups) > 1:
      # Check if all groups are non-empty after dropna
      valid_groups = [g for g in section_groups if len(g) > 0]
      if len(valid_groups) > 1:
          try:
              # Optional: Add variance check (Levene's test) before ANOVA if desired
              f_val, p_val = stats.f_oneway(*valid_groups)
              anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"
              anova_p_value = p_val
          except ValueError as e:
              anova_result = f"ANOVA could not be run. Error: {e}"
      else:
           anova_result = "ANOVA not applicable (fewer than 2 sections with valid data)."


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
2.  **Fixed Widths:** Grades 'C', 'B', and 'A' ranges are initially set to have a standard point width defined by you. 'A+' covers scores above the 'A' range.
3.  **Boundary Adjustments:** The tool checks near the initial D/C, C/B, B/A, and A/A+ boundaries:
    * If a significant *gap* in scores is found, the boundary may be shifted slightly into the gap.
    * If a *dense cluster* of students sits exactly on a boundary, it may be shifted slightly to keep the cluster together.
    * These shifts are limited by the 'Max Boundary Shift' parameter. Adjusting one boundary recalculates subsequent ones based on fixed widths.
4.  **Sanity Check:** It verifies if the percentage of students receiving 'A' or 'A+' meets a minimum threshold you set.
5.  **Reporting:** Provides grade distributions, GPA statistics per section, ANOVA test for section fairness, and the final grade list.
""")

st.info("**Instructions:**\n"
        "1. Set the desired grading parameters in the sidebar (hover over labels for details).\n"
        "2. Upload your score file (CSV or Excel).\n"
        "3. Select the columns containing Scores and Section identifiers (Student ID is optional).\n"
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
    help="Scores BELOW this value are assigned 'F'. This value also marks the absolute bottom of the 'D' range.")

# <<< CORRECTED PARAMETER LABEL AND KEY >>>
params['D_C_Boundary'] = st.sidebar.number_input(
    "D / C Boundary Score", value=65.0, step=0.5,
    help="Scores AT or ABOVE this value start the 'C' range. Scores BELOW this value (and >= F Threshold) are 'D'. This defines the upper limit for 'D'.")
# <<< END CORRECTION >>>

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
    help="Maximum points a boundary can be shifted from its initial position due to gaps or clusters.")
params['Significant_Gap_Size'] = st.sidebar.number_input(
    "Significant Gap Size (points)", value=2.0, step=0.5, min_value=0.1,
    help="A score gap must be at least this wide to trigger a potential boundary shift.")
params['Dense_Cluster_Threshold'] = st.sidebar.number_input(
    "Dense Cluster Threshold (% of students)", value=1.0, step=0.5, min_value=0.0,
    help="Percentage of students at a specific score needed to be considered a 'dense cluster' for adjustment.")
params['Cluster_Shift_Policy'] = st.sidebar.selectbox(
    "Cluster Shift Policy", ['Shift Down (Include in Higher Grade)', 'Shift Up (Exclude)'],
    help="If a boundary hits a dense cluster, shift slightly down (giving them the higher grade) or up (giving them the lower grade). 'Down' means lowering the upper boundary value.")

# Use number_input for sanity check
st.sidebar.markdown("---")
st.sidebar.markdown("**Sanity Check:**")
params['Min_A_Percentile'] = st.sidebar.number_input(
    "Min Acceptable A/A+ Percentile (%)", value=10.0, step=1.0, min_value=0.0, max_value=100.0,
    help="Flags a warning if the final percentage of A/A+ grades is below this value.")

# Standard GPA Scale (Adjust if your scale or +/- grades differ)
# Make sure the 'labels' in assign_letter_grades match the keys in GPA_SCALE
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0}
# The current assign_letter_grades uses ['F', 'D', 'C', 'B', 'A', 'A+'] - update GPA_SCALE if you add B+, C+ etc.


# --- Main Area Workflow ---
st.header("2. Upload & Prepare Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"])

# Initialize session state variables robustly
if 'df' not in st.session_state: st.session_state.df = None
if 'grading_results' not in st.session_state: st.session_state.grading_results = None
if 'calculation_done' not in st.session_state: st.session_state.calculation_done = False
if 'columns_selected' not in st.session_state: st.session_state.columns_selected = False

col_id, col_score, col_section = None, None, None # Define variables

if uploaded_file:
    try:
        # Read file into a temporary dataframe first
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")

        st.subheader("Map Columns")
        cols = df_upload.columns.tolist()
        cols_with_none = ["<Select Column>"] + cols

        # Use unique keys for select boxes to prevent state issues if columns have same name
        col_id = st.selectbox("Select Student ID Column (Optional)", options=cols_with_none, index=0, key='sel_id')
        col_score = st.selectbox("Select Score Column", options=cols_with_none, index=0, key='sel_score')
        col_section = st.selectbox("Select Section Column", options=cols_with_none, index=0, key='sel_section')

        # Set flag if essential columns are selected
        st.session_state.columns_selected = (col_score != "<Select Column>" and col_section != "<Select Column>")

        if st.session_state.columns_selected:
            # If columns newly selected or file changed, process the data
            # Use a simple check or compare checksums if needed for more robustness
            # Store the processed dataframe in session state
            st.session_state.df = df_upload.copy()

            # --- Data Cleaning & Validation ---
            try:
                st.session_state.df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
                if col_id != "<Select Column>":
                     st.session_state.df.rename(columns={col_id: 'StudentID'}, inplace=True)
                else:
                     st.session_state.df['StudentID'] = 'Stud_' + st.session_state.df.index.astype(str) # Create default ID

                st.session_state.df['Score'] = pd.to_numeric(st.session_state.df['Score'], errors='coerce')
                initial_rows = len(st.session_state.df)
                st.session_state.df.dropna(subset=['Score'], inplace=True) # Remove rows with non-numeric scores
                removed_rows = initial_rows - len(st.session_state.df)
                if removed_rows > 0:
                    st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")

                if st.session_state.df.empty:
                     st.error("No valid score data remaining after cleaning.")
                     st.session_state.df = None # Reset
                     st.session_state.columns_selected = False # Reset flag
                else:
                    # Ensure section is treated as categorical/string
                    st.session_state.df['Section'] = st.session_state.df['Section'].astype(str)

                    # Display Preview only after successful cleaning
                    st.subheader("Data Preview & Initial Stats")
                    st.dataframe(st.session_state.df[['StudentID', 'Score', 'Section']].head())
                    st.write(st.session_state.df['Score'].describe())

                    # Initial Histogram
                    st.subheader("Initial Score Distribution")
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(st.session_state.df['Score'], kde=True, ax=ax_hist)
                    ax_hist.set_title("Score Distribution")
                    st.pyplot(fig_hist)
                    plt.close(fig_hist) # Close plot to free memory

                    # --- Reset results display if data/params change ---
                    st.session_state.calculation_done = False
                    st.session_state.grading_results = None

            except Exception as e:
                 st.error(f"Error processing columns or cleaning data: {e}")
                 st.session_state.df = None
                 st.session_state.columns_selected = False

        # Warning if file uploaded but columns not selected
        elif uploaded_file:
            st.warning("Please select the Score and Section columns.")

    except Exception as e:
        st.error(f"An error occurred during file reading: {e}")
        # Reset state variables on file reading error
        st.session_state.df = None
        st.session_state.grading_results = None
        st.session_state.calculation_done = False
        st.session_state.columns_selected = False


# --- Grading Execution Button ---
# Show button only if data is loaded and columns are selected
if st.session_state.columns_selected and st.session_state.df is not None:
    st.header("3. Run Grading Process")
    if st.button("Calculate Grades"):
        st.session_state.calculation_done = True # Mark calculation attempt
        with st.spinner("Calculating grades..."):
            # Use the dataframe stored in session state
            df_calc = st.session_state.df.copy()

            try:
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

                # Store results in session state only if calculation successful
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
                    # Error occurred during stats calculation
                    st.error(f"Failed during statistics calculation: {stats_results.get('error')}")
                    st.session_state.grading_results = None # Clear results on error
                    st.session_state.calculation_done = False # Reset flag as calculation failed

            except Exception as e:
                 st.error(f"An error occurred during the grading calculation: {e}")
                 st.session_state.grading_results = None
                 st.session_state.calculation_done = False


# --- Display Results ---
# Only display results if calculation was attempted, completed, and results are valid
if st.session_state.calculation_done and st.session_state.grading_results:
    results = st.session_state.grading_results
    st.header("4. Grading Results")

    # Display Key Info
    st.subheader("Final Cutoffs Used")
    try:
        # Display cutoffs more readably
        cutoff_df = pd.DataFrame(list(results['final_cutoffs'].items()), columns=['Boundary', 'Score Threshold'])
        st.dataframe(cutoff_df.style.format({"Score Threshold": "{:.2f}"}))
    except Exception as e:
        st.warning(f"Could not display cutoffs table: {e}")
        st.json(results['final_cutoffs']) # Fallback to JSON


    st.subheader("Adjustments & Checks")
    col_adj, col_chk = st.columns(2)
    with col_adj:
        if results['adjustment_log']:
            with st.expander("Boundary Adjustment Log"):
                for log_entry in results['adjustment_log']:
                    st.write(f"- {log_entry}") # Add bullet point
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
        if "overall_dist" in stats_res and not stats_res['overall_dist'].empty:
            st.dataframe(stats_res['overall_dist'].apply("{:.1%}".format)) # Format as percentage
            st.write(f"**Overall Avg GPA:** {stats_res.get('overall_gpa', 'N/A'):.2f}")

            # Overall Distribution Plot
            try:
                fig_dist, ax_dist = plt.subplots()
                # Ensure index is sorted for consistent plotting if needed
                stats_res['overall_dist'].sort_index().plot(kind='bar', ax=ax_dist)
                ax_dist.set_ylabel("Proportion")
                ax_dist.set_title("Overall Grade Distribution")
                plt.xticks(rotation=45)
                st.pyplot(fig_dist)
                plt.close(fig_dist) # Close plot
            except Exception as e:
                st.warning(f"Could not generate overall distribution plot: {e}")
        else:
             st.write("Could not calculate overall distribution.")

    with col2:
        st.write("**Per Section Avg GPA:**")
        if "section_stats" in stats_res and not stats_res['section_stats'].empty:
             # Ensure correct columns exist before formatting
             cols_to_format = ['Avg_GPA']
             valid_cols = [col for col in cols_to_format if col in stats_res['section_stats'].columns]
             st.dataframe(stats_res['section_stats'].style.format({col: "{:.2f}" for col in valid_cols}))
        else:
             st.write("Could not calculate section statistics.")

        st.write(f"**ANOVA Result:** {stats_res.get('anova_result', 'N/A')}")
        anova_p = stats_res.get('anova_p_value')
        if anova_p is not None and anova_p < 0.05:
            st.warning("Statistically significant difference found between section GPAs (p < 0.05). Review Section Stats.")

        # Section GPA Boxplot
        try:
            if 'graded_df' in results and not results['graded_df'].empty:
                 fig_box, ax_box = plt.subplots()
                 sns.boxplot(data=results['graded_df'], x='Section', y='GPA', ax=ax_box)
                 ax_box.set_title("GPA Distribution by Section")
                 plt.xticks(rotation=45, ha='right')
                 st.pyplot(fig_box)
                 plt.close(fig_box) # Close plot
            else:
                 st.warning("Graded data not available for boxplot.")
        except Exception as e:
            st.warning(f"Could not generate section GPA boxplot: {e}")


    # Display Section Distributions Table (optional)
    if "section_dist" in stats_res and not stats_res['section_dist'].empty:
        with st.expander("Per Section Grade Distributions (%)"):
             st.dataframe(stats_res['section_dist'].style.format("{:.1%}"))
    else:
         st.write("Section distribution table could not be generated.")


    # Display Final Grades Table
    st.subheader("Assigned Grades")
    if 'graded_df' in results and not results['graded_df'].empty:
        display_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
        # Ensure columns exist before trying to display them
        display_cols = [col for col in display_cols if col in results['graded_df'].columns]
        st.dataframe(results['graded_df'][display_cols])

        # Download Button
        @st.cache_data # Cache the conversion to avoid re-running if df doesn't change
        def convert_df_to_csv(df_to_convert):
            # Select only relevant columns for download
            dl_cols = ['StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
            dl_cols = [col for col in dl_cols if col in df_to_convert.columns]
            if not dl_cols: return None # Cannot convert if no columns
            return df_to_convert[dl_cols].to_csv(index=False).encode('utf-8')

        try:
            csv_data = convert_df_to_csv(results['graded_df'])
            if csv_data:
                st.download_button(
                    label="Download Grades as CSV",
                    data=csv_data,
                    file_name='graded_student_scores.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Could not prepare download file: {e}")
    else:
         st.warning("Final grades table is unavailable.")


# Add footer or contact info if desired
st.sidebar.markdown("---")
st.sidebar.info("Grading Tool v1.2") # Incremented version
