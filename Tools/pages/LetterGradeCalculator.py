# Full code block with integrated fixes
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from io import BytesIO
import collections
import itertools

# ============================================
# Helper Functions
# ============================================
def calculate_initial_cutoffs_original(a_plus_start, gap):
    """Calculates initial cutoffs working downwards from A+ start with a uniform gap."""
    cutoffs = collections.OrderedDict()
    grades_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']
    current_start = a_plus_start
    for i, grade in enumerate(grades_order):
        cutoffs[f'{grade}_Start'] = current_start
        current_start -= gap if i < len(grades_order) - 1 else 0
    cutoffs['F_Max'] = cutoffs['D_Start'] # F is anything below D_Start
    return cutoffs

# FIX 1: Robust assign_letter_grades_from_starts function
def assign_letter_grades_from_starts(scores, start_cutoffs):
    """
    Assigns letter grades based on start scores (lower bound inclusive).
    Handles duplicate start scores by prioritizing the higher grade.
    """
    # 1. Monotonicity Check for raw input cutoffs (A+ score >= A score etc.)
    # start_cutoffs is an OrderedDict: {'A+_Start': 95, 'A_Start': 90 ...}
    expected_grade_order_keys = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    
    present_grade_keys = [k for k in expected_grade_order_keys if k in start_cutoffs]
    if len(present_grade_keys) > 1:
        ordered_scores_for_check = [start_cutoffs[k] for k in present_grade_keys]
        if not all(ordered_scores_for_check[i] >= ordered_scores_for_check[i+1] for i in range(len(ordered_scores_for_check)-1)):
            st.error("Input Error: Grade boundary start scores (from A+ down to D) must be monotonically decreasing or equal. Please check manual cutoff inputs.")
            return pd.Series(['Error - Non-monotonic Input Cutoffs'] * len(scores), index=scores.index, dtype='object')

    # 2. Prepare for pd.cut: Create a map from unique scores to the HIGHEST grade achieving that score.
    # `start_cutoffs` is an OrderedDict, processed from highest grade (A+) to lowest (D).
    score_to_highest_grade_label_map = collections.OrderedDict() 
    for grade_key_with_suffix, score_val in start_cutoffs.items():
        if grade_key_with_suffix == 'F_Max':
            continue
        grade_label = grade_key_with_suffix.replace('_Start', '')
        if score_val not in score_to_highest_grade_label_map: 
            score_to_highest_grade_label_map[score_val] = grade_label
    
    if not score_to_highest_grade_label_map:
        st.error("No valid grade boundaries found to use for assignment after processing.")
        return pd.Series(['Error - No Valid Cutoffs For Assignment'] * len(scores), index=scores.index, dtype='object')

    # 3. Get unique boundary scores, sorted ascending for pd.cut bins
    unique_ascending_boundary_scores = sorted(list(score_to_highest_grade_label_map.keys()))

    if not unique_ascending_boundary_scores:
         st.error("Internal Error: No unique boundary scores for binning.")
         return pd.Series(['Error - Binning Issue'] * len(scores), index=scores.index, dtype='object')

    # 4. Create labels for pd.cut: 'F' + grades corresponding to unique_ascending_boundary_scores
    labels_for_cut = ['F'] + [score_to_highest_grade_label_map[s] for s in unique_ascending_boundary_scores]
    
    bins = [-np.inf] + unique_ascending_boundary_scores + [np.inf]

    # 5. Bin/Label Mismatch Check
    if len(labels_for_cut) != len(bins) - 1:
        st.error(f"CRITICAL INTERNAL ERROR: Bin/Label mismatch. Bins ({len(bins)}): {bins}, Labels ({len(labels_for_cut)}): {labels_for_cut}. Please report this error.")
        return pd.Series(['Error - Internal Bin/Label Mismatch'] * len(scores), index=scores.index, dtype='object')

    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels_for_cut, right=False, ordered=False) # Set ordered=False for simplicity
    grades = grades.astype('object').fillna('Invalid Score') 
    return grades


def calculate_stats(df, grade_col, section_col, gpa_map):
    """Calculates distributions and statistics. Corrected GPA handling."""
    if grade_col not in df.columns or df[grade_col].astype(str).str.contains('Error', na=False).any():
        st.warning("Cannot calculate stats due to errors in grade assignment.")
        return {"error": "Grade assignment failed."}
    
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df_copy['GPA'] = df_copy[grade_col].map(gpa_map)
    df_copy['GPA'] = pd.to_numeric(df_copy['GPA'], errors='coerce')

    if df_copy['GPA'].isnull().all() and df_copy[grade_col].notna().any() and not df_copy[grade_col].isin(['Invalid Score']).all():
        st.warning("GPA calculation resulted in all non-numeric values. Check GPA_SCALE and grade assignments.")
    
    overall_dist = df_copy[grade_col].value_counts(normalize=True).sort_index()
    overall_gpa = df_copy['GPA'].mean()
    
    df_copy[section_col] = df_copy[section_col].astype(str)
    
    section_gpa_means = df_copy.groupby(section_col)['GPA'].mean()
    section_stats_agg = df_copy.groupby(section_col).agg(
        Count=('GPA', 'size'), 
        Valid_GPA_Count=('GPA', 'count')
    ).reset_index()
    section_stats = pd.merge(section_stats_agg, section_gpa_means.rename('Avg_GPA'), on=section_col, how='left')
    
    try:
        section_dist = df_copy.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)
    except Exception as e:
        st.warning(f"Could not generate section-wise grade distributions: {e}")
        section_dist = pd.DataFrame() 
        
    anova_result = "ANOVA not applicable."
    anova_p_value = None
    
    # Prepare groups for ANOVA: ensure at least two groups with more than one valid GPA value
    section_groups_for_anova = [
        group['GPA'].dropna().values for _, group in df_copy.groupby(section_col) 
        if group['GPA'].notna().sum() > 1
    ]
    
    if len(section_groups_for_anova) > 1: # Need at least two groups for comparison
        try:
            f_val, p_val = stats.f_oneway(*section_groups_for_anova)
            anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"
            anova_p_value = p_val
        except ValueError as e:
            anova_result = f"ANOVA Error: {e}"
            st.warning(f"ANOVA could not be computed. This might be due to insufficient data in groups. Error: {e}")
        except Exception as e: # Catch any other unexpected errors
            anova_result = f"ANOVA failed unexpectedly: {e}"
            st.warning(f"An unexpected error occurred during ANOVA calculation: {e}")

    return {
        "overall_dist": overall_dist, "overall_gpa": overall_gpa,
        "section_stats": section_stats, "section_dist": section_dist,
        "anova_result": anova_result, "anova_p_value": anova_p_value,
        "error": None
    }

# GPA Scale
GPA_SCALE = {
    'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 
    'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0,
    'Invalid Score': np.nan, 
    'Error - Non-monotonic Cutoffs': np.nan, 
    'Error - Non-monotonic Input Cutoffs': np.nan,
    'Error - Bin/Label Mismatch': np.nan,
    'Error - Internal Bin/Label Mismatch': np.nan,
    'Error - No Valid Cutoffs For Assignment': np.nan,
    'Error - Binning Issue': np.nan
}


# --- Styling Functions ---
GRADE_GRADIENT = { # White to Black/Gray
    'A+': '#FFFFFF', 'A': '#F2F2F2', 'B+': '#E6E6E6', 'B': '#D9D9D9',
    'C+': '#CCCCCC', 'C': '#BDBDBD', 'D+': '#B0B0B0', 'D': '#A3A3A3',
    'F': '#969696', 'default': '#FFFFFF' # default for errors or unexpected grades
}
SECTION_COLOR_PALETTE = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                         '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
section_color_map_generated = {} # Global for consistent coloring within a session view
color_cycle_obj = itertools.cycle(SECTION_COLOR_PALETTE) # Renamed to avoid conflict

def get_section_color_fixed(section_name):
    global color_cycle_obj # Ensure we're using the global cycle object
    str_section = str(section_name)
    if str_section not in section_color_map_generated:
        section_color_map_generated[str_section] = next(color_cycle_obj)
    return section_color_map_generated[str_section]

def reset_section_color_cycle(): # Call this when new data is loaded or plots need fresh palettes
    global section_color_map_generated, color_cycle_obj
    section_color_map_generated.clear()
    color_cycle_obj = itertools.cycle(SECTION_COLOR_PALETTE)


def highlight_upgraded(row, upgraded_students_set, id_col='StudentID', first_col='FirstName', last_col='LastName'):
    highlight = 'background-color: #90EE90 !important;' 
    default = ''
    student_id = row.get(id_col)
    style = pd.Series(default, index=row.index) 
    if student_id and str(student_id) in upgraded_students_set: 
        if first_col in row.index: style[first_col] = highlight
        if last_col in row.index: style[last_col] = highlight
        # Only highlight ID if it's the designated column and no name columns were highlighted
        if id_col in row.index and not (first_col in row.index and style[first_col]==highlight) and not (last_col in row.index and style[last_col]==highlight):
             style[id_col] = highlight
    return style

# ============================================
# Streamlit App Layout
# ============================================
st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant v1.11") # Version update for fixes
st.info("**Workflow:**\n"
        "1. Set **Initial Parameters** (A+ Start, Uniform Gap) in the sidebar.\n"
        "2. Click **'Calculate Initial Cutoffs'**.\n"
        "3. Upload your **Score File** and **Map Columns**.\n"
        "4. Review **Cutoffs & Widths**, **Visualizations**, and **Students Below Cutoffs**.\n"
        "5. *Optionally:* Manually Adjust **Start Scores** and click **'Apply Manual Cutoffs'**.\n"
        "6. *Optionally:* Select students for manual upgrade *highlighting* in the **'Manual Upgrades'** section.\n"
        "7. Review **Final Results**.\n"
        "8. **Download** grades.")

# --- Sidebar ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score (>= score)", value=95.0, step=0.1, format="%.2f", help="Min score for A+.")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap (points)", value=5.0, step=0.1, min_value=0.1, format="%.2f", help="Initial gap between grade starts.")
points_near_cutoff = st.sidebar.number_input(
    "Show Students Below Cutoff (within X points)", min_value=0.1, max_value=10.0, value=1.5, step=0.1, format="%.1f", key='points_near_num_v5', # Incremented key
    help="Range *below* active cutoffs to highlight students.")

# Initialize session state
if 'initial_cutoffs' not in st.session_state: st.session_state.initial_cutoffs = None
if 'active_cutoffs' not in st.session_state: st.session_state.active_cutoffs = None
if 'df_graded' not in st.session_state: st.session_state.df_graded = None
if 'stats_results' not in st.session_state: st.session_state.stats_results = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'upgraded_students' not in st.session_state: st.session_state.upgraded_students = set()

# --- Button to Calculate Initial Cutoffs ---
if st.sidebar.button("Calculate Initial Cutoffs"):
    st.session_state.initial_cutoffs = calculate_initial_cutoffs_original(a_plus_start_score, uniform_grade_gap)
    st.session_state.active_cutoffs = st.session_state.initial_cutoffs
    # Initialize manual_override_values based on these initial cutoffs
    manual_vals = {grade_key: score for grade_key, score in st.session_state.initial_cutoffs.items()}
    # Ensure all grade start keys are present for the manual input fields
    all_grade_input_keys = [f'{g}_Start' for g in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']]
    for key in all_grade_input_keys:
        if key not in manual_vals: # Should be there from calculate_initial_cutoffs_original
            manual_vals[key] = 0.0 # Default if somehow missing
    if 'F_Max' not in manual_vals: # Should be set
        manual_vals['F_Max'] = manual_vals.get('D_Start', 0.0)

    st.session_state.manual_override_values = manual_vals
    st.session_state.df_graded = None 
    st.session_state.stats_results = None 
    st.session_state.upgraded_students = set()
    st.sidebar.success("Initial cutoffs calculated.")
    if st.session_state.data_loaded: st.rerun()

# --- Main Area ---
col_cutoff_table, col_width_table = st.columns([1, 2])
with col_cutoff_table:
    st.header("Active Cutoffs")
    cutoff_display_area = st.empty()
    if st.session_state.active_cutoffs:
        cutoff_df = pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Grade Boundary', 'Score'])
        cutoff_display_area.dataframe(cutoff_df.style.format({"Score": "{:.2f}"}))
    else:
        cutoff_display_area.warning("Calculate initial cutoffs or apply manual cutoffs first.")

with col_width_table:
    if st.session_state.active_cutoffs:
        st.header("Grade Widths")
        active_cutoff_map = {g: s for g, s in st.session_state.active_cutoffs.items() if g != 'F_Max'}
        
        # Sort grades by their defined start scores, highest score first
        # The keys in active_cutoff_map are like 'A+_Start', 'A_Start'
        # We assume active_cutoffs is already in logical grade order (A+, A, B+...)
        # So, no need to re-sort by score if order is already A+, A, B+ ...
        # However, for width calculation, it's useful to have them by score to find upper/lower.
        
        # Correct approach: Iterate through grades in their logical order (A+ first)
        # The st.session_state.active_cutoffs SHOULD be an OrderedDict in the correct grade order.
        
        grades_in_desc_order = [key for key in st.session_state.active_cutoffs.keys() if key != 'F_Max'] # e.g. A+, A, B+...

        widths = collections.OrderedDict()
        max_score_cap = 100.0 # Assuming 100 is the max possible score for the highest grade's range end

        for i, grade_key in enumerate(grades_in_desc_order):
            start_score = st.session_state.active_cutoffs[grade_key]
            # Upper bound is max_score_cap for the highest grade (A+), or the start of the next higher grade
            if i == 0: # Highest grade (e.g., A+)
                upper_bound = max_score_cap 
            else:
                # The previous grade in this list is the one with a higher start score
                prev_grade_key = grades_in_desc_order[i-1] 
                upper_bound = st.session_state.active_cutoffs[prev_grade_key]
            
            width_val = upper_bound - start_score
            width_val = 0.0 if np.isclose(width_val, 0) or width_val < 0 else width_val # Width cannot be negative
            
            # Label for the grade (e.g. A+ from A+_Start)
            grade_label = grade_key.replace('_Start', '')
            widths[grade_label] = f"{width_val:.2f} [{start_score:.2f} - {upper_bound:.2f})"
        
        if widths:
            width_df = pd.DataFrame(list(widths.items()), columns=['Grade', 'Width [Start - End)'])
            st.dataframe(width_df)
        else:
            st.write("Not enough grade boundaries to calculate widths.")


# --- Upload Section ---
st.header("Upload & Prepare Data")
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"], key="file_uploader_v12") # Incremented key

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
        
        col_first = st.selectbox("First Name Column (Optional)", options=cols_with_none, index=0, key='sel_first_v12')
        col_last = st.selectbox("Last Name Column (Optional)", options=cols_with_none, index=0, key='sel_last_v12')
        col_id = st.selectbox("Student ID Column (Optional)", options=cols_with_none, index=0, key='sel_id_v13') # Incremented key
        col_score = st.selectbox("Score Column*", options=cols_with_none, index=0, key='sel_score_v13')
        col_section = st.selectbox("Section Column*", options=cols_with_none, index=0, key='sel_section_v13')
        st.caption("*Mandatory columns")

        if col_score != "<Select Column>" and col_section != "<Select Column>":
            df = df_upload.copy()
            df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
            
            final_cols_to_keep = ['Score', 'Section']
            # Handle optional columns and generate StudentID if not provided
            if col_id != "<Select Column>":
                df.rename(columns={col_id: 'StudentID'}, inplace=True)
                final_cols_to_keep.append('StudentID')
            elif 'StudentID' not in df.columns: # If no ID col selected AND 'StudentID' doesn't already exist
                df['StudentID'] = 'Stud_' + df.index.astype(str) # Generate unique ID
                final_cols_to_keep.append('StudentID')
            
            if col_first != "<Select Column>":
                df.rename(columns={col_first: 'FirstName'}, inplace=True)
                final_cols_to_keep.append('FirstName')
            if col_last != "<Select Column>":
                df.rename(columns={col_last: 'LastName'}, inplace=True)
                final_cols_to_keep.append('LastName')
            
            # Ensure only specified or generated columns are kept, handling cases where rename might not happen
            df = df[[col for col in final_cols_to_keep if col in df.columns]]

            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
            initial_rows = len(df)
            df.dropna(subset=['Score'], inplace=True)
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")
            
            if df.empty:
                st.error("No valid score data remaining.")
                st.session_state.data_loaded = False
                st.session_state.processed_df = None
            else:
                df['Section'] = df['Section'].astype(str)
                if 'StudentID' in df.columns: df['StudentID'] = df['StudentID'].astype(str)
                if 'FirstName' in df.columns: df['FirstName'] = df['FirstName'].astype(str).fillna('')
                if 'LastName' in df.columns: df['LastName'] = df['LastName'].astype(str).fillna('')
                
                st.session_state.processed_df = df
                st.session_state.data_loaded = True
                st.success("Data loaded and columns mapped.")
                reset_section_color_cycle() # Reset colors for new data

                st.subheader("Data Preview")
                # FIX 4: Streamlined Data Preview Column Logic
                preview_cols_ordered = []
                if 'FirstName' in df.columns: preview_cols_ordered.append('FirstName')
                if 'LastName' in df.columns: preview_cols_ordered.append('LastName')
                if not preview_cols_ordered and 'StudentID' in df.columns:
                    preview_cols_ordered.append('StudentID')
                for col_m in ['Score', 'Section']: # Mandatory renamed columns
                    if col_m in df.columns and col_m not in preview_cols_ordered:
                        preview_cols_ordered.append(col_m)
                
                final_preview_cols_to_show = [col for col in preview_cols_ordered if col in df.columns]
                if final_preview_cols_to_show:
                    st.dataframe(df[final_preview_cols_to_show].head())
                else:
                    st.warning("Could not determine columns for data preview.")

                # Reset downstream states
                st.session_state.df_graded = None
                st.session_state.stats_results = None
                st.session_state.upgraded_students = set() # Reset manual upgrades for new data
        else:
            st.warning("Please select the Score and Section columns.")
            st.session_state.data_loaded = False
            st.session_state.processed_df = None
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        st.session_state.data_loaded = False
        st.session_state.processed_df = None

# Assign df_display for use in downstream sections
df_display = None # Initialize
if st.session_state.data_loaded and st.session_state.processed_df is not None:
    df_display = st.session_state.processed_df

# --- Sections requiring data and active cutoffs ---
if st.session_state.data_loaded and df_display is not None and st.session_state.active_cutoffs is not None:

    # --- Manual Cutoff Adjustment ---
    st.header("Manual Cutoff Adjustment")
    st.markdown("Adjust the **Start Score** for each grade here and click 'Apply'. Review visualizations below.")
    
    manual_cutoffs_input = {}
    # Define the order of grades for input fields
    grade_keys_in_order = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    
    # Use session state for default values in number_input to persist changes
    current_manual_vals = st.session_state.manual_override_values 

    # Create columns for layout
    col_defs = [[st.columns(4) for _ in range(2)]] # Creates 2 rows of 4 columns
    flat_cols = [item for sublist in col_defs[0] for item in sublist] # Flatten the list of columns

    for i, key in enumerate(grade_keys_in_order):
        with flat_cols[i]:
            grade_label = key.replace('_Start', ' Start')
            # Get default value from session state, fallback to active_cutoffs or 0.0
            default_val = float(current_manual_vals.get(key, st.session_state.active_cutoffs.get(key, 0.0)))
            manual_cutoffs_input[key] = st.number_input(
                grade_label, value=default_val, step=0.1, 
                key=f'man_{key}_v13', format="%.2f" # Incremented key
            )
    
    # F_Max is tied to D_Start from the inputs
    manual_cutoffs_input['F_Max'] = manual_cutoffs_input.get('D_Start', 0.0) 

    if st.button("Apply Manual Cutoffs & Recalculate"):
        # Validate monotonicity (A+ score >= A score, etc.)
        scores_list = [manual_cutoffs_input[key] for key in grade_keys_in_order if key in manual_cutoffs_input]
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)):
            st.error("Manual Start Scores for grades (A+ down to D) must be monotonically decreasing or equal (e.g., A+ score >= A score).")
        else:
            # FIX 2: Construct active_cutoffs by explicit grade order
            new_active_cutoffs = collections.OrderedDict()
            # Iterate through the defined grade order to build the OrderedDict
            all_grade_keys_ordered = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start', 'F_Max']
            for key in all_grade_keys_ordered:
                if key in manual_cutoffs_input:
                     new_active_cutoffs[key] = manual_cutoffs_input[key]
            
            st.session_state.active_cutoffs = new_active_cutoffs
            st.session_state.manual_override_values = manual_cutoffs_input.copy() # Store raw inputs for next time
            
            st.session_state.df_graded = None
            st.session_state.stats_results = None
            # Upgraded students set is usually independent of cutoff changes, but can be reset if desired
            # st.session_state.upgraded_students = set() 
            st.success("Manual cutoffs applied. Recalculating results...")
            st.rerun()

    # --- Perform grade assignment and stats calculation ---
    # This block runs if data, display_df, and active_cutoffs are available
    # It will also run after a st.rerun() from manual cutoff application
    if st.session_state.df_graded is None: # Calculate only if not already done or reset
        try:
            df_calc = df_display.copy()
            df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)
            
            if not df_calc['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
                st.session_state.stats_results = calculate_stats(df_calc, 'Letter_Grade', 'Section', GPA_SCALE)
                st.session_state.df_graded = df_calc # Store the successfully graded DataFrame
            else:
                st.error("Statistics not calculated due to errors in grade assignment. Please check cutoff settings and input data.")
                st.session_state.stats_results = None
                st.session_state.df_graded = None # Ensure it's None if grading failed
        except Exception as e:
            st.error(f"Error during grade assignment or statistics calculation: {e}")
            st.session_state.stats_results = None
            st.session_state.df_graded = None

    # --- Visualization & Observation ---
    st.header("Visualization & Observation")
    # Create a map of grade start scores (excluding F_Max) for plotting lines
    active_cutoff_plot_map = {
        grade.replace("_Start", ""): score 
        for grade, score in st.session_state.active_cutoffs.items() 
        if grade != 'F_Max'
    }
    # Unique, sorted scores for drawing vertical lines on plots
    unique_cutoff_scores_for_plot = sorted(list(set(active_cutoff_plot_map.values())))

    st.subheader("Score Distribution with Active Cutoffs")
    hist_col, slider_col = st.columns([4, 1])
    with slider_col:
        num_bins = st.slider("Histogram Bins", 5, 50, 25, key='hist_bins_v10') # Incremented key
    with hist_col:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(df_display['Score'], kde=False, ax=ax_hist, bins=num_bins, stat="density") # Use density for KDE overlay
        sns.kdeplot(df_display['Score'], ax=ax_hist, color='orange', warn_singular=False)
        for cutoff_val in unique_cutoff_scores_for_plot:
            ax_hist.axvline(cutoff_val, color='red', linestyle='--', linewidth=1)
        ax_hist.set_title("Score Distribution with Grade Cutoffs")
        ax_hist.set_xlabel("Score")
        ax_hist.set_ylabel("Density / Frequency")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    st.subheader("Individual Scores with Active Cutoffs (Colored by Section)")
    if 'Section' in df_display.columns:
        reset_section_color_cycle() # Ensure fresh color cycle for this plot
        unique_sections_plot = sorted(df_display['Section'].unique())
        plot_palette = {section: get_section_color_fixed(section) for section in unique_sections_plot}
        
        fig_strip, ax_strip = plt.subplots(figsize=(10, max(4, len(unique_sections_plot) * 0.5)))
        sns.stripplot(data=df_display, x='Score', y='Section', hue='Section', 
                      order=unique_sections_plot, hue_order=unique_sections_plot, 
                      ax=ax_strip, jitter=0.3, size=4, alpha=0.7, legend="auto", palette=plot_palette)
        for cutoff_val in unique_cutoff_scores_for_plot:
            ax_strip.axvline(cutoff_val, color='red', linestyle='--', linewidth=1)
        ax_strip.set_title("Individual Scores by Section with Grade Cutoffs")
        ax_strip.set_xlabel("Score")
        ax_strip.set_ylabel("Section")
        if len(unique_sections_plot) > 1 : # Show legend if multiple sections
             ax_strip.legend(title='Section', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            if ax_strip.get_legend() is not None: ax_strip.get_legend().remove()

        plt.tight_layout(rect=[0, 0, 0.9 if len(unique_sections_plot) > 1 else 1, 1])
        st.pyplot(fig_strip)
        plt.close(fig_strip)
    else:
        st.warning("Section column not found in data, cannot generate section-colored plot.")

    # --- Students Near Cutoffs ---
    st.subheader(f"Students Below Cutoffs (within {points_near_cutoff} points)")
    students_near_df_local = pd.DataFrame() 
    
    # Use the graded DataFrame from session state if available and valid
    df_for_near_cutoff_check = st.session_state.df_graded 
    
    if df_for_near_cutoff_check is not None and \
       'Letter_Grade' in df_for_near_cutoff_check.columns and \
       not df_for_near_cutoff_check['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
        
        # FIX 3: Accurate target_grade determination
        # Create a map from a unique score boundary to the HIGHEST grade label associated with it.
        # st.session_state.active_cutoffs is an OrderedDict (A+ to D).
        score_to_definitive_grade_label_map = collections.OrderedDict()
        for grade_key_suffix, score_value in st.session_state.active_cutoffs.items():
            if grade_key_suffix == 'F_Max':
                continue
            grade_lbl = grade_key_suffix.replace('_Start', '')
            if score_value not in score_to_definitive_grade_label_map: # First time seeing score, so it's highest grade
                score_to_definitive_grade_label_map[score_value] = grade_lbl
        
        students_near_cutoff_list = []
        # unique_cutoff_scores_for_plot are the actual grade boundaries (e.g., D_Start, C_Start, etc.)
        for boundary_score_val in unique_cutoff_scores_for_plot: # Iterate through actual grade start scores
            min_score_for_nearby = boundary_score_val - points_near_cutoff
            
            # Select students between min_score_for_nearby (inclusive) and boundary_score_val (exclusive)
            nearby_students_df = df_for_near_cutoff_check[
                (df_for_near_cutoff_check['Score'] >= min_score_for_nearby) & 
                (df_for_near_cutoff_check['Score'] < boundary_score_val)
            ].copy() # Use .copy() to avoid SettingWithCopyWarning

            if not nearby_students_df.empty:
                nearby_students_df['Target_Boundary'] = boundary_score_val
                nearby_students_df['Points_to_Upgrade'] = boundary_score_val - nearby_students_df['Score']
                target_grade_label = score_to_definitive_grade_label_map.get(boundary_score_val, "N/A")
                nearby_students_df['Target_Grade'] = target_grade_label
                students_near_cutoff_list.append(nearby_students_df)
        
        if students_near_cutoff_list:
            students_near_df_local = pd.concat(students_near_cutoff_list).sort_values(by=['Target_Boundary', 'Score'])
            
            # Determine columns to display, prioritizing names
            name_cols_near = []
            if 'FirstName' in students_near_df_local.columns: name_cols_near.append('FirstName')
            if 'LastName' in students_near_df_local.columns: name_cols_near.append('LastName')
            if not name_cols_near and 'StudentID' in students_near_df_local.columns:
                name_cols_near.append('StudentID')
            
            cols_near_display = name_cols_near + ['Score', 'Section', 'Letter_Grade', 'Target_Grade', 'Target_Boundary', 'Points_to_Upgrade']
            cols_near_display_exist = [col for col in cols_near_display if col in students_near_df_local.columns]
            
            st.dataframe(students_near_df_local[cols_near_display_exist].style.format({
                "Score": "{:.2f}", "Points_to_Upgrade": "{:.2f}", "Target_Boundary": "{:.2f}"
            }))
            st.caption("Tip: To upgrade a student's grade, adjust the 'Start Score' for their 'Target_Grade' in the 'Manual Cutoff Adjustment' section, then click 'Apply'. Optionally, select them below for highlighting in the final table.")
        else:
            st.write(f"No students found scoring between (Cutoff - {points_near_cutoff:.1f}) and (Cutoff).")
    else:
        st.warning("Grade calculation needed or has errors. Cannot show students near cutoffs.")

    # --- Manual Upgrade Selection (using students_near_df_local) ---
    st.header("Manual Upgrades (Highlighting Only)")
    st.markdown("Select students *from the 'Students Below Cutoffs' list* whose rows should be highlighted green in the final table. This does NOT change their grade, only highlights the row.")
    
    if not students_near_df_local.empty:
        student_identifier_col_name = 'StudentID' if 'StudentID' in students_near_df_local.columns else None
        if student_identifier_col_name:
            # Ensure unique students for selection, even if they appear near multiple cutoffs
            eligible_students_for_highlight_df = students_near_df_local.drop_duplicates(subset=[student_identifier_col_name]).copy() # Use .copy()
            
            # Create display labels
            if 'FirstName' in eligible_students_for_highlight_df.columns and 'LastName' in eligible_students_for_highlight_df.columns:
                eligible_students_for_highlight_df.loc[:, 'DisplayLabel'] = eligible_students_for_highlight_df['FirstName'] + " " + \
                                                               eligible_students_for_highlight_df['LastName'] + " (" + \
                                                               eligible_students_for_highlight_df[student_identifier_col_name] + ")"
            else:
                eligible_students_for_highlight_df.loc[:, 'DisplayLabel'] = eligible_students_for_highlight_df[student_identifier_col_name]
            
            student_options_for_multiselect = sorted(eligible_students_for_highlight_df['DisplayLabel'].tolist())
            # Map display label back to the actual student ID
            student_id_map_for_multiselect = pd.Series(
                eligible_students_for_highlight_df[student_identifier_col_name].values, 
                index=eligible_students_for_highlight_df['DisplayLabel']
            ).to_dict()

            # Get currently selected students (their labels) to pre-fill multiselect
            currently_selected_student_labels = [
                label for label, student_id_val in student_id_map_for_multiselect.items() 
                if student_id_val in st.session_state.upgraded_students
            ]

            selected_display_labels = st.multiselect(
                "Highlight Students Below Cutoff as Upgraded:", 
                options=student_options_for_multiselect, 
                default=currently_selected_student_labels, 
                key="manual_upgrade_select_v5" # Incremented key
            )
            
            # Convert selected labels back to set of student IDs
            newly_selected_student_ids = set(
                student_id_map_for_multiselect.get(label) for label in selected_display_labels 
                if student_id_map_for_multiselect.get(label) is not None
            )

            if st.button("Update Upgrade Highlighting"):
                if newly_selected_student_ids != st.session_state.upgraded_students:
                    st.session_state.upgraded_students = newly_selected_student_ids
                    st.success("Upgrade highlighting selection updated.")
                    st.rerun() # Rerun to apply highlighting to the final table
                else:
                    st.info("Highlighting selection unchanged.")
        else:
            st.warning("StudentID column not found in the 'near cutoffs' list, cannot provide manual upgrade selection.")
    else:
        st.markdown("_Students potentially needing upgrades (from the 'Students Below Cutoffs' list) will appear here once grades are calculated and such students exist._")
    
    # --- Display Final Results ---
    st.header("Final Results (Based on Active Cutoffs)")
    if st.session_state.stats_results and st.session_state.stats_results.get("error") is None:
        results = st.session_state.stats_results
        st.subheader("Grade Distributions")
        if "overall_dist" in results and not results['overall_dist'].empty and \
           "section_dist" in results and not results['section_dist'].empty:
            try:
                overall_series = results['overall_dist'].rename('Overall (%)') * 100
                section_dist_df = results['section_dist'] * 100
                
                # Define a comprehensive order for grades, including possible error states
                all_grades_ordered_list = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F'] + \
                                     [g for g in GPA_SCALE.keys() if 'Error' in g or 'Invalid' in g]

                # Get grades present in the data, sorted by GPA scale (desc) then by the list order
                present_grades_in_data = overall_series.index.union(section_dist_df.columns)
                
                # Filter all_grades_ordered_list to only those present, maintaining preferred order
                grades_for_display_sorted = [
                    g for g in all_grades_ordered_list if g in present_grades_in_data
                ]
                # Add any other grades from data not in the preferred list (should be rare)
                grades_for_display_sorted.extend([g for g in present_grades_in_data if g not in grades_for_display_sorted])


                section_dist_df = section_dist_df.reindex(columns=grades_for_display_sorted, fill_value=0)
                overall_series = overall_series.reindex(grades_for_display_sorted, fill_value=0)
                
                combined_dist_df = pd.concat([overall_series, section_dist_df.T], axis=1).fillna(0)
                combined_dist_df.index.name = 'Grade'
                
                styler_dist = combined_dist_df.style.format("{:.1f}%").highlight_null(color='transparent')
                st.markdown(styler_dist.to_html(escape=False, index=True), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying distribution table: {e}")
                st.write("Raw Overall Distribution:", results.get("overall_dist"))
                st.write("Raw Section Distribution:", results.get("section_dist"))
        elif "overall_dist" in results and not results['overall_dist'].empty:
            st.write("**Overall Distribution Only:**")
            st.dataframe((results['overall_dist'] * 100).map("{:.1f}%".format))
        else:
            st.write("Distribution data could not be calculated or is empty.")

        overall_gpa_val = results.get('overall_gpa', np.nan)
        st.write(f"**Overall Avg GPA:** {overall_gpa_val:.2f}" if pd.notna(overall_gpa_val) else "**Overall Avg GPA:** N/A")
        st.write("---")

        st.subheader("Section GPA Comparison")
        col_gpa_table, col_gpa_plot = st.columns(2)
        with col_gpa_table:
            st.write("**Per Section Avg GPA:**")
            if "section_stats" in results and not results['section_stats'].empty:
                st.dataframe(
                    results['section_stats'][['Section', 'Avg_GPA', 'Valid_GPA_Count', 'Count']].rename(
                        columns={'Valid_GPA_Count':'Graded Students', 'Count':'Total Students'}
                    ).style.format({"Avg_GPA": "{:.2f}"}, na_rep='N/A')
                )
            else:
                st.write("Section GPA statistics not available.")
            
            st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
            anova_p = results.get('anova_p_value')
            if anova_p is not None and anova_p < 0.05:
                st.warning(f"Significant difference in section GPAs detected (p={anova_p:.3f}).")

        with col_gpa_plot:
            if st.session_state.df_graded is not None and \
               'GPA' in st.session_state.df_graded.columns and \
               st.session_state.df_graded['GPA'].notna().any():
                try:
                    fig_box, ax_box = plt.subplots()
                    # Sort sections for consistent boxplot order
                    sorted_sections_for_plot = sorted(st.session_state.df_graded['Section'].unique())
                    sns.boxplot(data=st.session_state.df_graded, x='Section', y='GPA', ax=ax_box, order=sorted_sections_for_plot)
                    ax_box.set_title("GPA Distribution by Section")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_box)
                    plt.close(fig_box)
                except Exception as e:
                    st.warning(f"Could not generate section GPA boxplot: {e}")
            else:
                st.warning("GPA data not available for boxplot.")
        
        # --- Failing Students Analysis ---
        st.subheader("Failing Students Analysis")
        if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns:
            passing_score_boundary = st.session_state.active_cutoffs.get('D_Start', None)
            if passing_score_boundary is not None:
                failing_students_df = st.session_state.df_graded[st.session_state.df_graded['Letter_Grade'] == 'F'].copy()
                if not failing_students_df.empty:
                    failing_students_df['Points_Below_Pass'] = passing_score_boundary - failing_students_df['Score']
                    failing_students_df.sort_values('Points_Below_Pass', ascending=True, inplace=True)
                    
                    st.write(f"Passing Score (Minimum for D): {passing_score_boundary:.2f}")
                    fail_cols_display = []
                    if 'FirstName' in failing_students_df.columns: fail_cols_display.append('FirstName')
                    if 'LastName' in failing_students_df.columns: fail_cols_display.append('LastName')
                    if not fail_cols_display and 'StudentID' in failing_students_df.columns:
                        fail_cols_display.append('StudentID')
                    fail_cols_display.extend(['Score', 'Section', 'Points_Below_Pass'])
                    
                    cols_fail_exist_display = [col for col in fail_cols_display if col in failing_students_df.columns]
                    st.dataframe(failing_students_df[cols_fail_exist_display].style.format({"Score": "{:.2f}", "Points_Below_Pass": "{:.2f}"}))
                else:
                    st.success("No students received an 'F' grade based on active cutoffs.")
            else:
                st.warning("Could not determine passing score ('D_Start') from active cutoffs to analyze failing students.")
        else:
            st.warning("Final grades needed to analyze failing students (or grades contain errors).")

        # --- Final Assigned Grades Table ---
        st.subheader("Final Assigned Grades Table")
        if st.session_state.df_graded is not None:
            df_final_display = st.session_state.df_graded.copy()
            
            display_cols_order = ['FirstName', 'LastName', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
            display_cols_exist_final = [col for col in display_cols_order if col in df_final_display.columns]
            df_to_style = df_final_display[display_cols_exist_final]

            reset_section_color_cycle() # Reset for consistent section colors in this table
            unique_sections_final_table = sorted(df_to_style['Section'].unique())
            for section_name_val in unique_sections_final_table: get_section_color_fixed(section_name_val) # Pre-populate map

            styler = df_to_style.style
            if 'Letter_Grade' in df_to_style.columns:
                styler = styler.apply(
                    lambda x: [f'background-color: {GRADE_GRADIENT.get(str(v), GRADE_GRADIENT["default"])}' for v in x], 
                    subset=['Letter_Grade']
                )
            if 'Section' in df_to_style.columns:
                styler = styler.apply(
                    lambda x: [f'background-color: {get_section_color_fixed(str(v))}' for v in x], 
                    subset=['Section']
                )
            
            # Apply upgrade highlight row-wise
            if 'StudentID' in df_to_style.columns and st.session_state.upgraded_students:
                id_col_for_highlight = 'StudentID'
                name_cols_for_highlight = [col for col in ['FirstName', 'LastName'] if col in df_to_style.columns]
                # Determine which columns to pass to the highlight function's subset
                subset_cols_for_highlight = name_cols_for_highlight if name_cols_for_highlight else \
                                           ([id_col_for_highlight] if id_col_for_highlight in df_to_style.columns else [])
                
                if subset_cols_for_highlight: # Only apply if there are columns to highlight
                    styler = styler.apply(
                        highlight_upgraded, 
                        upgraded_students_set=st.session_state.upgraded_students, 
                        id_col=id_col_for_highlight,
                        first_col='FirstName' if 'FirstName' in df_to_style.columns else None, # Pass actual column names
                        last_col='LastName' if 'LastName' in df_to_style.columns else None,
                        axis=1, 
                        subset=subset_cols_for_highlight # Critical: subset tells apply which cols `row` will contain
                    )
            
            st.markdown(
                styler.format({"Score": "{:.2f}", "GPA": "{:.2f}"}).hide(axis="index").to_html(escape=False), 
                unsafe_allow_html=True
            )

            # --- Download Section ---
            st.subheader("Download Grades")
            sections_for_download = ["All Sections"] + sorted(df_final_display['Section'].unique().tolist())
            selected_section_download = st.selectbox(
                "Select section to download:", 
                options=sections_for_download, 
                key="download_section_select_v10" # Incremented key
            )
            
            def convert_df_to_csv_download(df_to_convert, section_filter_val):
                if section_filter_val != "All Sections":
                    df_filtered_dl = df_to_convert[df_to_convert['Section'] == section_filter_val].copy()
                else:
                    df_filtered_dl = df_to_convert.copy()
                
                # Use the same column order as display for consistency
                dl_cols_exist = [col for col in display_cols_order if col in df_filtered_dl.columns]
                if not dl_cols_exist: return None
                return df_filtered_dl[dl_cols_exist].to_csv(index=False).encode('utf-8')

            try:
                csv_data_download = convert_df_to_csv_download(df_final_display, selected_section_download)
                if csv_data_download:
                    file_name_dl = f"final_grades_{selected_section_download.replace(' ', '_')}.csv" \
                                   if selected_section_download != "All Sections" else "final_grades_all_sections.csv"
                    st.download_button(
                        label=f"Download Grades for {selected_section_download}", 
                        data=csv_data_download, 
                        file_name=file_name_dl, 
                        mime='text/csv', 
                        key=f"download_btn_{selected_section_download}_v10" # Incremented key
                    )
            except Exception as e:
                st.error(f"Could not prepare download file: {e}")
        else:
            st.warning("Final graded data not yet available for display or download.")
    elif st.session_state.active_cutoffs: # If stats results had an error but cutoffs exist
        st.warning("Statistics could not be calculated. Check data, grade assignment, and cutoffs. Some information might be missing.")
    # If no active_cutoffs, the main conditional block `if st.session_state.data_loaded ...` handles this.

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Iterative Grading Tool v1.11")
