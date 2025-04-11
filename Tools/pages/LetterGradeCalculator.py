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
# calculate_initial_cutoffs_original - unchanged
def calculate_initial_cutoffs_original(a_plus_start, gap):
    cutoffs = collections.OrderedDict(); grades_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']
    current_start = a_plus_start
    for i, grade in enumerate(grades_order): cutoffs[f'{grade}_Start'] = current_start; current_start -= gap if i < len(grades_order) - 1 else 0
    cutoffs['F_Max'] = cutoffs['D_Start']; return cutoffs

# assign_letter_grades_from_starts - unchanged
def assign_letter_grades_from_starts(scores, start_cutoffs):
    lower_bounds_map = {grade: score for grade, score in start_cutoffs.items() if grade != 'F_Max'}
    boundary_scores = sorted(list(set(lower_bounds_map.values())))
    if not all(boundary_scores[i] < boundary_scores[i+1] for i in range(len(boundary_scores)-1)):
        unique_scores_count = len(boundary_scores); total_scores_count = len(lower_bounds_map)
        if unique_scores_count < total_scores_count:
             if not all(boundary_scores[i] <= boundary_scores[i+1] for i in range(len(boundary_scores)-1)):
                 st.error("Grade boundaries must increase monotonically (or be equal)."); return pd.Series(['Error - Non-monotonic Cutoffs'] * len(scores), index=scores.index, dtype='object')
             else: st.warning("Duplicate start scores detected; using unique boundaries.")
        else: st.error("Grade boundaries must increase monotonically."); return pd.Series(['Error - Non-monotonic Cutoffs'] * len(scores), index=scores.index, dtype='object')
    bins = [-np.inf] + boundary_scores + [np.inf]
    grades_sorted_by_score = sorted(lower_bounds_map.keys(), key=lambda grade: lower_bounds_map[grade])
    labels = ['F'] + grades_sorted_by_score
    if len(labels) != len(bins) - 1: st.error(f"CRITICAL: Bin/Label mismatch."); return pd.Series(['Error - Bin/Label Mismatch'] * len(scores), index=scores.index, dtype='object')
    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels, right=False, ordered=True)
    grades = grades.astype('object').fillna('Invalid Score'); return grades

# <<< Added Debugging Prints (Commented Out) >>>
def calculate_stats(df, grade_col, section_col, gpa_map):
    """Calculates distributions and statistics. Corrected GPA handling + Debug."""
    if grade_col not in df.columns or df[grade_col].astype(str).str.contains('Error', na=False).any():
        st.warning("Cannot calculate stats due to errors in grade assignment.")
        return {"error": "Grade assignment failed."}

    # -- Debugging: Check grades before mapping --
    # unique_grades_before_map = df[grade_col].unique()
    # st.sidebar.write("DEBUG: Unique Grades Before Map:", unique_grades_before_map)
    # st.sidebar.write("DEBUG: GPA Map Keys:", list(gpa_map.keys()))
    # unmatched = [g for g in unique_grades_before_map if g not in gpa_map]
    # if unmatched: st.sidebar.warning(f"DEBUG: Unmatched grades for GPA: {unmatched}")
    # --------------------------------------------

    df['GPA'] = df[grade_col].map(gpa_map)

    # -- Debugging: Check GPA after mapping ---
    # st.sidebar.write("DEBUG: GPA after map (head):", df['GPA'].head())
    # st.sidebar.write("DEBUG: GPA NaNs after map:", df['GPA'].isna().sum())
    # -----------------------------------------

    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')

    # -- Debugging: Check GPA after to_numeric ---
    # st.sidebar.write("DEBUG: GPA after to_numeric (head):", df['GPA'].head())
    # st.sidebar.write("DEBUG: GPA dtype after to_numeric:", df['GPA'].dtype)
    # st.sidebar.write("DEBUG: GPA NaNs after to_numeric:", df['GPA'].isna().sum())
    # st.sidebar.write("DEBUG: GPA Zeros after to_numeric:", (df['GPA'] == 0).sum())
    # -------------------------------------------

    if df['GPA'].isnull().all() and df[grade_col].notna().any() and not df[grade_col].isin(['Invalid Score']).all():
         st.warning("GPA calculation resulted in all non-numeric values. Check GPA Scale.")

    # --- Debugging: Check Value Counts ---
    # try:
    #     st.sidebar.write("DEBUG: Grade Counts:\n", df[grade_col].value_counts())
    # except Exception as e:
    #     st.sidebar.error(f"DEBUG: Error getting value_counts: {e}")
    # -----------------------------------

    overall_dist = df[grade_col].value_counts(normalize=True).sort_index()
    overall_gpa = df['GPA'].mean() # NaNs ignored

    df[section_col] = df[section_col].astype(str)
    # Calculate Avg_GPA carefully
    section_gpa_means = df.groupby(section_col)['GPA'].mean() # Default skipna=True
    # st.sidebar.write("DEBUG: Section GPA Means (raw):", section_gpa_means) # Debugging

    section_stats = df.groupby(section_col).agg(Count=('GPA', 'size'), Valid_GPA_Count = ('GPA', 'count')).reset_index()
    # Merge the calculated means
    section_stats = pd.merge(section_stats, section_gpa_means.rename('Avg_GPA'), on=section_col, how='left')
    # st.sidebar.write("DEBUG: Section Stats (merged):", section_stats) # Debugging


    try: section_dist = df.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)
    except Exception: section_dist = pd.DataFrame()

    # ANOVA unchanged
    anova_result = "ANOVA not applicable."; anova_p_value = None
    section_groups = [group['GPA'].dropna().values for name, group in df.groupby(section_col) if group['GPA'].notna().sum() > 1]
    if len(section_groups) > 1:
        valid_groups = [g for g in section_groups if len(g) > 0]
        if len(valid_groups) > 1:
            try: f_val, p_val = stats.f_oneway(*valid_groups); anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"; anova_p_value = p_val
            except ValueError as e: anova_result = f"ANOVA Error: {e}"

    return {"overall_dist": overall_dist, "overall_gpa": overall_gpa, "section_stats": section_stats, "section_dist": section_dist,
            "anova_result": anova_result, "anova_p_value": anova_p_value, "error": None}

# GPA Scale remains the same
GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0,
             'Invalid Score': np.nan, 'Error - Non-monotonic Cutoffs': np.nan, 'Error - Bin/Label Mismatch': np.nan}

# --- Styling Functions ---
GRADE_GRADIENT = { # White to Black/Gray
    'A+': '#FFFFFF', 'A': '#F2F2F2', 'B+': '#E6E6E6', 'B': '#D9D9D9',
    'C+': '#CCCCCC', 'C': '#BDBDBD', 'D+': '#B0B0B0', 'D': '#A3A3A3',
    'F': '#969696', 'default': '#FFFFFF' # White for errors etc.
}
SECTION_COLOR_PALETTE = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                       '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a'] # Example: Colorbrewer Paired
section_color_map_generated = {}
color_cycle = itertools.cycle(SECTION_COLOR_PALETTE)
def get_section_color_fixed(section_name):
    if section_name not in section_color_map_generated: section_color_map_generated[section_name] = next(color_cycle)
    return section_color_map_generated[section_name]

# Separate function for upgrade highlight - easier to manage
def highlight_upgraded(row, upgraded_students_set, id_col='StudentID', first_col='FirstName', last_col='LastName'):
    highlight = 'background-color: #90EE90' # LightGreen
    default = ''
    student_id = row.get(id_col)
    if student_id in upgraded_students_set:
        # Apply highlight to name/id columns that exist
        style = pd.Series(default, index=row.index)
        if first_col in row.index: style[first_col] = highlight
        if last_col in row.index: style[last_col] = highlight
        if id_col in row.index and first_col not in row.index and last_col not in row.index:
            style[id_col] = highlight # Highlight ID only if no name columns
        return style
    return pd.Series(default, index=row.index)


# ============================================
# Streamlit App Layout
# ============================================

st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant v1.9") # Version update
# ... (Instructions markdown remains largely the same) ...
st.info("**Workflow:**\n"
        "1. Set **Initial Parameters** (A+ Start, Uniform Gap) in the sidebar.\n"
        "2. Click **'Calculate Initial Cutoffs'**.\n"
        "3. Upload your **Score File** and **Map Columns**.\n"
        "4. Review **Cutoffs & Widths**, **Visualizations**, and **Students Below Cutoffs**.\n"
        "5. *Optionally:* Manually Adjust **Start Scores** and click **'Apply Manual Cutoffs'**.\n"
        "6. *Optionally:* Select students for manual upgrade *highlighting* in the **'Manual Upgrades'** section (use info from step 4 to inform cutoff adjustments in step 5).\n"
        "7. Review **Final Results**.\n"
        "8. **Download** grades.")

# --- Sidebar ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score (>= score)", value=95.0, step=0.1, format="%.2f", help="Min score for A+.")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap (points)", value=5.0, step=0.1, min_value=0.1, format="%.2f", help="Initial gap between grade starts.")
# <<< Changed Slider to Number Input & Updated Label >>>
points_near_cutoff = st.sidebar.number_input(
    "Show Students Below Cutoff (within X points)",
    min_value=0.1, max_value=10.0, value=1.5, step=0.1, format="%.1f", key='points_near_num_v3', # Incremented key
    help="Range *below* active cutoffs to highlight students potentially needing an upgrade.")

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
    manual_vals = {grade: score for grade, score in st.session_state.initial_cutoffs.items()}
    for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']:
        if f'{grade}_Start' not in manual_vals: manual_vals[f'{grade}_Start'] = 0.0
    if 'F_Max' not in manual_vals: manual_vals['F_Max'] = manual_vals.get('D_Start', 0.0)
    st.session_state.manual_override_values = manual_vals
    st.session_state.df_graded = None; st.session_state.stats_results = None
    st.session_state.upgraded_students = set()
    st.sidebar.success("Initial cutoffs calculated.")
    if st.session_state.data_loaded: st.rerun()

# --- Main Area ---
# <<< Added Header to Cutoff Table Column >>>
col_cutoff_table, col_width_table = st.columns([1, 2])
with col_cutoff_table:
    st.header("Active Cutoffs") # Alignment Header
    cutoff_display_area = st.empty()
    if st.session_state.active_cutoffs:
        cutoff_df = pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Grade Start / F Max', 'Score'])
        cutoff_display_area.dataframe(cutoff_df.style.format({"Score": "{:.2f}"}))
    else:
        cutoff_display_area.warning("Calculate initial cutoffs first.")
# <<< End Header Add >>>

# --- Calculate and Display Grade Widths ---
with col_width_table:
    if st.session_state.active_cutoffs:
        st.header("Grade Widths")
        active_cutoff_map = {g:s for g,s in st.session_state.active_cutoffs.items() if g != 'F_Max'}
        sorted_grades = sorted(active_cutoff_map.keys(), key=lambda g: active_cutoff_map[g], reverse=True)
        widths = collections.OrderedDict(); max_score = 100.0
        for i, grade in enumerate(sorted_grades):
            start_score = active_cutoff_map[grade]
            upper_bound = max_score if i == 0 else active_cutoff_map[sorted_grades[i-1]]
            width = upper_bound - start_score
            width = 0.0 if np.isclose(width, 0) else width
            widths[grade] = f"{width:.2f} [{start_score:.2f} - {upper_bound:.2f})"
        width_df = pd.DataFrame(list(widths.items()), columns=['Grade', 'Width [Start - End)'])
        st.dataframe(width_df)


# --- Upload Section ---
st.header("Upload & Prepare Data")
# ... (Upload and Mapping logic remains the same as v1.7) ...
uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"], key="file_uploader_v10")
col_id, col_score, col_section, col_first, col_last = None, None, None, None, None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): df_upload = pd.read_csv(uploaded_file)
        else: df_upload = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.subheader("Map Columns"); cols = df_upload.columns.tolist(); cols_with_none = ["<Select Column>"] + cols
        col_first = st.selectbox("First Name Column (Optional)", options=cols_with_none, index=0, key='sel_first_v10')
        col_last = st.selectbox("Last Name Column (Optional)", options=cols_with_none, index=0, key='sel_last_v10')
        col_id = st.selectbox("Student ID Column (Optional)", options=cols_with_none, index=0, key='sel_id_orig_v11')
        col_score = st.selectbox("Score Column*", options=cols_with_none, index=0, key='sel_score_orig_v11')
        col_section = st.selectbox("Section Column*", options=cols_with_none, index=0, key='sel_section_orig_v11')
        st.caption("*Mandatory columns")
        if col_score != "<Select Column>" and col_section != "<Select Column>":
             df = df_upload.copy(); df.rename(columns={col_score: 'Score', col_section: 'Section'}, inplace=True)
             final_cols_to_keep = ['Score', 'Section']; name_cols_found = []
             if col_id != "<Select Column>": df.rename(columns={col_id: 'StudentID'}, inplace=True); final_cols_to_keep.append('StudentID')
             elif 'StudentID' not in df.columns: df['StudentID'] = 'Stud_' + df.index.astype(str); final_cols_to_keep.append('StudentID')
             if col_first != "<Select Column>": df.rename(columns={col_first: 'FirstName'}, inplace=True); final_cols_to_keep.append('FirstName'); name_cols_found.append('FirstName')
             if col_last != "<Select Column>": df.rename(columns={col_last: 'LastName'}, inplace=True); final_cols_to_keep.append('LastName'); name_cols_found.append('LastName')
             df = df[[col for col in final_cols_to_keep if col in df.columns]]
             df['Score'] = pd.to_numeric(df['Score'], errors='coerce'); initial_rows = len(df); df.dropna(subset=['Score'], inplace=True); removed_rows = initial_rows - len(df)
             if removed_rows > 0: st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")
             if df.empty: st.error("No valid score data remaining."); st.session_state.data_loaded = False; st.session_state.processed_df = None
             else:
                  df['Section'] = df['Section'].astype(str)
                  if 'StudentID' in df.columns: df['StudentID'] = df['StudentID'].astype(str)
                  if 'FirstName' in df.columns: df['FirstName'] = df['FirstName'].astype(str).fillna('')
                  if 'LastName' in df.columns: df['LastName'] = df['LastName'].astype(str).fillna('')
                  st.session_state.processed_df = df; st.session_state.data_loaded = True; st.success("Data loaded and columns mapped.")
                  st.subheader("Data Preview"); preview_cols = ['Score', 'Section']
                  if 'LastName' in df.columns: preview_cols.insert(0,'LastName')
                  if 'FirstName' in df.columns: preview_cols.insert(0,'FirstName')
                  if not name_cols_found and 'StudentID' in df.columns: preview_cols.insert(0,'StudentID')
                  st.dataframe(st.session_state.processed_df[[col for col in preview_cols if col in df.columns]].head())
                  st.session_state.df_graded = None; st.session_state.stats_results = None; st.session_state.upgraded_students = set()
        else: st.warning("Please select the Score and Section columns."); st.session_state.data_loaded = False; st.session_state.processed_df = None
    except Exception as e: st.error(f"Error loading or processing file: {e}"); st.session_state.data_loaded = False; st.session_state.processed_df = None

# Assign df_display if data is loaded
if st.session_state.data_loaded and st.session_state.processed_df is not None: df_display = st.session_state.processed_df

# --- Sections requiring data and active cutoffs ---
if st.session_state.data_loaded and df_display is not None and st.session_state.active_cutoffs is not None:

    # --- Manual Cutoff Adjustment ---
    st.header("Manual Cutoff Adjustment")
    st.markdown("Adjust the **Start Score** for each grade here and click 'Apply'. Review visualizations below.")
    manual_cutoffs_input = {}
    col_a_plus, col_a, col_b_plus, col_b = st.columns(4); col_c_plus, col_c, col_d_plus, col_d = st.columns(4)
    current_manual_vals = st.session_state.manual_override_values
    grade_keys_in_order = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    cols_map = {0:col_a_plus, 1:col_a, 2:col_b_plus, 3:col_b, 4:col_c_plus, 5:col_c, 6:col_d_plus, 7:col_d}
    for i, key in enumerate(grade_keys_in_order):
        with cols_map[i]:
             grade_label = key.replace('_Start', ' Start'); default_val = current_manual_vals.get(key, 0.0)
             manual_cutoffs_input[key] = st.number_input(grade_label, value=float(default_val), step=0.1, key=f'man_{key}_v11', format="%.2f")
    manual_cutoffs_input['F_Max'] = manual_cutoffs_input['D_Start']
    if st.button("Apply Manual Cutoffs & Recalculate"):
        scores_list = [manual_cutoffs_input[key] for key in grade_keys_in_order]
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)):
             st.error("Manual Start Scores must be in descending or equal order.")
        else:
             st.session_state.active_cutoffs = collections.OrderedDict(sorted(manual_cutoffs_input.items(), key=lambda item: item[1], reverse=True))
             st.session_state.manual_override_values = manual_cutoffs_input.copy()
             st.session_state.df_graded = None; st.session_state.stats_results = None
             # Do NOT reset upgraded students here - cutoff adjustments are separate
             st.success("Manual cutoffs applied. Recalculating results..."); st.rerun() # Use stable rerun

    # --- Perform grade assignment and stats calculation ---
    try:
        df_calc = df_display.copy()
        df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)
        if not df_calc['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
            st.session_state.stats_results = calculate_stats(df_calc, 'Letter_Grade', 'Section', GPA_SCALE)
            st.session_state.df_graded = df_calc
        else: st.error("Stats not calculated due to grade assignment errors."); st.session_state.stats_results = None; st.session_state.df_graded = None
    except Exception as e: st.error(f"Error during grade/stats calculation: {e}"); st.session_state.stats_results = None; st.session_state.df_graded = None

    # --- Visualization & Observation ---
    st.header("Visualization & Observation")
    active_cutoff_values_map = {grade: score for grade, score in st.session_state.active_cutoffs.items() if grade != 'F_Max'}
    active_cutoff_scores_asc = sorted(list(set(active_cutoff_values_map.values())))
    # Histogram unchanged
    st.subheader("Score Distribution with Active Cutoffs")
    hist_col, slider_col = st.columns([4, 1]);
    with slider_col: num_bins = st.slider("Histogram Bins", 5, 50, 25, key='hist_bins_v8')
    with hist_col:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5)); sns.histplot(df_display['Score'], kde=False, ax=ax_hist, bins=num_bins); sns.kdeplot(df_display['Score'], ax=ax_hist, color='orange', warn_singular=False)
        for cutoff in active_cutoff_scores_asc: ax_hist.axvline(cutoff, color='red', linestyle='--', linewidth=1)
        ax_hist.set_title("Score Distribution"); ax_hist.set_xlabel("Score"); st.pyplot(fig_hist); plt.close(fig_hist)
    # Individual Scores Plot unchanged
    st.subheader("Individual Scores with Active Cutoffs (Colored by Section)")
    if 'Section' in df_display.columns:
        unique_sections_plot = sorted(df_display['Section'].unique()); section_color_map_generated.clear(); color_cycle = itertools.cycle(SECTION_COLOR_PALETTE); plot_palette = {section: get_section_color_fixed(section) for section in unique_sections_plot}
        fig_strip, ax_strip = plt.subplots(figsize=(10, max(4, len(unique_sections_plot) * 0.5)))
        sns.stripplot(data=df_display, x='Score', y='Section', hue='Section', order=unique_sections_plot, hue_order=unique_sections_plot, ax=ax_strip, jitter=0.3, size=4, alpha=0.7, legend=True, palette=plot_palette)
        for cutoff in active_cutoff_scores_asc: ax_strip.axvline(cutoff, color='red', linestyle='--', linewidth=1)
        ax_strip.set_title("Individual Scores by Section"); ax_strip.set_xlabel("Score"); ax_strip.set_ylabel("Section"); ax_strip.legend(title='Section', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.9, 1]); st.pyplot(fig_strip); plt.close(fig_strip)
    else: st.warning("Section column needed for colored plot.")

    # --- Students Near Cutoffs (Below Only & Points to Upgrade) ---
    st.subheader(f"Students Below Cutoffs (within {points_near_cutoff} points)")
    students_near_df = pd.DataFrame() # Initialize empty dataframe
    if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns and not st.session_state.df_graded['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
        df_temp_graded = st.session_state.df_graded # Use the df with grades
        start_score_to_grade = {score: grade for grade, score in active_cutoff_values_map.items()}
        students_near_cutoff_list = [] # Reset list

        for boundary_score in active_cutoff_scores_asc:
            min_score_below = boundary_score - points_near_cutoff
            nearby_df = df_temp_graded[(df_temp_graded['Score'] >= min_score_below) & (df_temp_graded['Score'] < boundary_score)].copy()
            if not nearby_df.empty:
                 nearby_df['Target_Boundary'] = boundary_score; nearby_df['Points_to_Upgrade'] = boundary_score - nearby_df['Score']
                 target_grade = start_score_to_grade.get(boundary_score, "N/A"); nearby_df['Target_Grade'] = target_grade
                 students_near_cutoff_list.append(nearby_df)

        if students_near_cutoff_list:
            students_near_df = pd.concat(students_near_cutoff_list).sort_values(['Target_Boundary', 'Score'])
            name_cols_near = [];
            if 'LastName' in students_near_df.columns: name_cols_near.append('LastName')
            if 'FirstName' in students_near_df.columns: name_cols_near.append('FirstName')
            if not name_cols_near and 'StudentID' in students_near_df.columns: name_cols_near.append('StudentID')
            cols_near = name_cols_near + ['Score', 'Section', 'Letter_Grade', 'Target_Grade', 'Target_Boundary', 'Points_to_Upgrade']
            cols_near_exist = [col for col in cols_near if col in students_near_df.columns]
            st.dataframe(students_near_df[cols_near_exist].style.format({"Score": "{:.2f}", "Points_to_Upgrade": "{:.2f}", "Target_Boundary": "{:.2f}"}))
            st.caption("Tip: To upgrade a student, note their 'Target_Boundary' score. Adjust the 'Start Score' for the 'Target_Grade' in the section above, then click 'Apply'. Optionally, mark the student in the 'Manual Upgrades' section below for highlighting.")
        else: st.write(f"No students found scoring between (Cutoff - {points_near_cutoff:.1f}) and (Cutoff).")
    else: st.warning("Grade calculation needed or failed to show students near cutoffs.")

    # --- Manual Upgrade Section (Select ONLY from near cutoff list) ---
    st.header("Manual Upgrades (Highlighting Only)")
    st.markdown("Select students *from the table above* whose rows should be highlighted green in the final table.")
    # <<< Modified Selection Logic >>>
    if not students_near_df.empty: # Check if the near cutoff df was populated
        student_identifier_col = 'StudentID' if 'StudentID' in students_near_df.columns else None
        if student_identifier_col:
            eligible_students_df = students_near_df.drop_duplicates(subset=[student_identifier_col]) # Use only eligible students
            # Create display labels (Name or ID)
            if 'FirstName' in eligible_students_df.columns and 'LastName' in eligible_students_df.columns:
                 eligible_students_df['DisplayLabel'] = eligible_students_df['FirstName'] + " " + eligible_students_df['LastName'] + " (" + eligible_students_df[student_identifier_col] + ")"
            else:
                 eligible_students_df['DisplayLabel'] = eligible_students_df[student_identifier_col]

            student_options = sorted(eligible_students_df['DisplayLabel'].tolist())
            # Map display label back to the actual StudentID
            student_id_map = pd.Series(eligible_students_df[student_identifier_col].values, index=eligible_students_df['DisplayLabel']).to_dict()

            currently_selected_labels = [label for label, sid in student_id_map.items() if sid in st.session_state.upgraded_students]

            selected_labels = st.multiselect(
                "Highlight Students Below Cutoff as Upgraded:", options=student_options, default=currently_selected_labels, key="manual_upgrade_select_v3"
            )

            newly_selected_ids = set(student_id_map.get(label) for label in selected_labels if student_id_map.get(label) is not None)
            if newly_selected_ids != st.session_state.upgraded_students:
                 st.session_state.upgraded_students = newly_selected_ids
                 st.rerun() # Rerun to update final table highlighting
        else: st.warning("StudentID column not found, cannot provide manual upgrade selection.")
    else:
         st.markdown("_Load data and calculate grades to see students eligible for upgrade highlighting._")
    # <<< End Modified Selection Logic >>>

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
                all_grades_ordered = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F', 'Invalid Score', 'Error - Non-monotonic Cutoffs', 'Error - Bin/Label Mismatch']
                present_grades = sorted([g for g in all_grades_ordered if g in overall_series.index or g in section_dist_df.columns], key=lambda g: GPA_SCALE.get(g, -1), reverse=True)
                section_dist_df = section_dist_df.reindex(columns=present_grades, fill_value=0)
                overall_series = overall_series.reindex(present_grades, fill_value=0)
                combined_dist = pd.concat([overall_series, section_dist_df.T], axis=1).fillna(0)
                combined_dist.index.name = 'Grade'
                styler_dist = combined_dist.style.format("{:.1f}%").highlight_null(color='transparent') # Use color=
                st.markdown(styler_dist.to_html(escape=False, index=True), unsafe_allow_html=True)
            except Exception as e: st.error(f"Error displaying distribution table: {e}"); st.write("Raw Distributions:", results)
        elif "overall_dist" in results and not results['overall_dist'].empty: st.write("**Overall Distribution Only:**"); st.dataframe(results['overall_dist'].apply("{:.1%}".format))
        else: st.write("Distribution data could not be calculated.")

        overall_gpa_val = results.get('overall_gpa', np.nan)
        st.write(f"**Overall Avg GPA:** {overall_gpa_val:.2f}" if pd.notna(overall_gpa_val) else "**Overall Avg GPA:** N/A")
        st.write("---")

        st.subheader("Section GPA Comparison")
        col_gpa1, col_gpa2 = st.columns(2)
        with col_gpa1:
             st.write("**Per Section Avg GPA:**")
             if "section_stats" in results and not results['section_stats'].empty:
                   # <<< Apply formatting using .style before display >>>
                   st.dataframe(
                       results['section_stats'][['Section', 'Avg_GPA', 'Count']]
                       .style.format({"Avg_GPA": "{:.2f}"}, na_rep='N/A', precision=2) # Added precision, na_rep
                   )
             else: st.write("N/A")
             st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
             anova_p = results.get('anova_p_value');
             if anova_p is not None and anova_p < 0.05: st.warning("Significant difference in section GPAs detected.")
        with col_gpa2: # Boxplot unchanged
             try:
                 if st.session_state.df_graded is not None and 'GPA' in st.session_state.df_graded.columns and st.session_state.df_graded['GPA'].notna().any():
                     fig_box, ax_box = plt.subplots(); sorted_sections = sorted(st.session_state.df_graded['Section'].unique())
                     sns.boxplot(data=st.session_state.df_graded, x='Section', y='GPA', ax=ax_box, order=sorted_sections)
                     ax_box.set_title("GPA Distribution by Section"); plt.xticks(rotation=45, ha='right'); st.pyplot(fig_box); plt.close(fig_box)
                 else: st.warning("GPA data not available for boxplot.")
             except Exception as e: st.warning(f"Could not generate section GPA boxplot: {e}")

        # --- Failing Students Analysis ---
        st.subheader("Failing Students Analysis")
        # ... (Failing students logic unchanged) ...
        if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns:
            passing_score = st.session_state.active_cutoffs.get('D_Start', None)
            if passing_score is not None:
                failing_students = st.session_state.df_graded[st.session_state.df_graded['Letter_Grade'] == 'F'].copy()
                if not failing_students.empty:
                    failing_students['Points_Below_Pass'] = passing_score - failing_students['Score']
                    failing_students.sort_values('Points_Below_Pass', ascending=True, inplace=True)
                    st.write(f"Passing Score (D Start): {passing_score:.2f}")
                    fail_cols = ['Score', 'Section', 'Points_Below_Pass'];
                    if 'LastName' in failing_students.columns: fail_cols.insert(0, 'LastName')
                    if 'FirstName' in failing_students.columns: fail_cols.insert(0, 'FirstName')
                    if 'FirstName' not in fail_cols and 'LastName' not in fail_cols and 'StudentID' in failing_students.columns: fail_cols.insert(0,'StudentID')
                    cols_fail_exist = [col for col in fail_cols if col in failing_students.columns]
                    st.dataframe(failing_students[cols_fail_exist].style.format({"Score": "{:.2f}", "Points_Below_Pass": "{:.2f}"}))
                else: st.success("No students received an 'F' grade based on active cutoffs.")
            else: st.warning("Could not determine passing score ('D_Start') from active cutoffs.")
        else: st.warning("Final grades needed to analyze failing students.")


        # --- Final Assigned Grades Table ---
        st.subheader("Final Assigned Grades Table")
        if st.session_state.df_graded is not None:
             df_final = st.session_state.df_graded.copy()
             display_cols = ['FirstName', 'LastName', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
             display_cols_exist = [col for col in display_cols if col in df_final.columns]
             df_to_display = df_final[display_cols_exist]

             # <<< Switched to Column-wise Apply for Styling >>>
             section_color_map_generated.clear(); color_cycle = itertools.cycle(SECTION_COLOR_PALETTE)
             unique_sections_final = sorted(df_to_display['Section'].unique())
             for section in unique_sections_final: get_section_color_fixed(section) # Pre-assign colors

             styler = df_to_display.style
             # Apply grade gradient
             if 'Letter_Grade' in df_to_display.columns:
                  styler = styler.apply(lambda x: [f'background-color: {GRADE_GRADIENT.get(str(v), GRADE_GRADIENT["default"])}' for v in x], subset=['Letter_Grade'])
             # Apply section color
             if 'Section' in df_to_display.columns:
                  styler = styler.apply(lambda x: [f'background-color: {get_section_color_fixed(str(v))}' for v in x], subset=['Section'])
             # Apply upgrade highlight (needs access to whole row/df to check ID)
             if 'StudentID' in df_to_display.columns and st.session_state.upgraded_students:
                  styler = styler.apply(highlight_upgraded, upgraded_students_set=st.session_state.upgraded_students, axis=1,
                                       subset=['FirstName','LastName','StudentID'] if 'FirstName' in df_to_display.columns else ['StudentID']) # Apply row-wise for this specific style

             st.markdown(styler.format({"Score": "{:.2f}", "GPA": "{:.2f}"}).hide(axis="index").to_html(escape=False), unsafe_allow_html=True)
             # <<< End Styling Update >>>


             # --- Download Section ---
             st.subheader("Download Grades")
             # ... (Download logic unchanged) ...
             sections = ["All Sections"] + sorted(df_final['Section'].unique().tolist())
             selected_section = st.selectbox("Select section to download:", options=sections, key="download_section_select_v8")
             def convert_df_to_csv_orig(df_to_convert, section_filter):
                 if section_filter != "All Sections": df_filtered = df_to_convert[df_to_convert['Section'] == section_filter].copy()
                 else: df_filtered = df_to_convert.copy()
                 dl_cols = ['FirstName', 'LastName', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
                 dl_cols_exist = [col for col in dl_cols if col in df_filtered.columns]
                 if not dl_cols_exist: return None
                 return df_filtered[dl_cols_exist].to_csv(index=False).encode('utf-8')
             try:
                csv_data = convert_df_to_csv_orig(df_final, selected_section)
                if csv_data:
                    file_name = f"final_grades_{selected_section.replace(' ', '_')}.csv" if selected_section != "All Sections" else "final_grades_all.csv"
                    st.download_button(label=f"Download Grades for {selected_section}", data=csv_data, file_name=file_name, mime='text/csv', key=f"download_{selected_section}_v8")
             except Exception as e: st.error(f"Could not prepare download file: {e}")

        else: st.warning("Final grade assignments not yet calculated.")
    elif st.session_state.active_cutoffs:
         st.warning("Statistics could not be calculated. Check data, grade assignment, and cutoffs.")


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Iterative Grading Tool v1.9")
