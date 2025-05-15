# Full code block with fixes for new issues (cutoff copy removal, file.id error)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For better color palettes
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
    cutoffs['F_Max'] = cutoffs['D_Start']
    return cutoffs

def assign_letter_grades_from_starts(scores, start_cutoffs):
    """
    Assigns letter grades based on start scores (lower bound inclusive).
    Handles duplicate start scores by prioritizing the higher grade.
    """
    expected_grade_order_keys = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']

    present_grade_keys = [k for k in expected_grade_order_keys if k in start_cutoffs]
    if len(present_grade_keys) > 1:
        ordered_scores_for_check = [start_cutoffs[k] for k in present_grade_keys]
        if not all(ordered_scores_for_check[i] >= ordered_scores_for_check[i+1] for i in range(len(ordered_scores_for_check)-1)):
            st.error("Input Error: Grade boundary start scores (from A+ down to D) must be monotonically decreasing or equal. Please check manual cutoff inputs.")
            return pd.Series(['Error - Non-monotonic Input Cutoffs'] * len(scores), index=scores.index, dtype='object')

    score_to_highest_grade_label_map = collections.OrderedDict()
    for grade_key_with_suffix, score_val in start_cutoffs.items():
        if grade_key_with_suffix == 'F_Max':
            continue
        grade_label = grade_key_with_suffix.replace('_Start', '')
        if score_val not in score_to_highest_grade_label_map:
            score_to_highest_grade_label_map[score_val] = grade_label

    if not score_to_highest_grade_label_map:
        st.error("No valid grade boundaries found for assignment after processing.")
        return pd.Series(['Error - No Valid Cutoffs For Assignment'] * len(scores), index=scores.index, dtype='object')

    unique_ascending_boundary_scores = sorted(list(score_to_highest_grade_label_map.keys()))

    if not unique_ascending_boundary_scores:
         st.error("Internal Error: No unique boundary scores for binning.")
         return pd.Series(['Error - Binning Issue'] * len(scores), index=scores.index, dtype='object')

    labels_for_cut = ['F'] + [score_to_highest_grade_label_map[s] for s in unique_ascending_boundary_scores]
    bins = [-np.inf] + unique_ascending_boundary_scores + [np.inf]

    if len(labels_for_cut) != len(bins) - 1:
        st.error(f"CRITICAL INTERNAL ERROR: Bin/Label mismatch. Bins ({len(bins)}): {bins}, Labels ({len(labels_for_cut)}): {labels_for_cut}.")
        return pd.Series(['Error - Internal Bin/Label Mismatch'] * len(scores), index=scores.index, dtype='object')

    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels_for_cut, right=False, ordered=False)
    grades = grades.astype('object').fillna('Invalid Score')
    return grades

def calculate_stats(df, grade_col, section_col, gpa_map):
    if grade_col not in df.columns or df[grade_col].astype(str).str.contains('Error', na=False).any():
        st.warning("Cannot calculate stats due to errors in grade assignment.")
        return {"error": "Grade assignment failed."}

    df_copy = df.copy()
    df_copy['GPA'] = df_copy[grade_col].map(gpa_map)
    df_copy['GPA'] = pd.to_numeric(df_copy['GPA'], errors='coerce')

    if df_copy['GPA'].isnull().all() and df_copy[grade_col].notna().any() and not df_copy[grade_col].isin(['Invalid Score']).all():
        st.warning("GPA calculation resulted in all non-numeric values. Check GPA_SCALE and grade assignments.")

    overall_dist_percent = df_copy[grade_col].value_counts(normalize=True).sort_index()
    overall_dist_count = df_copy[grade_col].value_counts(normalize=False).sort_index()
    overall_gpa = df_copy['GPA'].mean()

    df_copy[section_col] = df_copy[section_col].astype(str)

    section_gpa_means = df_copy.groupby(section_col)['GPA'].mean()
    section_stats_agg = df_copy.groupby(section_col).agg(
        Count=('GPA', 'size'),
        Valid_GPA_Count=('GPA', 'count')
    ).reset_index()
    section_stats = pd.merge(section_stats_agg, section_gpa_means.rename('Avg_GPA'), on=section_col, how='left')

    section_dist_percent = pd.DataFrame()
    section_dist_count = pd.DataFrame()
    try:
        section_dist_percent = df_copy.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)
        section_dist_count = df_copy.groupby(section_col)[grade_col].value_counts(normalize=False).unstack(fill_value=0)
    except Exception as e:
        st.warning(f"Could not generate section-wise grade distributions: {e}")

    anova_result = "ANOVA not applicable."
    anova_p_value = None
    section_groups_for_anova = [
        group['GPA'].dropna().values for _, group in df_copy.groupby(section_col)
        if group['GPA'].notna().sum() > 1
    ]

    if len(section_groups_for_anova) > 1:
        try:
            f_val, p_val = stats.f_oneway(*section_groups_for_anova)
            anova_result = f"ANOVA F={f_val:.2f}, p={p_val:.3f}"
            anova_p_value = p_val
        except ValueError as e:
            anova_result = f"ANOVA Error: {e}"
        except Exception as e:
            anova_result = f"ANOVA failed unexpectedly: {e}"

    return {
        "overall_dist_percent": overall_dist_percent,
        "overall_dist_count": overall_dist_count,
        "overall_gpa": overall_gpa,
        "section_stats": section_stats,
        "section_dist_percent": section_dist_percent,
        "section_dist_count": section_dist_count,
        "anova_result": anova_result, "anova_p_value": anova_p_value,
        "error": None
    }

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

GRADE_GRADIENT = {
    'A+': '#FFFFFF', 'A': '#F2F2F2', 'B+': '#E6E6E6', 'B': '#D9D9D9',
    'C+': '#CCCCCC', 'C': '#BDBDBD', 'D+': '#B0B0B0', 'D': '#A3A3A3',
    'F': '#969696', 'default': '#F0F0F0'
}
SECTION_COLOR_PALETTE = [cm.get_cmap('tab20')(i) for i in range(20)]
section_color_map_generated = {}
color_cycle_obj = itertools.cycle(SECTION_COLOR_PALETTE)

def get_section_color_fixed(section_name):
    global color_cycle_obj
    str_section = str(section_name)
    if str_section not in section_color_map_generated:
        rgba_color = next(color_cycle_obj)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
        section_color_map_generated[str_section] = hex_color
    return section_color_map_generated[str_section]

def reset_section_color_cycle():
    global section_color_map_generated, color_cycle_obj
    section_color_map_generated.clear()
    color_cycle_obj = itertools.cycle(SECTION_COLOR_PALETTE)

def highlight_upgraded(row, upgraded_students_set, id_col='StudentID', first_col_name='FirstName', last_col_name='LastName'):
    highlight_style = 'background-color: #90EE90 !important;'
    default_style = ''
    output_styles = pd.Series(default_style, index=row.index)
    current_row_student_id = row.get(id_col)

    if current_row_student_id and str(current_row_student_id) in upgraded_students_set:
        if first_col_name in row.index: output_styles[first_col_name] = highlight_style
        if last_col_name in row.index: output_styles[last_col_name] = highlight_style
        name_cols_actually_highlighted = False
        if first_col_name in row.index and output_styles[first_col_name] == highlight_style: name_cols_actually_highlighted = True
        if last_col_name in row.index and output_styles[last_col_name] == highlight_style: name_cols_actually_highlighted = True
        if id_col in row.index and not name_cols_actually_highlighted: output_styles[id_col] = highlight_style
    return output_styles

# ============================================
# Streamlit App Layout
# ============================================
st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant v1.13") # Version update
st.info("**Workflow:**\n"
        "1. Set **Initial Parameters** in the sidebar.\n"
        "2. Click **'Calculate Initial Cutoffs'**.\n"
        "3. Upload **Score File** & **Map Columns** (selections now persist better!).\n"
        "4. Review **Cutoffs**, **Visualizations**, **Students Near Cutoffs** (adjust range using input below Visualizations).\n"
        "5. *Optionally:* Manually Adjust **Start Scores** & **'Apply Manual Cutoffs'**.\n"
        "6. *Optionally:* Select students for *highlighting* in **'Manual Upgrades'** section.\n"
        "7. Review **Final Results** (Distributions now show Count & Percent).\n"
        "8. **Download** grades.")

# --- Session State Initialization ---
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {'col_first': None, 'col_last': None, 'col_id': None, 'col_score': None, 'col_section': None} # Removed file_id
if 'initial_cutoffs' not in st.session_state: st.session_state.initial_cutoffs = None
if 'active_cutoffs' not in st.session_state: st.session_state.active_cutoffs = None
if 'df_graded' not in st.session_state: st.session_state.df_graded = None
if 'stats_results' not in st.session_state: st.session_state.stats_results = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'upgraded_students' not in st.session_state: st.session_state.upgraded_students = set()
if 'points_near_cutoff_active' not in st.session_state: st.session_state.points_near_cutoff_active = 1.5

# --- Sidebar ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score (>= score)", value=95.0, step=0.1, format="%.2f", help="Min score for A+.")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap (points)", value=5.0, step=0.1, min_value=0.1, format="%.2f", help="Initial gap between grade starts.")

def update_active_points_from_sidebar():
    st.session_state.points_near_cutoff_active = st.session_state.points_near_num_v5_sidebar_key

st.sidebar.number_input(
    "Initial: Students Below Cutoff (X pts)", min_value=0.1, max_value=10.0,
    value=st.session_state.points_near_cutoff_active,
    step=0.1, format="%.1f",
    key='points_near_num_v5_sidebar_key',
    on_change=update_active_points_from_sidebar,
    help="Initial range. Can be overridden in the main panel under 'Visualizations'."
)

if st.sidebar.button("Calculate Initial Cutoffs"):
    st.session_state.initial_cutoffs = calculate_initial_cutoffs_original(a_plus_start_score, uniform_grade_gap)
    st.session_state.active_cutoffs = st.session_state.initial_cutoffs
    manual_vals = {grade_key: score for grade_key, score in st.session_state.initial_cutoffs.items()}
    all_grade_input_keys = [f'{g}_Start' for g in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']]
    for key in all_grade_input_keys:
        if key not in manual_vals: manual_vals[key] = 0.0
    if 'F_Max' not in manual_vals: manual_vals['F_Max'] = manual_vals.get('D_Start', 0.0)
    st.session_state.manual_override_values = manual_vals
    st.session_state.df_graded = None
    st.session_state.stats_results = None
    st.session_state.upgraded_students = set()
    st.sidebar.success("Initial cutoffs calculated.")
    if st.session_state.data_loaded: st.rerun()

# --- Main Area ---
col_cutoff_table, col_width_table = st.columns([1, 2]) # Adjusted column ratio
with col_cutoff_table:
    st.header("Active Cutoffs")
    cutoff_display_area = st.empty()
    if st.session_state.active_cutoffs:
        cutoff_df_display = pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Grade Boundary', 'Score'])
        cutoff_display_area.dataframe(cutoff_df_display.style.format({"Score": "{:.2f}"}))
        # FIX 1 (Issue from previous round): Removed manual cutoff copy feature
        # st.subheader("Copyable Active Cutoffs") ... and st.code block removed
    else:
        cutoff_display_area.warning("Calculate initial cutoffs or apply manual cutoffs first.")

with col_width_table:
    if st.session_state.active_cutoffs:
        st.header("Grade Widths")
        grades_in_desc_order = [key for key in st.session_state.active_cutoffs.keys() if key != 'F_Max']
        widths = collections.OrderedDict()
        max_score_cap = 100.0
        for i, grade_key in enumerate(grades_in_desc_order):
            start_score = st.session_state.active_cutoffs[grade_key]
            upper_bound = max_score_cap if i == 0 else st.session_state.active_cutoffs[grades_in_desc_order[i-1]]
            width_val = upper_bound - start_score
            width_val = 0.0 if np.isclose(width_val, 0) or width_val < 0 else width_val
            grade_label = grade_key.replace('_Start', '')
            widths[grade_label] = f"{width_val:.2f} [{start_score:.2f} – <{upper_bound:.2f})"
        if widths:
            width_df_display = pd.DataFrame(list(widths.items()), columns=['Grade', 'Width [Start – End)'])
            st.dataframe(width_df_display)
        else: st.write("Not enough grade boundaries to calculate widths.")

# --- Upload Section ---
st.header("Upload & Prepare Data")

def new_file_uploaded_callback():
    st.session_state.column_mappings = {'col_first': None, 'col_last': None, 'col_id': None, 'col_score': None, 'col_section': None} # Reset mappings
    st.session_state.data_loaded = False
    st.session_state.processed_df = None
    st.session_state.df_graded = None
    st.session_state.stats_results = None
    st.session_state.upgraded_students = set()

uploaded_file = st.file_uploader("Upload course scores (CSV or Excel)", type=["csv", "xlsx"], key="file_uploader_v14_main", on_change=new_file_uploaded_callback)

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        st.subheader("Map Columns")
        cols_from_file = df_upload.columns.tolist()
        cols_with_none_option = ["<Select Column>"] + cols_from_file

        current_mappings = st.session_state.column_mappings
        def get_col_index(col_name_to_find, all_cols_list_with_none):
            if col_name_to_find and col_name_to_find in all_cols_list_with_none:
                return all_cols_list_with_none.index(col_name_to_find)
            return 0

        key_suffix = "_fix_v14"
        col_first_selection = st.selectbox("First Name Column (Optional)", options=cols_with_none_option, index=get_col_index(current_mappings['col_first'], cols_with_none_option), key='sel_first'+key_suffix)
        col_last_selection = st.selectbox("Last Name Column (Optional)", options=cols_with_none_option, index=get_col_index(current_mappings['col_last'], cols_with_none_option), key='sel_last'+key_suffix)
        col_id_selection = st.selectbox("Student ID Column (Optional)", options=cols_with_none_option, index=get_col_index(current_mappings['col_id'], cols_with_none_option), key='sel_id'+key_suffix)
        col_score_selection = st.selectbox("Score Column*", options=cols_with_none_option, index=get_col_index(current_mappings['col_score'], cols_with_none_option), key='sel_score'+key_suffix)
        col_section_selection = st.selectbox("Section Column*", options=cols_with_none_option, index=get_col_index(current_mappings['col_section'], cols_with_none_option), key='sel_section'+key_suffix)
        st.caption("*Mandatory columns")

        if col_score_selection != "<Select Column>" and col_section_selection != "<Select Column>":
            st.session_state.column_mappings['col_first'] = col_first_selection if col_first_selection != "<Select Column>" else None
            st.session_state.column_mappings['col_last'] = col_last_selection if col_last_selection != "<Select Column>" else None
            st.session_state.column_mappings['col_id'] = col_id_selection if col_id_selection != "<Select Column>" else None
            st.session_state.column_mappings['col_score'] = col_score_selection
            st.session_state.column_mappings['col_section'] = col_section_selection
            # FIX 2 (Issue from previous round): Removed uploaded_file.id assignment
            # st.session_state.column_mappings['file_id'] = uploaded_file.id # REMOVED

            df = df_upload.copy()
            df.rename(columns={col_score_selection: 'Score', col_section_selection: 'Section'}, inplace=True)

            final_cols_to_keep = ['Score', 'Section']
            if col_id_selection != "<Select Column>":
                df.rename(columns={col_id_selection: 'StudentID'}, inplace=True); final_cols_to_keep.append('StudentID')
            elif 'StudentID' not in df.columns:
                df['StudentID'] = 'Stud_' + df.index.astype(str); final_cols_to_keep.append('StudentID')

            if col_first_selection != "<Select Column>":
                df.rename(columns={col_first_selection: 'FirstName'}, inplace=True); final_cols_to_keep.append('FirstName')
            if col_last_selection != "<Select Column>":
                df.rename(columns={col_last_selection: 'LastName'}, inplace=True); final_cols_to_keep.append('LastName')

            df = df[[col for col in final_cols_to_keep if col in df.columns]]
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
            initial_rows = len(df)
            df.dropna(subset=['Score'], inplace=True)
            removed_rows = initial_rows - len(df)
            if removed_rows > 0: st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")

            if df.empty:
                st.error("No valid score data remaining."); st.session_state.data_loaded = False; st.session_state.processed_df = None
            else:
                df['Section'] = df['Section'].astype(str)
                if 'StudentID' in df.columns: df['StudentID'] = df['StudentID'].astype(str)
                if 'FirstName' in df.columns: df['FirstName'] = df['FirstName'].astype(str).fillna('')
                if 'LastName' in df.columns: df['LastName'] = df['LastName'].astype(str).fillna('')

                st.session_state.processed_df = df; st.session_state.data_loaded = True;
                st.success("Data loaded and columns mapped.")
                reset_section_color_cycle()

                st.subheader("Data Preview")
                preview_cols_ordered = []
                if 'FirstName' in df.columns: preview_cols_ordered.append('FirstName')
                if 'LastName' in df.columns: preview_cols_ordered.append('LastName')
                if not preview_cols_ordered and 'StudentID' in df.columns: preview_cols_ordered.append('StudentID')
                for col_m in ['Score', 'Section']:
                    if col_m in df.columns and col_m not in preview_cols_ordered: preview_cols_ordered.append(col_m)
                final_preview_cols_to_show = [col for col in preview_cols_ordered if col in df.columns]
                if final_preview_cols_to_show: st.dataframe(df[final_preview_cols_to_show].head())
                else: st.warning("Could not determine columns for data preview.")

                st.session_state.df_graded = None; st.session_state.stats_results = None;
        else:
            st.warning("Please select the Score and Section columns."); st.session_state.data_loaded = False; st.session_state.processed_df = None
    except Exception as e: # Catch potential errors during file processing
        st.error(f"Error loading or processing file: {e}")
        st.session_state.data_loaded = False
        st.session_state.processed_df = None


df_display = st.session_state.processed_df if st.session_state.data_loaded else None

# --- Sections requiring data and active cutoffs ---
if st.session_state.data_loaded and df_display is not None and st.session_state.active_cutoffs is not None:

    # --- Manual Cutoff Adjustment ---
    st.header("Manual Cutoff Adjustment")
    st.markdown("Adjust **Start Score** for each grade. Scores must be monotonically decreasing (A+ ≥ A ≥ ... ≥ D).")

    manual_cutoffs_input = {}
    grade_keys_in_order = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    current_manual_vals = st.session_state.manual_override_values
    num_grades = len(grade_keys_in_order)
    cols_per_row = 4
    num_rows = (num_grades + cols_per_row - 1) // cols_per_row 
    
    for r in range(num_rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < num_grades:
                key = grade_keys_in_order[idx]
                with cols[c]:
                    grade_label = key.replace('_Start', ' Start')
                    default_val = float(current_manual_vals.get(key, st.session_state.active_cutoffs.get(key, 0.0)))
                    manual_cutoffs_input[key] = st.number_input(grade_label, value=default_val, step=0.1, key=f'man_{key}_v15', format="%.2f") # Incremented key
    manual_cutoffs_input['F_Max'] = manual_cutoffs_input.get('D_Start', 0.0)

    if st.button("Apply Manual Cutoffs & Recalculate"):
        scores_list = [manual_cutoffs_input[key] for key in grade_keys_in_order if key in manual_cutoffs_input]
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)):
            st.error("Manual Start Scores for grades (A+ down to D) must be monotonically decreasing or equal.")
        else:
            new_active_cutoffs = collections.OrderedDict()
            all_grade_keys_ordered = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start', 'F_Max']
            for key in all_grade_keys_ordered:
                if key in manual_cutoffs_input: new_active_cutoffs[key] = manual_cutoffs_input[key]
            st.session_state.active_cutoffs = new_active_cutoffs
            st.session_state.manual_override_values = manual_cutoffs_input.copy()
            st.session_state.df_graded = None; st.session_state.stats_results = None
            st.success("Manual cutoffs applied. Recalculating results..."); st.rerun()

    if st.session_state.df_graded is None:
        try:
            df_calc = df_display.copy()
            df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)
            if not df_calc['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
                st.session_state.stats_results = calculate_stats(df_calc, 'Letter_Grade', 'Section', GPA_SCALE)
                st.session_state.df_graded = df_calc
            else:
                st.error("Stats not calculated due to errors in grade assignment. Check cutoffs and data."); st.session_state.stats_results = None; st.session_state.df_graded = None
        except Exception as e:
            st.error(f"Error during grade/stats calculation: {e}"); st.session_state.stats_results = None; st.session_state.df_graded = None

    st.header("Visualization & Observation")
    st.session_state.points_near_cutoff_active = st.number_input(
        "Range for 'Students Below Cutoffs' (points)",
        min_value=0.1, max_value=10.0,
        value=st.session_state.points_near_cutoff_active,
        step=0.1, format="%.1f",
        key='points_near_cutoff_override_main_v3', # Incremented key
        help="Dynamically change the range below active cutoffs shown in the table further down."
    )
    active_points_near_cutoff_for_display = st.session_state.points_near_cutoff_active

    active_cutoff_plot_map = { grade.replace("_Start", ""): score for grade, score in st.session_state.active_cutoffs.items() if grade != 'F_Max'}
    unique_cutoff_scores_for_plot = sorted(list(set(active_cutoff_plot_map.values())))

    st.subheader("Score Distribution with Active Cutoffs")
    hist_col, slider_col = st.columns([4, 1])
    with slider_col: num_bins = st.slider("Histogram Bins", 5, 50, 25, key='hist_bins_v12') # Incremented key
    with hist_col:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(df_display['Score'], kde=False, ax=ax_hist, bins=num_bins, stat="density")
        sns.kdeplot(df_display['Score'], ax=ax_hist, color='orange', warn_singular=False)
        for cutoff_val in unique_cutoff_scores_for_plot: ax_hist.axvline(cutoff_val, color='red', linestyle='--', linewidth=1)
        ax_hist.set_title("Score Distribution with Grade Cutoffs"); ax_hist.set_xlabel("Score"); ax_hist.set_ylabel("Density / Frequency")
        st.pyplot(fig_hist); plt.close(fig_hist)

    st.subheader("Individual Scores with Active Cutoffs (Colored by Section)")
    if 'Section' in df_display.columns:
        reset_section_color_cycle()
        unique_sections_plot = sorted(df_display['Section'].unique())
        plot_palette_for_strip = {section: get_section_color_fixed(section) for section in unique_sections_plot}
        fig_strip, ax_strip = plt.subplots(figsize=(10, max(4, len(unique_sections_plot) * 0.6)))
        sns.stripplot(data=df_display, x='Score', y='Section', hue='Section', order=unique_sections_plot, hue_order=unique_sections_plot,
                      ax=ax_strip, jitter=0.3, size=5, alpha=0.8, legend="auto", palette=plot_palette_for_strip)
        for cutoff_val in unique_cutoff_scores_for_plot: ax_strip.axvline(cutoff_val, color='red', linestyle='--', linewidth=1)
        ax_strip.set_title("Individual Scores by Section with Grade Cutoffs"); ax_strip.set_xlabel("Score"); ax_strip.set_ylabel("Section")
        if len(unique_sections_plot) > 1 and len(unique_sections_plot) <= 20 :
             ax_strip.legend(title='Section', bbox_to_anchor=(1.02, 1), loc='upper left', ncol= 1 if len(unique_sections_plot) <=10 else 2)
        else:
            if ax_strip.get_legend() is not None: ax_strip.get_legend().remove()
        plt.tight_layout(rect=[0, 0, 0.85 if len(unique_sections_plot) > 1 and len(unique_sections_plot) <=20 else 1, 1])
        st.pyplot(fig_strip); plt.close(fig_strip)
    else: st.warning("Section column not found, cannot generate section-colored plot.")

    st.subheader(f"Students Below Cutoffs (within {active_points_near_cutoff_for_display:.1f} points)")
    students_near_df_local = pd.DataFrame()
    df_for_near_cutoff_check = st.session_state.df_graded

    if df_for_near_cutoff_check is not None and 'Letter_Grade' in df_for_near_cutoff_check.columns and \
       not df_for_near_cutoff_check['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
        score_to_definitive_grade_label_map = collections.OrderedDict()
        for grade_key_suffix, score_value in st.session_state.active_cutoffs.items():
            if grade_key_suffix == 'F_Max': continue
            grade_lbl = grade_key_suffix.replace('_Start', '')
            if score_value not in score_to_definitive_grade_label_map: score_to_definitive_grade_label_map[score_value] = grade_lbl

        students_near_cutoff_list = []
        for boundary_score_val in unique_cutoff_scores_for_plot:
            min_score_for_nearby = boundary_score_val - active_points_near_cutoff_for_display
            nearby_students_df = df_for_near_cutoff_check[
                (df_for_near_cutoff_check['Score'] >= min_score_for_nearby) & (df_for_near_cutoff_check['Score'] < boundary_score_val)].copy()
            if not nearby_students_df.empty:
                nearby_students_df['Target_Boundary'] = boundary_score_val
                nearby_students_df['Points_to_Upgrade'] = boundary_score_val - nearby_students_df['Score']
                target_grade_label = score_to_definitive_grade_label_map.get(boundary_score_val, "N/A")
                nearby_students_df['Target_Grade'] = target_grade_label
                students_near_cutoff_list.append(nearby_students_df)

        if students_near_cutoff_list:
            students_near_df_local = pd.concat(students_near_cutoff_list).sort_values(by=['Target_Boundary', 'Score'])
            name_cols_near = []
            if 'FirstName' in students_near_df_local.columns: name_cols_near.append('FirstName')
            if 'LastName' in students_near_df_local.columns: name_cols_near.append('LastName')
            if not name_cols_near and 'StudentID' in students_near_df_local.columns: name_cols_near.append('StudentID')
            cols_near_display = name_cols_near + ['Score', 'Section', 'Letter_Grade', 'Target_Grade', 'Target_Boundary', 'Points_to_Upgrade']
            cols_near_display_exist = [col for col in cols_near_display if col in students_near_df_local.columns]
            st.dataframe(students_near_df_local[cols_near_display_exist].style.format({"Score": "{:.2f}", "Points_to_Upgrade": "{:.2f}", "Target_Boundary": "{:.2f}"}))
        else: st.write(f"No students found scoring between (Cutoff - {active_points_near_cutoff_for_display:.1f}) and (Cutoff).")
    else: st.warning("Grade calculation needed or has errors. Cannot show students near cutoffs.")

    st.header("Manual Upgrades (Highlighting Only)")
    st.markdown("Select students *from the 'Students Below Cutoffs' list* to highlight their rows green in the final table. This does NOT change their grade calculation.")
    if not students_near_df_local.empty:
        student_identifier_col_name = 'StudentID' if 'StudentID' in students_near_df_local.columns else None
        if student_identifier_col_name:
            eligible_students_for_highlight_df = students_near_df_local.drop_duplicates(subset=[student_identifier_col_name]).copy()
            if 'FirstName' in eligible_students_for_highlight_df.columns and 'LastName' in eligible_students_for_highlight_df.columns:
                eligible_students_for_highlight_df.loc[:, 'DisplayLabel'] = eligible_students_for_highlight_df['FirstName'] + " " + eligible_students_for_highlight_df['LastName'] + " (" + eligible_students_for_highlight_df[student_identifier_col_name] + ")"
            else: eligible_students_for_highlight_df.loc[:, 'DisplayLabel'] = eligible_students_for_highlight_df[student_identifier_col_name]
            student_options_for_multiselect = sorted(eligible_students_for_highlight_df['DisplayLabel'].tolist())
            student_id_map_for_multiselect = pd.Series(eligible_students_for_highlight_df[student_identifier_col_name].values, index=eligible_students_for_highlight_df['DisplayLabel']).to_dict()
            currently_selected_student_labels = [label for label, student_id_val in student_id_map_for_multiselect.items() if student_id_val in st.session_state.upgraded_students]
            selected_display_labels = st.multiselect("Highlight Students Below Cutoff as Upgraded:", options=student_options_for_multiselect, default=currently_selected_student_labels, key="manual_upgrade_select_v7") # Incremented key
            newly_selected_student_ids = set(student_id_map_for_multiselect.get(label) for label in selected_display_labels if student_id_map_for_multiselect.get(label) is not None)
            if st.button("Update Upgrade Highlighting"):
                if newly_selected_student_ids != st.session_state.upgraded_students:
                    st.session_state.upgraded_students = newly_selected_student_ids
                    st.success("Upgrade highlighting selection updated."); st.rerun()
                else: st.info("Highlighting selection unchanged.")
        else: st.warning("StudentID column not found, cannot provide manual upgrade selection.")
    else: st.markdown("_Students potentially needing upgrades will appear here once grades are calculated and such students exist._")

    st.header("Final Results (Based on Active Cutoffs)")
    if st.session_state.stats_results and st.session_state.stats_results.get("error") is None:
        results = st.session_state.stats_results
        st.subheader("Grade Distributions (Count & Percentage)")
        if "overall_dist_percent" in results and not results['overall_dist_percent'].empty:
            try:
                overall_series_pct = results['overall_dist_percent']
                overall_series_cnt = results['overall_dist_count']
                section_dist_df_pct = results['section_dist_percent']
                section_dist_df_cnt = results['section_dist_count']

                all_grades_ordered_list = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F'] + [g for g in GPA_SCALE.keys() if 'Error' in g or 'Invalid' in g]
                present_grades_in_data = overall_series_cnt.index.union(section_dist_df_cnt.columns if not section_dist_df_cnt.empty else [])
                grades_for_display_sorted = [g for g in all_grades_ordered_list if g in present_grades_in_data]
                grades_for_display_sorted.extend([g for g in present_grades_in_data if g not in grades_for_display_sorted])

                combined_display_list = []
                for grade_val in grades_for_display_sorted:
                    overall_cnt_val = overall_series_cnt.get(grade_val, 0)
                    overall_pct_val = overall_series_pct.get(grade_val, 0) * 100
                    row_data = {'Grade': grade_val, 'Overall': f"{overall_cnt_val} ({overall_pct_val:.1f}%)"}
                    if not section_dist_df_cnt.empty:
                        for section_name in section_dist_df_cnt.index:
                            sec_cnt_val = section_dist_df_cnt.loc[section_name, grade_val] if grade_val in section_dist_df_cnt.columns else 0
                            sec_pct_val = (section_dist_df_pct.loc[section_name, grade_val] if grade_val in section_dist_df_pct.columns else 0) * 100
                            row_data[str(section_name)] = f"{sec_cnt_val} ({sec_pct_val:.1f}%)"
                    combined_display_list.append(row_data)

                if combined_display_list:
                    combined_dist_display_df = pd.DataFrame(combined_display_list).set_index('Grade')
                    st.markdown(combined_dist_display_df.to_html(escape=False, index=True), unsafe_allow_html=True)
                else: st.write("No grade distribution data to display.")
            except Exception as e: st.error(f"Error displaying combined count/distribution table: {e}")
        else: st.write("Distribution data could not be calculated or is empty.")

        overall_gpa_val = results.get('overall_gpa', np.nan)
        st.write(f"**Overall Avg GPA:** {overall_gpa_val:.2f}" if pd.notna(overall_gpa_val) else "**Overall Avg GPA:** N/A"); st.write("---")

        st.subheader("Section GPA Comparison")
        col_gpa_table, col_gpa_plot = st.columns(2)
        with col_gpa_table:
            st.write("**Per Section Avg GPA:**")
            if "section_stats" in results and not results['section_stats'].empty:
                st.dataframe(results['section_stats'][['Section', 'Avg_GPA', 'Valid_GPA_Count', 'Count']].rename(columns={'Valid_GPA_Count':'Graded Students', 'Count':'Total Students'}).style.format({"Avg_GPA": "{:.2f}"}, na_rep='N/A'))
            else: st.write("Section GPA statistics not available.")

            st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
            anova_p_val_interp = results.get('anova_p_value')
            alpha_sig = 0.05
            if anova_p_val_interp is not None:
                if anova_p_val_interp < alpha_sig: st.markdown(f"<span style='color:orange;'>Significant difference in section GPAs detected (p={anova_p_val_interp:.3f}). This suggests average GPA is not the same across all sections.</span>", unsafe_allow_html=True)
                else: st.markdown(f"No significant difference detected in section GPAs (p={anova_p_val_interp:.3f}). Observed differences may be due to chance.", unsafe_allow_html=True)

        with col_gpa_plot:
            if st.session_state.df_graded is not None and 'GPA' in st.session_state.df_graded.columns and st.session_state.df_graded['GPA'].notna().any():
                try:
                    fig_box, ax_box = plt.subplots(); sorted_sections_for_plot = sorted(st.session_state.df_graded['Section'].unique())
                    sns.boxplot(data=st.session_state.df_graded, x='Section', y='GPA', ax=ax_box, order=sorted_sections_for_plot)
                    ax_box.set_title("GPA Distribution by Section"); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig_box); plt.close(fig_box)
                except Exception as e: st.warning(f"Could not generate section GPA boxplot: {e}")
            else: st.warning("GPA data (e.g., all NaN or calculation errors) not available for boxplot. Check grade assignments and GPA scale.")

        st.subheader("Failing Students Analysis")
        if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns:
            passing_score_boundary = st.session_state.active_cutoffs.get('D_Start', None)
            if passing_score_boundary is not None:
                failing_students_df = st.session_state.df_graded[st.session_state.df_graded['Letter_Grade'] == 'F'].copy()
                if not failing_students_df.empty:
                    failing_students_df['Points_Below_Pass'] = passing_score_boundary - failing_students_df['Score']
                    failing_students_df.sort_values('Points_Below_Pass', ascending=True, inplace=True)
                    st.write(f"Passing Score (Minimum for D): {passing_score_boundary:.2f}"); fail_cols_display = []
                    if 'FirstName' in failing_students_df.columns: fail_cols_display.append('FirstName')
                    if 'LastName' in failing_students_df.columns: fail_cols_display.append('LastName')
                    if not fail_cols_display and 'StudentID' in failing_students_df.columns: fail_cols_display.append('StudentID')
                    fail_cols_display.extend(['Score', 'Section', 'Points_Below_Pass'])
                    cols_fail_exist_display = [col for col in fail_cols_display if col in failing_students_df.columns]
                    st.dataframe(failing_students_df[cols_fail_exist_display].style.format({"Score": "{:.2f}", "Points_Below_Pass": "{:.2f}"}))
                else: st.success("No students received an 'F' grade based on active cutoffs.")
            else: st.warning("Could not determine D_Start from cutoffs to analyze failing students.")
        else: st.warning("Final grades needed or contain errors; cannot analyze failing students.")

        st.subheader("Final Assigned Grades Table")
        if st.session_state.df_graded is not None:
            df_final_display = st.session_state.df_graded.copy()
            display_cols_order = ['FirstName', 'LastName', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
            display_cols_exist_final = [col for col in display_cols_order if col in df_final_display.columns]
            df_to_style = df_final_display[display_cols_exist_final]

            reset_section_color_cycle(); unique_sections_final_table = sorted(df_to_style['Section'].unique())
            for section_name_val in unique_sections_final_table: get_section_color_fixed(section_name_val)

            styler = df_to_style.style
            if 'Letter_Grade' in df_to_style.columns: styler = styler.apply(lambda x: [f'background-color: {GRADE_GRADIENT.get(str(v), GRADE_GRADIENT["default"])}' for v in x], subset=['Letter_Grade'])
            if 'Section' in df_to_style.columns: styler = styler.apply(lambda x: [f'background-color: {get_section_color_fixed(str(v))}' for v in x], subset=['Section'])

            if 'StudentID' in df_to_style.columns and st.session_state.upgraded_students:
                id_col_highlight = 'StudentID'; fn_col_highlight = 'FirstName'; ln_col_highlight = 'LastName'
                subset_cols_for_effect = []
                if fn_col_highlight in df_to_style.columns: subset_cols_for_effect.append(fn_col_highlight)
                if ln_col_highlight in df_to_style.columns: subset_cols_for_effect.append(ln_col_highlight)
                if not subset_cols_for_effect and id_col_highlight in df_to_style.columns: subset_cols_for_effect.append(id_col_highlight)

                if subset_cols_for_effect:
                    styler = styler.apply(highlight_upgraded, axis=1,
                                          upgraded_students_set=st.session_state.upgraded_students,
                                          id_col=id_col_highlight,
                                          first_col_name=fn_col_highlight, last_col_name=ln_col_highlight,
                                          subset=subset_cols_for_effect)

            st.dataframe(styler.format({"Score": "{:.2f}", "GPA": "{:.2f}"}), use_container_width=True)

            st.subheader("Download Grades")
            sections_for_download = ["All Sections"] + sorted(df_final_display['Section'].unique().tolist())
            selected_section_download = st.selectbox("Select section to download:", options=sections_for_download, key="download_section_select_v12") # Incremented key
            def convert_df_to_csv_download(df_to_convert, section_filter_val):
                df_filtered_dl = df_to_convert[df_to_convert['Section'] == section_filter_val].copy() if section_filter_val != "All Sections" else df_to_convert.copy()
                dl_cols_exist = [col for col in display_cols_order if col in df_filtered_dl.columns]
                return df_filtered_dl[dl_cols_exist].to_csv(index=False).encode('utf-8') if dl_cols_exist else None
            try:
                csv_data_download = convert_df_to_csv_download(df_final_display, selected_section_download)
                if csv_data_download:
                    file_name_dl = f"final_grades_{selected_section_download.replace(' ', '_')}.csv" if selected_section_download != "All Sections" else "final_grades_all_sections.csv"
                    st.download_button(label=f"Download Grades for {selected_section_download}", data=csv_data_download, file_name=file_name_dl, mime='text/csv', key=f"download_btn_{selected_section_download}_v12") # Incremented key
            except Exception as e: st.error(f"Could not prepare download file: {e}")
        else: st.warning("Final graded data not available.")
    elif st.session_state.active_cutoffs: st.warning("Statistics could not be calculated. Check data, grade assignment, and cutoffs.")

st.sidebar.markdown("---"); st.sidebar.info("Iterative Grading Tool v1.13")
