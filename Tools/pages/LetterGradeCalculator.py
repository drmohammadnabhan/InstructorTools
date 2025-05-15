# Full code block v1.21 - Added alternative cutoff initialization method
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    current_start = float(a_plus_start)
    gap = float(gap)
    for i, grade in enumerate(grades_order):
        cutoffs[f'{grade}_Start'] = round(current_start, 2)
        if i < len(grades_order) - 1: # Don't subtract for the last grade (D)
            current_start -= gap
    cutoffs['F_Max'] = cutoffs['D_Start']
    return cutoffs

def calculate_cutoffs_by_endpoints(a_plus_start_score, d_start_score):
    """Calculates initial cutoffs based on A+ Start and D Start, with equal intervals in between."""
    cutoffs = collections.OrderedDict()
    grades_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D'] # 8 grades mean 7 intervals
    num_intervals = 7.0

    a_plus_start_score = float(a_plus_start_score)
    d_start_score = float(d_start_score)

    # Validation is done before calling, but as a safeguard:
    if a_plus_start_score <= d_start_score:
        # This should be caught by UI validation ideally
        st.error("A+ Start Score must be strictly greater than D Start Score for this method.")
        return None 

    total_range = a_plus_start_score - d_start_score
    if num_intervals == 0: return None # Should not happen
    
    individual_gap = total_range / num_intervals

    current_score = a_plus_start_score
    for i, grade in enumerate(grades_order):
        # The start of D is fixed by d_start_score, other grades are derived
        if grade == 'D':
            cutoffs[f'{grade}_Start'] = round(d_start_score, 2)
        else:
            cutoffs[f'{grade}_Start'] = round(current_score, 2)
        
        if i < len(grades_order) - 1: # For all grades except D
            current_score -= individual_gap
            
    cutoffs['F_Max'] = cutoffs['D_Start']
    return cutoffs


def assign_letter_grades_from_starts(scores, start_cutoffs):
    # ... (assign_letter_grades_from_starts function from v1.19 - assumed correct and unchanged)
    expected_grade_order_keys = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    present_grade_keys = [k for k in expected_grade_order_keys if k in start_cutoffs]
    if len(present_grade_keys) > 1:
        ordered_scores_for_check = [start_cutoffs[k] for k in present_grade_keys]
        if not all(ordered_scores_for_check[i] >= ordered_scores_for_check[i+1] for i in range(len(ordered_scores_for_check)-1)):
            st.error("Input Error: Grade boundary start scores (A+ to D) must be monotonically decreasing or equal.")
            return pd.Series(['Error - Non-monotonic Input Cutoffs'] * len(scores), index=scores.index, dtype='object')

    score_to_highest_grade_label_map = collections.OrderedDict()
    for grade_key_with_suffix, score_val in start_cutoffs.items():
        if grade_key_with_suffix == 'F_Max': continue
        grade_label = grade_key_with_suffix.replace('_Start', '')
        if score_val not in score_to_highest_grade_label_map:
            score_to_highest_grade_label_map[score_val] = grade_label

    if not score_to_highest_grade_label_map:
        st.error("No valid grade boundaries for assignment after processing.")
        return pd.Series(['Error - No Valid Cutoffs For Assignment'] * len(scores), index=scores.index, dtype='object')

    unique_ascending_boundary_scores = sorted(list(score_to_highest_grade_label_map.keys()))
    if not unique_ascending_boundary_scores:
         st.error("Internal Error: No unique boundary scores for binning."); return pd.Series(['Error - Binning Issue'] * len(scores), index=scores.index, dtype='object')

    labels_for_cut = ['F'] + [score_to_highest_grade_label_map[s] for s in unique_ascending_boundary_scores]
    bins = [-np.inf] + unique_ascending_boundary_scores + [np.inf]

    if len(labels_for_cut) != len(bins) - 1:
        st.error(f"CRITICAL INTERNAL ERROR: Bin/Label mismatch. Bins ({len(bins)}), Labels ({len(labels_for_cut)})."); return pd.Series(['Error - Internal Bin/Label Mismatch'] * len(scores), index=scores.index, dtype='object')

    numeric_scores = pd.to_numeric(scores, errors='coerce')
    grades = pd.cut(numeric_scores, bins=bins, labels=labels_for_cut, right=False, ordered=False)
    return grades.astype('object').fillna('Invalid Score')


def calculate_stats(df_with_gpa, grade_col, section_col, gpa_map):
    # ... (calculate_stats function from v1.19 - assumed correct and unchanged)
    if grade_col not in df_with_gpa.columns or 'GPA' not in df_with_gpa.columns:
        st.warning("Cannot calculate stats: Grade or GPA column missing in the provided DataFrame.")
        return {"error": "Missing Grade/GPA column for stats."}
    if df_with_gpa[grade_col].astype(str).str.contains('Error', na=False).any():
        st.warning("Cannot calculate reliable stats due to errors in grade assignments present in data.")

    if df_with_gpa['GPA'].isnull().all() and df_with_gpa[grade_col].notna().any() and not df_with_gpa[grade_col].isin(['Invalid Score']).all():
        st.warning("GPA values are all non-numeric despite valid letter grades (excluding 'Invalid Score'). Check GPA_SCALE.")

    overall_dist_percent = df_with_gpa[grade_col].value_counts(normalize=True).sort_index()
    overall_dist_count = df_with_gpa[grade_col].value_counts(normalize=False).sort_index()
    overall_gpa = df_with_gpa['GPA'].mean()
    df_copy_for_stats = df_with_gpa.copy()
    df_copy_for_stats[section_col] = df_copy_for_stats[section_col].astype(str)
    section_gpa_means = df_copy_for_stats.groupby(section_col)['GPA'].mean()
    section_stats_agg = df_copy_for_stats.groupby(section_col).agg(Count=('GPA', 'size'), Valid_GPA_Count=('GPA', 'count')).reset_index()
    section_stats = pd.merge(section_stats_agg, section_gpa_means.rename('Avg_GPA'), on=section_col, how='left')
    section_dist_percent, section_dist_count = pd.DataFrame(), pd.DataFrame()
    try:
        section_dist_percent = df_copy_for_stats.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)
        section_dist_count = df_copy_for_stats.groupby(section_col)[grade_col].value_counts(normalize=False).unstack(fill_value=0)
    except Exception: st.warning("Could not generate section-wise grade distributions.")
    anova_result, anova_p_value = "ANOVA not applicable.", None
    section_groups_for_anova = [g['GPA'].dropna().values for _, g in df_copy_for_stats.groupby(section_col) if g['GPA'].notna().sum() > 1]
    if len(section_groups_for_anova) > 1:
        try:
            f_val, p_val = stats.f_oneway(*section_groups_for_anova); anova_result, anova_p_value = f"ANOVA F={f_val:.2f}, p={p_val:.3f}", p_val
        except Exception as e: anova_result = f"ANOVA Error: {e}"
    return {"overall_dist_percent": overall_dist_percent, "overall_dist_count": overall_dist_count, "overall_gpa": overall_gpa,
            "section_stats": section_stats, "section_dist_percent": section_dist_percent, "section_dist_count": section_dist_count,
            "anova_result": anova_result, "anova_p_value": anova_p_value, "error": None}

GPA_SCALE = {'A+': 4.0, 'A': 3.75, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0,
             'Invalid Score': np.nan, 'Error - Non-monotonic Cutoffs': np.nan, 'Error - Non-monotonic Input Cutoffs': np.nan,
             'Error - Bin/Label Mismatch': np.nan, 'Error - Internal Bin/Label Mismatch': np.nan,
             'Error - No Valid Cutoffs For Assignment': np.nan, 'Error - Binning Issue': np.nan}
GRADE_GRADIENT = {'A+': '#FFFFFF', 'A': '#F2F2F2', 'B+': '#E6E6E6', 'B': '#D9D9D9', 'C+': '#CCCCCC', 'C': '#BDBDBD',
                  'D+': '#B0B0B0', 'D': '#A3A3A3', 'F': '#969696', 'default': '#F0F0F0'}
SECTION_COLOR_PALETTE = [cm.get_cmap('tab20')(i) for i in range(20)]
section_color_map_generated = {}; color_cycle_obj = itertools.cycle(SECTION_COLOR_PALETTE)

def get_section_color_fixed(section_name):
    global color_cycle_obj; str_section = str(section_name)
    if str_section not in section_color_map_generated:
        rgba_color = next(color_cycle_obj)
        section_color_map_generated[str_section] = '#{:02x}{:02x}{:02x}'.format(int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
    return section_color_map_generated[str_section]

def reset_section_color_cycle():
    global section_color_map_generated, color_cycle_obj
    section_color_map_generated.clear(); color_cycle_obj = itertools.cycle(SECTION_COLOR_PALETTE)

# ============================================
# Streamlit App Layout
# ============================================
st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant v1.21") # Version update
st.info("**Workflow:**\n"
        "1. Set Initial Parameters (choose initialization method).\n2. Calculate Initial Cutoffs.\n3. Upload Score File & Map Columns.\n"
        "4. Review (adjust 'Students Below Cutoffs' range if needed).\n5. Optionally: Adjust Start Scores & Apply.\n"
        "6. Review Final Results.\n7. Download grades.")

# --- Session State Initialization ---
if 'column_mappings' not in st.session_state: st.session_state.column_mappings = {'col_first': None, 'col_last': None, 'col_id': None, 'col_score': None, 'col_section': None}
for key in ['initial_cutoffs', 'active_cutoffs', 'df_graded', 'stats_results', 'processed_df']:
    if key not in st.session_state: st.session_state[key] = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'points_near_cutoff_active' not in st.session_state: st.session_state.points_near_cutoff_active = 1.5
# For persisting sidebar input values
if 'a_plus_start_val_sidebar' not in st.session_state: st.session_state.a_plus_start_val_sidebar = 95.0
if 'uniform_gap_val_sidebar' not in st.session_state: st.session_state.uniform_gap_val_sidebar = 5.0
if 'd_start_val_sidebar' not in st.session_state: st.session_state.d_start_val_sidebar = 60.0


# --- Sidebar ---
st.sidebar.header("1. Initial Parameters")

init_method_options = ("A+ Start & Uniform Gap", "A+ Start & D Start (Equal Intervals)")
init_method = st.sidebar.radio(
    "Cutoff Initialization Method",
    options=init_method_options,
    key='init_method_radio_selector'
)

# Store and retrieve values from session state to persist them across method switches
st.session_state.a_plus_start_val_sidebar = st.sidebar.number_input(
    "A+ Start Score",
    value=float(st.session_state.a_plus_start_val_sidebar), # Use float for consistency
    step=0.1, format="%.2f",
    help="Minimum score for A+.",
    key='common_a_plus_start_input'
)

uniform_grade_gap_input = None
d_start_score_input = None

if init_method == init_method_options[0]: # "A+ Start & Uniform Gap"
    st.session_state.uniform_gap_val_sidebar = st.sidebar.number_input(
        "Uniform Grade Gap (points)",
        value=float(st.session_state.uniform_gap_val_sidebar),
        step=0.1, min_value=0.01, format="%.2f", # min_value > 0
        help="Gap between grade starts (e.g., A+ to A, A to B+).",
        key='uniform_gap_specific_input'
    )
    uniform_grade_gap_input = st.session_state.uniform_gap_val_sidebar
else: # "A+ Start & D Start (Equal Intervals)"
    st.session_state.d_start_val_sidebar = st.sidebar.number_input(
        "D Start Score",
        value=float(st.session_state.d_start_val_sidebar),
        step=0.1, format="%.2f",
        help="Minimum score for D.",
        key='d_start_specific_input'
    )
    d_start_score_input = st.session_state.d_start_val_sidebar


def update_active_points_from_sidebar(): st.session_state.points_near_cutoff_active = st.session_state.points_near_num_v8_sidebar_key # Incremented key
st.sidebar.number_input("Initial: Students Below Cutoff (X pts)", 0.1, 10.0, st.session_state.points_near_cutoff_active, 0.1, "%.1f",
                        key='points_near_num_v8_sidebar_key', on_change=update_active_points_from_sidebar,
                        help="Initial range. Can be overridden in main panel.")

if st.sidebar.button("Calculate Initial Cutoffs"):
    calculated_cutoffs_result = None
    a_plus_val = st.session_state.a_plus_start_val_sidebar

    if init_method == init_method_options[0]:
        if uniform_grade_gap_input is not None and uniform_grade_gap_input > 0:
            calculated_cutoffs_result = calculate_initial_cutoffs_original(a_plus_val, uniform_grade_gap_input)
        else:
            st.sidebar.error("Uniform Grade Gap must be greater than 0.")
    elif init_method == init_method_options[1]:
        if d_start_score_input is not None:
            if a_plus_val <= d_start_score_input:
                st.sidebar.error("A+ Start Score must be strictly greater than D Start Score.")
            else:
                calculated_cutoffs_result = calculate_cutoffs_by_endpoints(a_plus_val, d_start_score_input)
        else: # Should not happen due to UI structure
            st.sidebar.error("D Start Score is missing for the selected method.")

    if calculated_cutoffs_result:
        st.session_state.initial_cutoffs = calculated_cutoffs_result
        st.session_state.active_cutoffs = calculated_cutoffs_result
        st.session_state.manual_override_values = {**calculated_cutoffs_result}
        st.session_state.df_graded, st.session_state.stats_results = None, None
        st.sidebar.success(f"Initial cutoffs calculated using: {init_method}")
        if st.session_state.data_loaded: st.rerun()
    elif not (init_method == init_method_options[1] and a_plus_val <= d_start_score_input) and \
         not (init_method == init_method_options[0] and (uniform_grade_gap_input is None or uniform_grade_gap_input <=0)):
        # If not one of the explicit user input errors above, but calculation still failed
        st.sidebar.error("Failed to calculate cutoffs. Please check input values.")


# --- Main Area ---
# ... (Rest of the Main Area code from v1.19, starting with col_cutoff_table, col_width_table) ...
# ... This includes Upload Section, Manual Cutoff Adjustment, Recalculation logic, ...
# ... Visualizations, Students Below Cutoffs, Final Results (Table, Plot, GPA, Failing, Download) ...
# Ensure all keys for widgets are unique and incremented if structure changed significantly.

col_cutoff_table, col_width_table = st.columns([1, 2])
with col_cutoff_table:
    st.header("Active Cutoffs")
    if st.session_state.active_cutoffs:
        st.dataframe(pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Boundary', 'Score']).style.format({"Score": "{:.2f}"}))
    else: st.warning("Calculate or apply cutoffs first.")
with col_width_table:
    if st.session_state.active_cutoffs:
        st.header("Grade Widths"); grades_in_desc_order = [k for k in st.session_state.active_cutoffs.keys() if k != 'F_Max']
        widths = collections.OrderedDict(); max_cap = 100.0
        for i, gk_s in enumerate(grades_in_desc_order):
            start_s = st.session_state.active_cutoffs[gk_s]
            upper_b = max_cap if i == 0 else st.session_state.active_cutoffs[grades_in_desc_order[i-1]]
            width_v = upper_b - start_s
            widths[gk_s.replace('_Start','')] = f"{max(0,width_v):.2f} [{start_s:.2f} – <{upper_b:.2f})"
        if widths: st.dataframe(pd.DataFrame(list(widths.items()), columns=['Grade', 'Width [Start – End)']))
        else: st.write("Cannot calculate widths.")

st.header("Upload & Prepare Data")
def new_file_uploaded_callback_v21(): # Renamed for clarity
    st.session_state.column_mappings = {'col_first': None, 'col_last': None, 'col_id': None, 'col_score': None, 'col_section': None}
    st.session_state.data_loaded = False; st.session_state.processed_df = None
    st.session_state.df_graded = None; st.session_state.stats_results = None
uploaded_file = st.file_uploader("Upload scores (CSV/Excel)", ["csv", "xlsx"], key="file_uploader_v22_main", on_change=new_file_uploaded_callback_v21) # Incremented key

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded.")
        st.subheader("Map Columns"); cols_from_file = df_upload.columns.tolist(); cols_with_none_option = ["<Select Column>"] + cols_from_file
        current_map = st.session_state.column_mappings
        def get_idx(cn, cfn): return cfn.index(cn) if cn and cn in cfn else 0
        key_sfx = "_fix_v22" # Incremented key
        col_first_selection = st.selectbox("First Name (Optional)", cols_with_none_option, get_idx(current_map['col_first'], cols_with_none_option), key='sel_first'+key_sfx)
        col_last_selection = st.selectbox("Last Name (Optional)", cols_with_none_option, get_idx(current_map['col_last'], cols_with_none_option), key='sel_last'+key_sfx)
        col_id_selection = st.selectbox("Student ID (Optional)", cols_with_none_option, get_idx(current_map['col_id'], cols_with_none_option), key='sel_id'+key_sfx)
        col_score_selection = st.selectbox("Score Column*", cols_with_none_option, get_idx(current_map['col_score'], cols_with_none_option), key='sel_score'+key_sfx)
        col_section_selection = st.selectbox("Section Column*", cols_with_none_option, get_idx(current_map['col_section'], cols_with_none_option), key='sel_section'+key_sfx)
        st.caption("*Mandatory columns")
        if col_score_selection != "<Select Column>" and col_section_selection != "<Select Column>":
            current_map['col_first'] = col_first_selection if col_first_selection != "<Select Column>" else None
            current_map['col_last'] = col_last_selection if col_last_selection != "<Select Column>" else None
            current_map['col_id'] = col_id_selection if col_id_selection != "<Select Column>" else None
            current_map['col_score'] = col_score_selection; current_map['col_section'] = col_section_selection
            df = df_upload.copy(); df.rename(columns={col_score_selection: 'Score', col_section_selection: 'Section'}, inplace=True)
            final_cols = ['Score', 'Section']
            if current_map['col_id']: df.rename(columns={current_map['col_id']: 'StudentID'}, inplace=True); final_cols.append('StudentID')
            elif 'StudentID' not in df.columns: df['StudentID'] = 'Stud_' + df.index.astype(str); final_cols.append('StudentID')
            if current_map['col_first']: df.rename(columns={current_map['col_first']: 'FirstName'}, inplace=True); final_cols.append('FirstName')
            if current_map['col_last']: df.rename(columns={current_map['col_last']: 'LastName'}, inplace=True); final_cols.append('LastName')
            df = df[[c for c in final_cols if c in df.columns]]
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce'); initial_rows = len(df)
            df.dropna(subset=['Score'], inplace=True); removed_rows = initial_rows - len(df)
            if removed_rows > 0: st.warning(f"Removed {removed_rows} rows with invalid/missing scores.")
            if df.empty: st.error("No valid score data."); st.session_state.data_loaded, st.session_state.processed_df = False, None
            else:
                for col, type_func in [('Section', str), ('StudentID', str), ('FirstName', str), ('LastName', str)]:
                    if col in df.columns: df[col] = df[col].astype(type_func).fillna('')
                st.session_state.processed_df = df; st.session_state.data_loaded = True; st.success("Data mapped.")
                reset_section_color_cycle()
                preview_cols = [c for c in ['FirstName','LastName','StudentID','Score','Section'] if c in df.columns][:5]
                if preview_cols : st.dataframe(df[preview_cols].head())
                st.session_state.df_graded, st.session_state.stats_results = None, None
        else: st.warning("Select Score & Section columns."); st.session_state.data_loaded, st.session_state.processed_df = False, None
    except Exception as e: st.error(f"Error loading/processing file: {e}"); st.session_state.data_loaded, st.session_state.processed_df = False, None

df_display = st.session_state.processed_df

if st.session_state.data_loaded and df_display is not None and st.session_state.active_cutoffs:
    st.header("Manual Cutoff Adjustment")
    st.markdown("Adjust **Start Score** for each grade. Scores must be monotonically decreasing (A+ ≥ A ≥ ... ≥ D).")
    manual_cutoffs_input = {}; grade_keys_in_order = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start']
    current_manual_vals = st.session_state.manual_override_values; num_grades = len(grade_keys_in_order)
    cols_per_row = 4; num_rows = (num_grades + cols_per_row - 1) // cols_per_row
    for r_idx in range(num_rows): # Renamed r to r_idx
        cols = st.columns(cols_per_row)
        for c_idx in range(cols_per_row):
            grade_idx = r_idx * cols_per_row + c_idx
            if grade_idx < num_grades:
                key = grade_keys_in_order[grade_idx];
                with cols[c_idx]:
                    default_val = float(current_manual_vals.get(key, st.session_state.active_cutoffs.get(key, 0.0)))
                    manual_cutoffs_input[key] = st.number_input(key.replace('_Start', ' Start'), value=default_val, step=0.1, key=f'man_{key}_v23', format="%.2f") # Incremented key
    manual_cutoffs_input['F_Max'] = manual_cutoffs_input.get('D_Start', 0.0)
    if st.button("Apply Manual Cutoffs & Recalculate"):
        scores_list = [manual_cutoffs_input[key] for key in grade_keys_in_order if key in manual_cutoffs_input]
        if not all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)): st.error("Manual Start Scores (A+ to D) must be monotonically decreasing or equal.")
        else:
            new_active_cutoffs = collections.OrderedDict()
            all_grade_keys_ordered = ['A+_Start', 'A_Start', 'B+_Start', 'B_Start', 'C+_Start', 'C_Start', 'D+_Start', 'D_Start', 'F_Max']
            for key in all_grade_keys_ordered:
                if key in manual_cutoffs_input: new_active_cutoffs[key] = manual_cutoffs_input[key]
            st.session_state.active_cutoffs = new_active_cutoffs; st.session_state.manual_override_values = manual_cutoffs_input.copy()
            st.session_state.df_graded, st.session_state.stats_results = None, None
            st.success("Manual cutoffs applied. Recalculating results..."); st.rerun()

    if st.session_state.df_graded is None:
        temp_df_for_grading = None
        try:
            df_calc = df_display.copy()
            df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)
            if 'Letter_Grade' in df_calc.columns:
                df_calc['GPA'] = df_calc['Letter_Grade'].map(GPA_SCALE)
                df_calc['GPA'] = pd.to_numeric(df_calc['GPA'], errors='coerce')
            else:
                st.warning("Internal issue: 'Letter_Grade' column was not created. 'GPA' column will be all NaN.")
                df_calc['GPA'] = np.nan
            temp_df_for_grading = df_calc
            if 'Letter_Grade' in temp_df_for_grading.columns and not temp_df_for_grading['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
                st.session_state.stats_results = calculate_stats(temp_df_for_grading, 'Letter_Grade', 'Section', GPA_SCALE)
            else:
                st.error("Statistics not calculated due to errors in grade assignment or missing Letter_Grade column.")
                st.session_state.stats_results = None
            st.session_state.df_graded = temp_df_for_grading
        except Exception as e:
            st.error(f"An unexpected error occurred during grade assignment or statistics calculation: {e}")
            st.session_state.df_graded = temp_df_for_grading if temp_df_for_grading is not None else None
            if st.session_state.df_graded is None and df_display is not None:
                st.session_state.df_graded = df_display.copy()
                if 'Letter_Grade' not in st.session_state.df_graded.columns: st.session_state.df_graded['Letter_Grade'] = "Processing Error"
                if 'GPA' not in st.session_state.df_graded.columns: st.session_state.df_graded['GPA'] = np.nan
            st.session_state.stats_results = None

    st.header("Visualization & Observation")
    st.session_state.points_near_cutoff_active = st.number_input("Range for 'Students Below Cutoffs'", 0.1, 10.0, st.session_state.points_near_cutoff_active, 0.1, "%.1f", key='points_near_override_v12') # Incremented key
    active_points_near = st.session_state.points_near_cutoff_active
    plot_cutoffs = sorted(list(set(v for k, v in st.session_state.active_cutoffs.items() if k != 'F_Max')))
    st.subheader("Score Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(df_display['Score'], kde=True, ax=ax_hist, bins='auto', stat="density", color=get_section_color_fixed("OverallDistributionScore"))
    for co in plot_cutoffs: ax_hist.axvline(co, color='red', ls='--', lw=1)
    ax_hist.set_title("Score Distribution"); ax_hist.set_xlabel("Score"); st.pyplot(fig_hist); plt.close(fig_hist)
    st.subheader("Individual Scores by Section")
    if 'Section' in df_display.columns:
        reset_section_color_cycle(); unique_sections_plot = sorted(df_display['Section'].unique())
        plot_palette_strip = {section: get_section_color_fixed(section) for section in unique_sections_plot}
        fig_strip, ax_strip = plt.subplots(figsize=(10, max(4, len(unique_sections_plot) * 0.6)))
        sns.stripplot(data=df_display, x='Score', y='Section', hue='Section', order=unique_sections_plot, hue_order=unique_sections_plot,
                      ax=ax_strip, jitter=0.3, size=5, alpha=0.8, legend="auto", palette=plot_palette_strip)
        for co in plot_cutoffs: ax_strip.axvline(co, color='red', ls='--', lw=1)
        ax_strip.set_title("Individual Scores by Section"); ax_strip.set_xlabel("Score"); ax_strip.set_ylabel("Section")
        if len(unique_sections_plot) > 1 and len(unique_sections_plot) <= 20 : ax_strip.legend(title='Section', bbox_to_anchor=(1.02, 1), loc='upper left', ncol= 1 if len(unique_sections_plot) <=10 else 2)
        elif ax_strip.get_legend() is not None: ax_strip.get_legend().remove()
        plt.tight_layout(rect=[0, 0, 0.85 if len(unique_sections_plot) > 1 and len(unique_sections_plot) <=20 else 1, 1]); st.pyplot(fig_strip); plt.close(fig_strip)
    else: st.warning("Section column not found for section-colored plot.")

    st.subheader(f"Students Below Cutoffs (within {active_points_near:.1f} points)")
    students_near_df_local = pd.DataFrame(); df_for_near_cutoff_check = st.session_state.df_graded
    if df_for_near_cutoff_check is not None and 'Letter_Grade' in df_for_near_cutoff_check.columns and not df_for_near_cutoff_check['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
        score_to_definitive_grade_label_map = collections.OrderedDict()
        for gk_s_map, sv_s_map in st.session_state.active_cutoffs.items():
            if gk_s_map == 'F_Max': continue
            gl_s_map = gk_s_map.replace('_Start', '')
            if sv_s_map not in score_to_definitive_grade_label_map: score_to_definitive_grade_label_map[sv_s_map] = gl_s_map
        students_near_cutoff_list = []
        for boundary_score_val in plot_cutoffs:
            min_score_nearby = boundary_score_val - active_points_near
            nearby_df = df_for_near_cutoff_check[(df_for_near_cutoff_check['Score'] >= min_score_nearby) & (df_for_near_cutoff_check['Score'] < boundary_score_val)].copy()
            if not nearby_df.empty:
                nearby_df.loc[:, 'Target_Boundary'] = boundary_score_val; nearby_df.loc[:, 'Points_to_Upgrade'] = boundary_score_val - nearby_df['Score']
                nearby_df.loc[:, 'Target_Grade'] = score_to_definitive_grade_label_map.get(boundary_score_val, "N/A")
                students_near_cutoff_list.append(nearby_df)
        if students_near_cutoff_list:
            students_near_df_local = pd.concat(students_near_cutoff_list).sort_values(by=['Target_Boundary', 'Score'])
            display_cols_near = [c for c in ['FirstName','LastName','StudentID','Score','Section','Letter_Grade','Target_Grade','Target_Boundary','Points_to_Upgrade'] if c in students_near_df_local.columns]
            if display_cols_near: st.dataframe(students_near_df_local[display_cols_near].style.format({"Score": "{:.2f}", "Points_to_Upgrade": "{:.2f}", "Target_Boundary": "{:.2f}"}))
        else: st.write(f"No students found scoring between (Cutoff - {active_points_near:.1f}) and (Cutoff).")
    else: st.warning("Grade calculation needed or has errors for 'Students Near Cutoffs'.")

    st.header("Final Results")
    if st.session_state.stats_results and not st.session_state.stats_results.get("error"):
        results = st.session_state.stats_results
        st.subheader("Grade Distributions (Count & Percentage)")
        if "overall_dist_percent" in results and not results['overall_dist_percent'].empty:
            try:
                overall_pct, overall_cnt = results['overall_dist_percent'], results['overall_dist_count']
                section_pct, section_cnt = results['section_dist_percent'], results['section_dist_count']
                all_grades_order = ['A+','A','B+','B','C+','C','D+','D','F'] + [g for g in GPA_SCALE if 'Error' in g or 'Invalid' in g]
                present_grades = overall_cnt.index.union(section_cnt.columns if not section_cnt.empty else [])
                sorted_grades_display = [g for g in all_grades_order if g in present_grades] + [g for g in present_grades if g not in all_grades_order]
                rows = []
                for grade in sorted_grades_display:
                    row = {'Grade': grade, 'Overall': f"{overall_cnt.get(grade,0)} ({overall_pct.get(grade,0)*100:.1f}%)"}
                    if not section_cnt.empty and isinstance(section_cnt, pd.DataFrame):
                        for sec_name in section_cnt.index:
                            s_cnt = section_cnt.loc[sec_name, grade] if grade in section_cnt.columns else 0
                            s_pct_val = 0.0
                            if not section_pct.empty and isinstance(section_pct, pd.DataFrame) and sec_name in section_pct.index and grade in section_pct.columns:
                                s_pct_val = section_pct.loc[sec_name, grade] * 100
                            row[str(sec_name)] = f"{s_cnt} ({s_pct_val:.1f}%)"
                    rows.append(row)
                if rows: st.markdown(pd.DataFrame(rows).set_index('Grade').to_html(escape=False), unsafe_allow_html=True)
                else: st.write("No distribution data for table.")
                st.write("---")
                st.subheader("Grade Counts by Section (Bar Plot)")
                if not section_cnt.empty and isinstance(section_cnt, pd.DataFrame):
                    df_plot_counts = section_cnt.T.reindex(sorted_grades_display).fillna(0)
                    df_plot_counts = df_plot_counts.loc[(df_plot_counts.sum(axis=1) > 0)]
                    if not df_plot_counts.empty:
                        reset_section_color_cycle(); plot_colors = [get_section_color_fixed(str(sec)) for sec in df_plot_counts.columns]
                        fig_bar_dist, ax_bar_dist = plt.subplots(figsize=(12, 6))
                        df_plot_counts.plot(kind='bar', ax=ax_bar_dist, color=plot_colors, width=0.8)
                        ax_bar_dist.set_title('Grade Counts by Section'); ax_bar_dist.set_ylabel('Number of Students'); ax_bar_dist.set_xlabel('Grade')
                        plt.xticks(rotation=45, ha='right'); ax_bar_dist.legend(title='Section', bbox_to_anchor=(1.02, 1), loc='upper left')
                        plt.tight_layout(rect=[0, 0, 0.85, 1]); st.pyplot(fig_bar_dist); plt.close(fig_bar_dist)
                    else: st.write("No data to plot for grade counts by section (all counts might be zero).")
                else: st.write("Section count data not available for bar plot.")
            except Exception as e: st.error(f"Error displaying grade distributions or plot: {e}")
        else: st.write("Distribution data missing.")
        st.write(f"**Overall Avg GPA:** {results.get('overall_gpa', np.nan):.2f}" if pd.notna(results.get('overall_gpa')) else "**Overall Avg GPA:** N/A"); st.write("---")
        st.subheader("Section GPA Comparison")
        col_gpa_table_final, col_gpa_plot_final = st.columns(2)
        with col_gpa_table_final:
            st.write("**Per Section Avg GPA:**")
            if "section_stats" in results and not results['section_stats'].empty:
                st.dataframe(results['section_stats'][['Section', 'Avg_GPA', 'Valid_GPA_Count', 'Count']].rename(columns={'Valid_GPA_Count':'Graded', 'Count':'Total'}).style.format({"Avg_GPA": "{:.2f}"}))
            st.write(f"**ANOVA Result:** {results.get('anova_result', 'N/A')}")
            anova_p = results.get('anova_p_value'); alpha = 0.05
            if anova_p is not None:
                if anova_p < alpha: st.markdown(f"<span style='color:orange;'>Significant difference in section GPAs detected (p={anova_p:.3f}). Pairwise comparisons (e.g., Fisher's LSD) could identify specific differences if numeric GPA data is reliable.</span>", unsafe_allow_html=True)
                else: st.markdown(f"No significant difference in section GPAs (p={anova_p:.3f}).", unsafe_allow_html=True)
        with col_gpa_plot_final:
            st.write("**GPA Distribution by Section (Boxplot)**")
            gpa_boxplot_drawn = False
            if st.session_state.df_graded is not None and 'GPA' in st.session_state.df_graded.columns and st.session_state.df_graded['GPA'].notna().any():
                try:
                    fig_box, ax_box = plt.subplots(); sections_sorted = sorted(st.session_state.df_graded['Section'].unique())
                    sns.boxplot(data=st.session_state.df_graded, x='Section', y='GPA', ax=ax_box, order=sections_sorted)
                    ax_box.set_title("GPA Distribution by Section"); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig_box); plt.close(fig_box)
                    gpa_boxplot_drawn = True
                except Exception as e: st.warning(f"Could not generate GPA boxplot: {e}")
            if not gpa_boxplot_drawn: st.warning("Boxplot not generated. This is likely because all derived GPA values are non-numeric (NaN) or data is unavailable. Please check assigned letter grades and ensure they map to numeric GPAs.")

        st.subheader("Failing Students Analysis")
        if st.session_state.df_graded is not None and 'Letter_Grade' in st.session_state.df_graded.columns:
            passing_score_boundary = st.session_state.active_cutoffs.get('D_Start', None)
            if passing_score_boundary is not None:
                failing_students_df = st.session_state.df_graded[st.session_state.df_graded['Letter_Grade'] == 'F'].copy()
                if not failing_students_df.empty:
                    failing_students_df['Points_Below_Pass'] = passing_score_boundary - failing_students_df['Score']
                    failing_students_df.sort_values('Points_Below_Pass', ascending=True, inplace=True)
                    st.write(f"Passing Score (Minimum for D): {passing_score_boundary:.2f}")
                    fail_cols_display = [c for c in ['FirstName','LastName','StudentID','Score','Section','Points_Below_Pass'] if c in failing_students_df.columns]
                    st.dataframe(failing_students_df[fail_cols_display].style.format({"Score": "{:.2f}", "Points_Below_Pass": "{:.2f}"}))
                else: st.success("No students received an 'F' grade.")
            else: st.warning("Could not determine D_Start to analyze failing students.")
        else: st.warning("Final grades needed/contain errors; cannot analyze failing students.")

        st.subheader("Final Assigned Grades Table")
        if st.session_state.df_graded is not None:
            df_final_styled = st.session_state.df_graded.copy()
            display_cols_final = [c for c in ['FirstName','LastName','StudentID','Score','Section','Letter_Grade','GPA'] if c in df_final_styled.columns]
            df_to_style_final = df_final_styled[display_cols_final]
            reset_section_color_cycle(); unique_sections_for_final_table = sorted(df_to_style_final['Section'].unique())
            for sec_name_final in unique_sections_for_final_table: get_section_color_fixed(sec_name_final)
            styler_final = df_to_style_final.style
            if 'Letter_Grade' in df_to_style_final.columns: styler_final = styler_final.apply(lambda x: [f'background-color: {GRADE_GRADIENT.get(str(v), GRADE_GRADIENT["default"])}' for v in x], subset=['Letter_Grade'])
            if 'Section' in df_to_style_final.columns: styler_final = styler_final.apply(lambda x: [f'background-color: {get_section_color_fixed(str(v))}' for v in x], subset=['Section'])
            st.dataframe(styler_final.format({"Score": "{:.2f}", "GPA": "{:.2f}"}), use_container_width=True)

            st.subheader("Download Grades")
            if not df_final_styled.empty:
                sections_for_download = ["All Sections"] + sorted(df_final_styled['Section'].unique().tolist())
                selected_section_download = st.selectbox("Select section to download:", sections_for_download, key="download_section_select_v19") # Inc key
                def convert_df_to_csv_download(df_to_convert, section_filter_val):
                    df_filtered_dl = df_to_convert.copy()
                    if section_filter_val != "All Sections": df_filtered_dl = df_to_convert[df_to_convert['Section'] == section_filter_val].copy()
                    dl_cols_ordered = ['FirstName', 'LastName', 'StudentID', 'Score', 'Section', 'Letter_Grade', 'GPA']
                    dl_cols_exist_in_df = [col for col in dl_cols_ordered if col in df_filtered_dl.columns]
                    return df_filtered_dl[dl_cols_exist_in_df].to_csv(index=False).encode('utf-8') if dl_cols_exist_in_df else None
                try:
                    csv_data_download = convert_df_to_csv_download(df_final_styled, selected_section_download)
                    if csv_data_download:
                        file_name_dl = f"final_grades_{selected_section_download.replace(' ', '_')}.csv" if selected_section_download != "All Sections" else "final_grades_all_sections.csv"
                        st.download_button(label=f"Download Grades for {selected_section_download}", data=csv_data_download, file_name=file_name_dl, mime='text/csv', key=f"download_btn_{selected_section_download.replace(' ', '_')}_v19") # Inc key
                    elif selected_section_download: st.warning(f"No data for section: {selected_section_download}")
                except Exception as e: st.error(f"Could not prepare download file: {e}")
            else: st.warning("No final graded data to download.")
        else: st.warning("Final graded data not available.")
    elif st.session_state.active_cutoffs: st.warning("Statistics could not be calculated. Check data, grade assignment, and cutoffs.")

st.sidebar.markdown("---"); st.sidebar.info("Iterative Grading Tool v1.21")
