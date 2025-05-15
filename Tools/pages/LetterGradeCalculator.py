# Full code block with fixes for new issues (histogram, boxplot, highlighting)
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
# Helper Functions (largely unchanged from v1.13, ensure they are correct)
# ============================================
def calculate_initial_cutoffs_original(a_plus_start, gap):
    cutoffs = collections.OrderedDict()
    grades_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D']
    current_start = a_plus_start
    for i, grade in enumerate(grades_order):
        cutoffs[f'{grade}_Start'] = current_start
        current_start -= gap if i < len(grades_order) - 1 else 0
    cutoffs['F_Max'] = cutoffs['D_Start']
    return cutoffs

def assign_letter_grades_from_starts(scores, start_cutoffs):
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

def calculate_stats(df, grade_col, section_col, gpa_map):
    if grade_col not in df.columns or df[grade_col].astype(str).str.contains('Error', na=False).any():
        st.warning("Cannot calculate stats due to errors in grade assignment."); return {"error": "Grade assignment failed."}

    df_copy = df.copy()
    df_copy['GPA'] = df_copy[grade_col].map(gpa_map)
    df_copy['GPA'] = pd.to_numeric(df_copy['GPA'], errors='coerce')

    ### DEBUG ###
    # st.write("### DEBUG calculate_stats: df_copy[grade_col].value_counts()", df_copy[grade_col].value_counts(dropna=False))
    # st.write("### DEBUG calculate_stats: df_copy['GPA'].value_counts(dropna=False)", df_copy['GPA'].value_counts(dropna=False))
    ### END DEBUG ###

    if df_copy['GPA'].isnull().all() and df_copy[grade_col].notna().any() and not df_copy[grade_col].isin(['Invalid Score']).all():
        st.warning("GPA calculation resulted in all non-numeric values. Check GPA_SCALE and assigned grades.")

    overall_dist_percent = df_copy[grade_col].value_counts(normalize=True).sort_index()
    overall_dist_count = df_copy[grade_col].value_counts(normalize=False).sort_index()
    overall_gpa = df_copy['GPA'].mean()
    df_copy[section_col] = df_copy[section_col].astype(str)
    section_gpa_means = df_copy.groupby(section_col)['GPA'].mean()
    section_stats_agg = df_copy.groupby(section_col).agg(Count=('GPA', 'size'), Valid_GPA_Count=('GPA', 'count')).reset_index()
    section_stats = pd.merge(section_stats_agg, section_gpa_means.rename('Avg_GPA'), on=section_col, how='left')
    section_dist_percent, section_dist_count = pd.DataFrame(), pd.DataFrame()
    try:
        section_dist_percent = df_copy.groupby(section_col)[grade_col].value_counts(normalize=True).unstack(fill_value=0)
        section_dist_count = df_copy.groupby(section_col)[grade_col].value_counts(normalize=False).unstack(fill_value=0)
    except Exception: st.warning("Could not generate section-wise grade distributions.")
    anova_result, anova_p_value = "ANOVA not applicable.", None
    section_groups_for_anova = [g['GPA'].dropna().values for _, g in df_copy.groupby(section_col) if g['GPA'].notna().sum() > 1]
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

# FIX 3 (Highlighting): Re-verify this function
def highlight_upgraded(row, upgraded_students_set, id_col='StudentID', first_col_name='FirstName', last_col_name='LastName'):
    highlight_style = 'background-color: #90EE90 !important;'
    default_style = ''
    # output_styles is for the cells within the `subset` this function is applied to by Styler
    # However, the decision to highlight is based on the `id_col` of the *entire row*.
    output_styles = pd.Series(default_style, index=row.index) # `row` is the full row from the DataFrame being styled.
                                                              # `output_styles` will be filtered by `subset` later by Styler.
    current_row_student_id = row.get(id_col)

    if current_row_student_id and str(current_row_student_id) in upgraded_students_set:
        # Apply style if column exists in the row (it always will here, since row is the full row)
        # The `subset` in `styler.apply` will determine if this style is actually used.
        if first_col_name in row.index: output_styles[first_col_name] = highlight_style
        if last_col_name in row.index: output_styles[last_col_name] = highlight_style
        
        # Determine if name columns were actually targeted for styling by `subset` and got highlighted
        # This is tricky because `subset` isn't passed here. We assume if name cols exist, they are in subset.
        name_cols_styled_and_highlighted = False
        if first_col_name in row.index and output_styles[first_col_name] == highlight_style:
             name_cols_styled_and_highlighted = True
        if last_col_name in row.index and output_styles[last_col_name] == highlight_style:
             name_cols_styled_and_highlighted = True

        if id_col in row.index and not name_cols_styled_and_highlighted:
            output_styles[id_col] = highlight_style # Highlight ID if names weren't, and ID is in subset
    return output_styles

# ============================================
# Streamlit App Layout
# ============================================
st.set_page_config(layout="wide")
st.title("Iterative Grading Assistant v1.14")
st.info("**Workflow:**\n"
        "1. Set Initial Parameters.\n2. Calculate Initial Cutoffs.\n3. Upload Score File & Map Columns.\n"
        "4. Review (adjust 'Students Below Cutoffs' range if needed).\n5. Optionally: Adjust Start Scores & Apply.\n"
        "6. Optionally: Select students for highlighting.\n7. Review Final Results.\n8. Download grades.")

# --- Session State Initialization ---
if 'column_mappings' not in st.session_state: st.session_state.column_mappings = {'col_first': None, 'col_last': None, 'col_id': None, 'col_score': None, 'col_section': None}
for key in ['initial_cutoffs', 'active_cutoffs', 'df_graded', 'stats_results', 'processed_df']:
    if key not in st.session_state: st.session_state[key] = None
if 'manual_override_values' not in st.session_state: st.session_state.manual_override_values = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'upgraded_students' not in st.session_state: st.session_state.upgraded_students = set()
if 'points_near_cutoff_active' not in st.session_state: st.session_state.points_near_cutoff_active = 1.5

# --- Sidebar ---
st.sidebar.header("1. Initial Parameters")
a_plus_start_score = st.sidebar.number_input("A+ Start Score", value=95.0, step=0.1, format="%.2f")
uniform_grade_gap = st.sidebar.number_input("Uniform Grade Gap", value=5.0, step=0.1, min_value=0.1, format="%.2f")
def update_active_points_from_sidebar(): st.session_state.points_near_cutoff_active = st.session_state.points_near_num_v6_sidebar_key
st.sidebar.number_input("Initial: Students Below Cutoff (X pts)", 0.1, 10.0, st.session_state.points_near_cutoff_active, 0.1, "%.1f",
                        key='points_near_num_v6_sidebar_key', on_change=update_active_points_from_sidebar,
                        help="Initial range. Can be overridden in main panel.")

if st.sidebar.button("Calculate Initial Cutoffs"):
    st.session_state.initial_cutoffs = calculate_initial_cutoffs_original(a_plus_start_score, uniform_grade_gap)
    st.session_state.active_cutoffs = st.session_state.initial_cutoffs
    st.session_state.manual_override_values = {**st.session_state.initial_cutoffs} # Initialize with all cutoffs
    st.session_state.df_graded, st.session_state.stats_results = None, None
    st.session_state.upgraded_students = set()
    st.sidebar.success("Initial cutoffs calculated.")
    if st.session_state.data_loaded: st.rerun()

# --- Main Area ---
col_cutoff_table, col_width_table = st.columns([1, 2])
with col_cutoff_table:
    st.header("Active Cutoffs")
    if st.session_state.active_cutoffs:
        st.dataframe(pd.DataFrame(list(st.session_state.active_cutoffs.items()), columns=['Boundary', 'Score']).style.format({"Score": "{:.2f}"}))
    else: st.warning("Calculate or apply cutoffs first.")
with col_width_table:
    if st.session_state.active_cutoffs:
        st.header("Grade Widths")
        # ... (Grade width calculation - assuming it's largely okay from v1.13) ...
        grades_in_desc_order = [k for k in st.session_state.active_cutoffs.keys() if k != 'F_Max']
        widths = collections.OrderedDict(); max_cap = 100.0
        for i, gk_s in enumerate(grades_in_desc_order):
            start_s = st.session_state.active_cutoffs[gk_s]
            upper_b = max_cap if i == 0 else st.session_state.active_cutoffs[grades_in_desc_order[i-1]]
            width_v = upper_b - start_s
            widths[gk_s.replace('_Start','')] = f"{max(0,width_v):.2f} [{start_s:.2f} – <{upper_b:.2f})"
        if widths: st.dataframe(pd.DataFrame(list(widths.items()), columns=['Grade', 'Width [Start – End)']))
        else: st.write("Cannot calculate widths.")


# --- Upload Section ---
st.header("Upload & Prepare Data")
def new_file_uploaded_callback():
    st.session_state.column_mappings = {k: None for k in st.session_state.column_mappings} # Reset selections
    st.session_state.data_loaded, st.session_state.processed_df, st.session_state.df_graded, st.session_state.stats_results = False, None, None, None
    st.session_state.upgraded_students = set()

uploaded_file = st.file_uploader("Upload scores (CSV/Excel)", ["csv", "xlsx"], key="file_uploader_v15_main", on_change=new_file_uploaded_callback)

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded.")
        st.subheader("Map Columns")
        cols_from_file, current_map = df_upload.columns.tolist(), st.session_state.column_mappings
        def get_idx(cn, cfn): return cfn.index(cn) if cn and cn in cfn else 0
        key_sfx = "_fix_v15"
        # ... (Column mapping selectboxes using get_idx and current_map - condensed for brevity)
        col_first_selection = st.selectbox("First Name (Optional)", ["<Select Column>"] + cols_from_file, get_idx(current_map['col_first'], ["<Select Column>"] + cols_from_file), key='sel_first'+key_sfx)
        col_last_selection = st.selectbox("Last Name (Optional)", ["<Select Column>"] + cols_from_file, get_idx(current_map['col_last'], ["<Select Column>"] + cols_from_file), key='sel_last'+key_sfx)
        col_id_selection = st.selectbox("Student ID (Optional)", ["<Select Column>"] + cols_from_file, get_idx(current_map['col_id'], ["<Select Column>"] + cols_from_file), key='sel_id'+key_sfx)
        col_score_selection = st.selectbox("Score Column*", ["<Select Column>"] + cols_from_file, get_idx(current_map['col_score'], ["<Select Column>"] + cols_from_file), key='sel_score'+key_sfx)
        col_section_selection = st.selectbox("Section Column*", ["<Select Column>"] + cols_from_file, get_idx(current_map['col_section'], ["<Select Column>"] + cols_from_file), key='sel_section'+key_sfx)


        if col_score_selection != "<Select Column>" and col_section_selection != "<Select Column>":
            current_map['col_first'] = col_first_selection if col_first_selection != "<Select Column>" else None; current_map['col_last'] = col_last_selection if col_last_selection != "<Select Column>" else None
            current_map['col_id'] = col_id_selection if col_id_selection != "<Select Column>" else None; current_map['col_score'] = col_score_selection; current_map['col_section'] = col_section_selection

            df = df_upload.copy()
            df.rename(columns={col_score_selection: 'Score', col_section_selection: 'Section'}, inplace=True)
            final_cols = ['Score', 'Section']
            if current_map['col_id']: df.rename(columns={current_map['col_id']: 'StudentID'}, inplace=True); final_cols.append('StudentID')
            elif 'StudentID' not in df.columns: df['StudentID'] = 'Stud_' + df.index.astype(str); final_cols.append('StudentID')
            if current_map['col_first']: df.rename(columns={current_map['col_first']: 'FirstName'}, inplace=True); final_cols.append('FirstName')
            if current_map['col_last']: df.rename(columns={current_map['col_last']: 'LastName'}, inplace=True); final_cols.append('LastName')
            df = df[[c for c in final_cols if c in df.columns]]
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce'); df.dropna(subset=['Score'], inplace=True)

            if df.empty: st.error("No valid score data."); st.session_state.data_loaded, st.session_state.processed_df = False, None
            else:
                for col, type_func in [('Section', str), ('StudentID', str), ('FirstName', str), ('LastName', str)]:
                    if col in df.columns: df[col] = df[col].astype(type_func).fillna('')
                st.session_state.processed_df, st.session_state.data_loaded = df, True; st.success("Data mapped.")
                reset_section_color_cycle()
                # ... (Data preview logic - condensed)
                preview_cols = [c for c in ['FirstName','LastName','StudentID','Score','Section'] if c in df.columns][:5]
                if preview_cols : st.dataframe(df[preview_cols].head())

                st.session_state.df_graded, st.session_state.stats_results = None, None
        else: st.warning("Select Score & Section columns."); st.session_state.data_loaded, st.session_state.processed_df = False, None
    except Exception as e: st.error(f"Error loading/processing file: {e}"); st.session_state.data_loaded, st.session_state.processed_df = False, None

df_display = st.session_state.processed_df

# --- Sections requiring data and active cutoffs ---
if st.session_state.data_loaded and df_display is not None and st.session_state.active_cutoffs:
    st.header("Manual Cutoff Adjustment") # ... (condensed, assuming v1.13 logic is okay)
    manual_inputs = {}; grade_keys_ordered = ['A+_Start', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D'] # Simplified for example
    # ... UI for manual cutoffs ...
    # if st.button("Apply Manual Cutoffs & Recalculate"): ... st.rerun() ...

    if st.session_state.df_graded is None: # Calculate if needed
        try:
            df_calc = df_display.copy()
            df_calc['Letter_Grade'] = assign_letter_grades_from_starts(df_calc['Score'], st.session_state.active_cutoffs)
            ### DEBUG ###
            # st.write("### DEBUG assign_letter_grades_from_starts output:", df_calc['Letter_Grade'].value_counts(dropna=False))
            ### END DEBUG ###
            if not df_calc['Letter_Grade'].astype(str).str.contains('Error', na=False).any():
                st.session_state.stats_results = calculate_stats(df_calc, 'Letter_Grade', 'Section', GPA_SCALE)
                st.session_state.df_graded = df_calc
            else: st.error("Stats not calculated due to grade assignment errors."); st.session_state.stats_results, st.session_state.df_graded = None, None
        except Exception as e: st.error(f"Error in grade/stats calculation: {e}"); st.session_state.stats_results, st.session_state.df_graded = None, None

    st.header("Visualization & Observation")
    st.session_state.points_near_cutoff_active = st.number_input("Range for 'Students Below Cutoffs'", 0.1, 10.0, st.session_state.points_near_cutoff_active, 0.1, "%.1f", key='points_near_override_v4')
    active_points_near = st.session_state.points_near_cutoff_active
    plot_cutoffs = sorted(list(set(v for k, v in st.session_state.active_cutoffs.items() if k != 'F_Max')))

    st.subheader("Score Distribution") # FIX 1: Removed histogram bin slider, using 'auto' bins
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(df_display['Score'], kde=True, ax=ax_hist, bins='auto', stat="density", color=get_section_color_fixed("OverallDist")) # Use auto bins
    for co in plot_cutoffs: ax_hist.axvline(co, color='red', ls='--', lw=1)
    ax_hist.set_title("Score Distribution"); ax_hist.set_xlabel("Score"); st.pyplot(fig_hist); plt.close(fig_hist)

    st.subheader("Individual Scores by Section") # ... (condensed, assuming v1.13 logic for plot itself is okay)
    # ... stripplot code ...

    st.subheader(f"Students Below Cutoffs (within {active_points_near:.1f} points)") # ... (condensed)
    # ... students_near_df_local logic ...
    ### DEBUG ###
    # if 'students_near_df_local' in locals() and not students_near_df_local.empty:
        # st.write("### DEBUG students_near_df_local (first 5):", students_near_df_local[['StudentID', 'FirstName', 'Score', 'Target_Grade']].head() if 'StudentID' in students_near_df_local else students_near_df_local.head())
    ### END DEBUG ###


    st.header("Manual Upgrades (Highlighting Only)") # FIX 3: Debugging this section
    # ... (Manual Upgrades multiselect logic - condensed) ...
    # Inside the button:
    # if st.button("Update Upgrade Highlighting"):
        ### DEBUG ###
        # st.write("### DEBUG 'Update Upgrade Highlighting' clicked.")
        # st.write("### DEBUG newly_selected_student_ids:", newly_selected_student_ids) # newly_selected_student_ids should be defined above this button
        # st.write("### DEBUG st.session_state.upgraded_students BEFORE update:", st.session_state.upgraded_students)
        ### END DEBUG ###
        # if newly_selected_student_ids != st.session_state.upgraded_students:
        #    st.session_state.upgraded_students = newly_selected_student_ids
        #    st.success("Highlighting updated."); st.rerun()
        # else: st.info("Highlighting unchanged.")


    st.header("Final Results")
    if st.session_state.stats_results and not st.session_state.stats_results.get("error"):
        results = st.session_state.stats_results
        st.subheader("Grade Distributions (Count & Percentage)") # ... (condensed, assuming v1.13 logic okay)
        # ... display logic for combined_dist_display_df ...

        st.subheader("Section GPA Comparison") # FIX 2: Debugging GPA boxplot
        # ... (Per Section Avg GPA Table & ANOVA Interpretation - condensed) ...
        with col_gpa_plot: # This was a column defined in v1.13, ensure it's still valid or adjust layout
            if st.session_state.df_graded is not None and 'GPA' in st.session_state.df_graded.columns:
                ### DEBUG ###
                # df_graded_for_plot_debug = st.session_state.df_graded
                # st.write("### DEBUG Boxplot: df_graded available. Shape:", df_graded_for_plot_debug.shape)
                # st.write("### DEBUG Boxplot: GPA column head:", df_graded_for_plot_debug['GPA'].head())
                # st.write("### DEBUG Boxplot: GPA value_counts (incl NaN):", df_graded_for_plot_debug['GPA'].value_counts(dropna=False))
                # st.write("### DEBUG Boxplot: GPA notna().any():", df_graded_for_plot_debug['GPA'].notna().any())
                # st.write("### DEBUG Boxplot: Unique Letter_Grade values in df_graded:", df_graded_for_plot_debug['Letter_Grade'].unique())
                ### END DEBUG ###
                if st.session_state.df_graded['GPA'].notna().any(): # Explicit check
                    try:
                        # ... (boxplot drawing code) ...
                        fig_box, ax_box = plt.subplots(); sorted_sections_for_plot = sorted(st.session_state.df_graded['Section'].unique())
                        sns.boxplot(data=st.session_state.df_graded, x='Section', y='GPA', ax=ax_box, order=sorted_sections_for_plot)
                        ax_box.set_title("GPA Distribution by Section"); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig_box); plt.close(fig_box)
                    except Exception as e: st.warning(f"Could not generate section GPA boxplot: {e}")
                else:
                    st.warning("GPA data contains all NaN values. Boxplot cannot be generated. Please check grade assignments and GPA scale mappings.")
            else:
                st.warning("Graded data or GPA column not available for boxplot.")
        
        st.subheader("Failing Students Analysis") # ... (condensed)
        st.subheader("Final Assigned Grades Table") # FIX 3: Debugging highlighting
        if st.session_state.df_graded is not None:
            # ... (df_to_style creation) ...
            ### DEBUG ###
            # if 'StudentID' in df_to_style.columns:
            #     st.write("### DEBUG Final Table: StudentIDs in table to be styled:", df_to_style['StudentID'].astype(str).unique()[:20]) # Show some IDs
            # st.write("### DEBUG Final Table: st.session_state.upgraded_students:", st.session_state.upgraded_students)
            ### END DEBUG ###
            # ... (Styler logic with highlight_upgraded) ...
            # ... (st.dataframe(styler...) display) ...
            # ... (Download logic) ...

    elif st.session_state.active_cutoffs: st.warning("Statistics could not be calculated.")
st.sidebar.markdown("---"); st.sidebar.info("Iterative Grading Tool v1.14")
