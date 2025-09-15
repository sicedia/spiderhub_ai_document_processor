import os
import glob
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 0) Asegúrate de que exista la carpeta de salida
out_dir = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(out_dir, exist_ok=True)

# helper para guardar siempre en charts/
def save(fig, filename):
    path = os.path.join(out_dir, filename)
    fig.write_html(path)
    print(f"Saved: {path}")

# 1. Load all JSON reports
reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
files = glob.glob(os.path.join(reports_dir, '*.json'))
records = []
for f in files:
    with open(f, encoding='utf-8') as r:
        data = json.load(r)
        # flatten extra_data into top‐level
        em = data.get('extra_data', {})
        rec = {
            'title': data.get('title'),
            'start_date': em.get('start_date'),
            'end_date': em.get('end_date'),
            'lead_country': em.get('lead_country_iso'),
            'agreement_type': em.get('agreement_type'),
            'budget': em.get('budget_amount_eur', 0.0),
            'impl_degree': em.get('implementation_degree_pct', 0.0),
            'actionability': em.get('actionability_score', 0.0),
            'fin_source': em.get('financing_source'),
            'fin_instr': em.get('financing_instrument'),
            'country_list': em.get('country_list_iso', []),
            'beneficiaries': em.get('beneficiary_group', []),
            'themes': list(data.get('themes', {}).keys()),
            'actor_types': list(data.get('actors', {}).keys()),
            'eu_policies': em.get('eu_policy_alignment', []),
            'sdg_alignment': em.get('sdg_alignment', []),
            'commitment_details': em.get('commitment_details', []),
            'legal_bindingness': em.get('legal_bindingness'),
            'coverage_scope': em.get('coverage_scope'),
            'review_schedule': em.get('review_schedule')
        }
        records.append(rec)
df = pd.DataFrame(records)

# Normalize beneficiary dicts into separate lists
df['beneficiary_categories'] = df['beneficiaries'].apply(lambda bs: [b.get('category') for b in bs])
df['beneficiary_labels']     = df['beneficiaries'].apply(lambda bs: [b.get('label')    for b in bs])

# Add: preserve original row index for later joins
df = df.reset_index().rename(columns={'index':'idx'})

# explode lists (keep idx around)
df_countries = df.explode('country_list')
df_bens_cat  = df.explode('beneficiary_categories')
df_bens_lbl  = df.explode('beneficiary_labels')
df_themes    = df.explode('themes')
df_pols      = df.explode('eu_policies')

# build commitments DataFrame
cd = pd.json_normalize(df.to_dict(orient='records'), 'commitment_details', ['title'])
cd.rename(columns={'text':'commitment','commitment_class':'class','implementation_status':'status'}, inplace=True)

# Chart 1: Choropleth of lead_country
fig1 = px.choropleth(df, locations='lead_country', color='lead_country',
                     title='Map of Lead Countries')
save(fig1, 'chart1_choropleth.html')

# Chart 2: Gantt / timeline
# Filter out rows with invalid or placeholder dates
valid_dates_mask = (
    pd.to_datetime(df['start_date'], errors='coerce').notna() & 
    pd.to_datetime(df['end_date'], errors='coerce').notna() &
    ~df['start_date'].str.contains('YYYY', na=False) &
    ~df['end_date'].str.contains('YYYY', na=False)
)
df_timeline = df[valid_dates_mask].copy()

if not df_timeline.empty:
    # Convert agreement_type list to string for visualization
    df_timeline['agreement_type_str'] = df_timeline['agreement_type'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) and x else 'Unknown'
    )
    
    fig2 = px.timeline(df_timeline, x_start='start_date', x_end='end_date',
                       y='agreement_type_str', color='agreement_type_str',
                       title='Timeline of Agreements')
    fig2.update_yaxes(autorange="reversed")
    save(fig2, 'chart2_timeline.html')
else:
    print("No valid dates found for timeline chart")

# Chart 3: Bar Plot of Commitment Class Counts
fig3 = px.histogram(cd, x='class',
                    title='Commitment Class Counts')
save(fig3, 'chart3_commitment_counts.html')

# Chart 4: Scatter Budget vs Implementation Degree
fig4 = px.scatter(df, x='budget', y='impl_degree', size='actionability',
                  hover_data=['title'],
                  title='Budget vs Implementation Degree (size=Actionability)')
save(fig4, 'chart4_budget_vs_impl.html')

# Chart 5: Sankey Financing Source → Instrument → Commitment Class
# -------------------------------------------------------------------
# build flattened commitment DataFrame with title → class
cd = pd.json_normalize(
    df.to_dict(orient='records'),
    'commitment_details',
    ['title']
)
cd.rename(
    columns={
        'text': 'commitment',
        'commitment_class': 'class',
        'implementation_status': 'status'
    },
    inplace=True
)

# collect all unique node labels
sources_set = set(df['fin_source'].dropna())
instrs_set  = set(df['fin_instr'].dropna())
classes_set = set(cd['class'].dropna())

all_labels = list(sources_set | instrs_set | classes_set)
label_idx  = {lbl: i for i, lbl in enumerate(all_labels)}

# build links
sources = []
targets = []
values  = []

# Source → Instrument
for _, row in df.dropna(subset=['fin_source','fin_instr']).iterrows():
    s = label_idx[row['fin_source']]
    t = label_idx[row['fin_instr']]
    sources.append(s)
    targets.append(t)
    values.append(1)

# Instrument → Commitment Class
for _, row in cd.dropna(subset=['class']).iterrows():
    instr = df.loc[df['title'] == row['title'], 'fin_instr']
    if not instr.empty and pd.notna(instr.iloc[0]):
        s = label_idx[instr.iloc[0]]
        t = label_idx[row['class']]
        sources.append(s)
        targets.append(t)
        values.append(1)

fig5 = go.Figure(data=[go.Sankey(
    node  = dict(label=all_labels),
    link  = dict(source=sources, target=targets, value=values)
)])
fig5.update_layout(
    title_text="Sankey: Financing Source → Instrument → Commitment Class",
    font_size=10
)
save(fig5, 'chart5_sankey.html')

# Chart 6: Heatmap Theme × Beneficiary Category
df_theme    = df[['idx','themes']].explode('themes')
df_ben_cat  = df[['idx','beneficiary_categories']].explode('beneficiary_categories') \
                   .rename(columns={'beneficiary_categories':'beneficiary_category'})
pairs       = df_theme.merge(df_ben_cat, on='idx')
th_be       = pd.crosstab(pairs['themes'], pairs['beneficiary_category'])
fig6 = px.imshow(th_be, title='Heatmap: Theme vs Beneficiary Category')
save(fig6, 'chart6_heatmap.html')

# Chart 7: Radar of SDG Alignment
df_sdg = df.explode('sdg_alignment')
sdg_counts = df_sdg['sdg_alignment'].value_counts().reset_index()
sdg_counts.columns = ['sdg','count']
fig7 = px.line_polar(sdg_counts, r='count', theta='sdg', line_close=True,
                    title='SDG Alignment Distribution',
                    template='plotly_dark')
save(fig7, 'chart7_sdg_radar.html')


# Chart 8: Treemap of country_list per agreement
# Filter out rows with empty or null country_list
df_countries_filtered = df_countries[
    df_countries['country_list'].notna() & 
    (df_countries['country_list'] != '')
]

if not df_countries_filtered.empty:
    fig8 = px.treemap(df_countries_filtered, path=['title','country_list'], 
                      title='Treemap of Countries per Agreement')
    save(fig8, 'chart8_treemap.html')
else:
    print("No valid country data found for treemap chart")

# Chart 9: Bar of Beneficiary Category (count)
ben_count = df_bens_cat['beneficiary_categories'] \
             .value_counts().reset_index()
ben_count.columns = ['beneficiary_category','count']
fig9 = px.bar(ben_count,
              x='beneficiary_category',
              y='count',
              title='Beneficiary Category Counts')
save(fig9, 'chart9_beneficiary_bar.html')

# Chart 10: Time series of # of agreements & total budget per year
df['year'] = pd.to_datetime(df['start_date'], errors='coerce').dt.year
ts = df.groupby('year').agg({'title':'count','budget':'sum'}).reset_index()
fig10 = px.line(ts, x='year', y=['title','budget'], markers=True,
                labels={'value':'Count/Budget','variable':'Metric'},
                title='Agreements & Budget Over Time')
save(fig10, 'chart10_timeseries.html')

# Chart 11: Bar of Agreements by Main Theme
theme_counts = df_themes['themes'].value_counts().reset_index()
theme_counts.columns = ['theme','count']
fig11 = px.bar(theme_counts, x='theme', y='count',
               title='Number of Agreements by Main Theme')
save(fig11, 'chart11_theme_bar.html')

# Chart 12: Bar of Agreements by Actor Type
df_actor_types = df.explode('actor_types')
actor_counts = df_actor_types['actor_types'].value_counts().reset_index()
actor_counts.columns = ['actor_type','count']
fig12 = px.bar(actor_counts, x='actor_type', y='count',
               title='Number of Agreements by Actor Type')
save(fig12, 'chart12_actor_bar.html')

# Chart 13: Sankey Theme → Actor Type
# replace the faulty merge/select with an explicit join of only the needed cols
pairs_ta = df_themes[['idx','themes']] \
    .merge(df_actor_types[['idx','actor_types']], on='idx')

link_df = pairs_ta.groupby(['themes','actor_types']) \
    .size().reset_index(name='value')

labels = list(link_df['themes'].unique()) + list(link_df['actor_types'].unique())
label_idx = {lbl: i for i, lbl in enumerate(labels)}

sources = link_df['themes'].map(label_idx)
targets = link_df['actor_types'].map(label_idx)
values  = link_df['value']

fig13 = go.Figure(data=[go.Sankey(
    node=dict(label=labels),
    link=dict(source=sources, target=targets, value=values)
)])
fig13.update_layout(title_text='Sankey: Theme → Actor Type', font_size=10)
save(fig13, 'chart13_theme_actor_sankey.html')

# Chart 14: Pie of Legal Bindingness
fig14 = px.pie(df, names='legal_bindingness',
               title='Legal Bindingness Distribution')
save(fig14, 'chart14_legal_bindingness_pie.html')

# Chart 15: Bar of Coverage Scope
scope_counts = df['coverage_scope'].value_counts().reset_index()
scope_counts.columns = ['coverage_scope','count']
fig15 = px.bar(scope_counts, x='coverage_scope', y='count',
               title='Coverage Scope of Agreements')
save(fig15, 'chart15_coverage_scope_bar.html')

# Chart 16: Timeline by Review Schedule
# Apply the same date filtering as Chart 2
df_review_timeline = df[valid_dates_mask].copy()

if not df_review_timeline.empty:
    fig16 = px.timeline(df_review_timeline, x_start='start_date', x_end='end_date',
                        y='review_schedule', color='review_schedule',
                        title='Timeline of Agreements by Review Schedule')
    fig16.update_yaxes(autorange="reversed")
    save(fig16, 'chart16_review_timeline.html')
else:
    print("No valid dates found for review schedule timeline chart")

# Chart 17: Heatmap of Lead Countries
# Count agreements by lead country
country_counts = df['lead_country'].value_counts().reset_index()
country_counts.columns = ['country', 'count']

# Create a matrix format for heatmap (single row)
heatmap_data = country_counts.set_index('country').T

fig17 = px.imshow(heatmap_data, 
                  title='Heatmap: Number of Agreements by Lead Country',
                  labels=dict(x="Country", y="Metric", color="Count"),
                  aspect="auto")
fig17.update_layout(
    xaxis_title="Lead Country",
    yaxis_title="Agreement Count"
)
save(fig17, 'chart17_country_heatmap.html')

if __name__ == '__main__':
    for i in range(1,18):
        print(f'Chart {i} saved as charts/chart{i}_*.html')