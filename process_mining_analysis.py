import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import timedelta
from deep_translator import GoogleTranslator
import networkx as nx

# Loading Data
# Import XES log format to extract structured event data, validate schema completeness to ensure 
# all case IDs, activities, timestamps, and resources are properly captured for downstream process discovery.
log = pm4py.read_xes('financial_log.xes')

print(f"Type of log object: {type(log)}")
print(f"\nFirst 10 events for sanity check:")
print(log.head(10))

# Basic info
print(f"\nTotal events in log: {len(log)}")
print(f"Columns available: {log.columns.tolist()}")

# Data Exploration
# Understand activity taxonomy, temporal distribution, and baseline metrics, translate Dutch activity 
# labels to English to improve interpretability and create a reference mapping for stakeholders.
# Initialize language translator
translator = GoogleTranslator(source='nl', target='en')

# Total number of cases
num_cases = log['case:concept:name'].nunique()
print(f"\nTotal unique loan applications: {num_cases}")

# Total number of events
num_events = len(log)
print(f"Total events: {num_events}")

# Different types of activities (with translation)
activities = log['concept:name'].unique()
print(f"\nTotal unique activities: {len(activities)}")
print("\nAll activities (with translations):")

# Creating a translation dictionary
activity_translations = {}
for activity in sorted(activities):
    if ' ' in activity and not activity.startswith(('A_', 'O_')):
        try:
            translation = translator.translate(activity)
            activity_translations[activity] = translation
        except Exception as e:
            print(f" Translation failed for '{activity}': {e}")
            activity_translations[activity] = activity
    else:
        activity_translations[activity] = activity

# Print activities with counts and translations
for i, activity in enumerate(sorted(activities), 1):
    count = len(log[log['concept:name'] == activity])
    translated = activity_translations[activity]
    if activity != translated:
        print(f"{i:2d}. {activity:40s} → {translated:30s} ({count:6d} occurrences)")
    else:
        print(f"{i:2d}. {activity:40s} ({count:6d} occurrences)")

# Case outcomes - find terminal activities
print("\nCase Outcomes (Terminal Activities):")
terminal_activities = log.groupby('case:concept:name')['concept:name'].last()
outcome_counts = terminal_activities.value_counts()

print("\nOutcome distribution:")
for activity, count in outcome_counts.items():
    translated = activity_translations.get(activity, activity)
    if activity != translated:
        print(f"  {activity:40s} → {translated:30s}: {count:5d} cases")
    else:
        print(f"  {activity:40s}: {count:5d} cases")

# Visualizing process map (Directly-Follows Graph)
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.save(gviz, "process_map_dfg.png")
print("DFG saved as 'process_map_dfg.png'")

# Save translations to file for reference
translations_df = pd.DataFrame(list(activity_translations.items()), 
                               columns=['Dutch Activity', 'English Translation'])
translations_df.to_csv('activity_translations.csv', index=False)
print("Translations saved to 'activity_translations.csv'")

# Create interactive version of DFG using network graph visualization to enable exploration of process 
# flows with hover details, zoom capabilities, and dynamic filtering for improved stakeholder engagement.
start_activities = pm4py.get_start_activities(log)
end_activities = pm4py.get_end_activities(log)

# Extract nodes and edges from DFG
nodes = set()
edges = []
for (source, target), freq in dfg.items():
    nodes.add(source)
    nodes.add(target)
    edges.append({
        'source': source,
        'target': target,
        'frequency': freq
    })

# Add start and end markers
for activity, freq in start_activities.items():
    nodes.add(activity)
    edges.append({
        'source': 'START',
        'target': activity,
        'frequency': freq
    })
nodes.add('START')

for activity, freq in end_activities.items():
    nodes.add(activity)
    edges.append({
        'source': activity,
        'target': 'END',
        'frequency': freq
    })
nodes.add('END')

# Create network graph for layout
G = nx.DiGraph()
for node in nodes:
    G.add_node(node)
for edge in edges:
    G.add_edge(edge['source'], edge['target'], weight=edge['frequency'])

# Use hierarchical layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Create edge traces
edge_traces = []
for edge in edges:
    x0, y0 = pos[edge['source']]
    x1, y1 = pos[edge['target']]
    
    # Calculate edge width based on frequency
    max_freq = max([e['frequency'] for e in edges])
    edge_width = 1 + (edge['frequency'] / max_freq) * 5
    
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode='lines',
        line=dict(width=edge_width, color='#888'),
        hoverinfo='text',
        text=f"{edge['source']} → {edge['target']}<br>Frequency: {edge['frequency']}",
        showlegend=False
    )
    edge_traces.append(edge_trace)

# Create node trace
node_x = []
node_y = []
node_text = []
node_colors = []
node_sizes = []

for node in nodes:
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    
    # Get node frequency (sum of incoming edges)
    node_freq = sum([e['frequency'] for e in edges if e['target'] == node])
    node_sizes.append(20 + (node_freq / max([sum([e['frequency'] for e in edges if e['target'] == n]) for n in nodes]) * 30))
    
    # Color coding
    if node == 'START':
        node_colors.append('lightgreen')
    elif node == 'END':
        node_colors.append('lightcoral')
    else:
        node_colors.append('lightblue')
    
    # Add translation to hover text
    translated = activity_translations.get(node, node)
    if translated != node:
        node_text.append(f"{node}<br>{translated}<br>Frequency: {node_freq}")
    else:
        node_text.append(f"{node}<br>Frequency: {node_freq}")

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[n[:15] + '...' if len(n) > 15 else n for n in nodes],
    hovertext=node_text,
    textposition="top center",
    marker=dict(
        size=node_sizes,
        color=node_colors,
        line=dict(width=2, color='#333')
    ),
    showlegend=False
)

# Create figure
fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=dict(text='Interactive Process Map (Directly-Follows Graph)', font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                ))

fig.write_html('process_map_dfg.html')
print("Interactive DFG saved as 'process_map_dfg.html'")

# Save translations to file for reference
translations_df = pd.DataFrame(list(activity_translations.items()), 
                               columns=['Dutch Activity', 'English Translation'])
translations_df.to_csv('activity_translations.csv', index=False)
print("Translations saved to 'activity_translations.csv'")

# Throughput time analysis
# Identify bottlenecks and process efficienc, measure cycle time distribution and detect outlier cases 
# that may indicate process deviations or compliance risks requiring investigation.
# Calculating case durations
case_durations = log.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
case_durations['duration'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / 86400

print(f"\nCase Duration Statistics (in days):")
print(f"  Mean duration:   {case_durations['duration'].mean():.2f} days")
print(f"  Median duration: {case_durations['duration'].median():.2f} days")
print(f"  Min duration:    {case_durations['duration'].min():.2f} days")
print(f"  Max duration:    {case_durations['duration'].max():.2f} days")
print(f"  Std deviation:   {case_durations['duration'].std():.2f} days")

# Identifying outliers (using IQR)
Q1 = case_durations['duration'].quantile(0.25)
Q3 = case_durations['duration'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
outliers = case_durations[case_durations['duration'] > outlier_threshold]

print(f"\nOutlier Analysis:")
print(f"  Q1 (25th percentile): {Q1:.2f} days")
print(f"  Q3 (75th percentile): {Q3:.2f} days")
print(f"  IQR: {IQR:.2f} days")
print(f"  Outlier threshold (Q3 + 1.5*IQR): {outlier_threshold:.2f} days")
print(f"  Number of outlier cases: {len(outliers)} ({len(outliers)/len(case_durations)*100:.1f}%)")

# Visualize distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(case_durations['duration'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(case_durations['duration'].mean(), color='red', linestyle='--', label=f'Mean: {case_durations["duration"].mean():.1f}d')
axes[0].axvline(case_durations['duration'].median(), color='green', linestyle='--', label=f'Median: {case_durations["duration"].median():.1f}d')
axes[0].set_xlabel('Duration (days)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Case Durations')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].boxplot(case_durations['duration'], vert=True)
axes[1].set_ylabel('Duration (days)')
axes[1].set_title('Box Plot of Case Durations')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('throughput_time_analysis.png', dpi=300, bbox_inches='tight')
print("Throughput time visualization saved as 'throughput_time_analysis.png'")
plt.close()

# Interactive version
fig = go.Figure()
fig.add_trace(go.Histogram(x=case_durations['duration'], nbinsx=50, name='Duration'))
fig.add_vline(x=case_durations['duration'].mean(), line_dash="dash", line_color="red", 
              annotation_text=f"Mean: {case_durations['duration'].mean():.1f}d")
fig.add_vline(x=case_durations['duration'].median(), line_dash="dash", line_color="green",
              annotation_text=f"Median: {case_durations['duration'].median():.1f}d")
fig.update_layout(title='Distribution of Case Durations', xaxis_title='Duration (days)', yaxis_title='Frequency')
fig.write_html('throughput_time_analysis.html')
print("Interactive throughput time saved as 'throughput_time_analysis.html'")

# Rework Detection
# Quantify process inefficiencies by tracking repeated activities, identify rework hotspots that 
# correlate with decision points or validation failures to target process improvement initiatives.
activity_counts_per_case = log.groupby(['case:concept:name', 'concept:name']).size().reset_index(name='count')
rework_cases = activity_counts_per_case[activity_counts_per_case['count'] > 1]

rework_stats = rework_cases.groupby('concept:name').agg({
    'count': ['sum', 'mean', 'max'],
    'case:concept:name': 'count'
}).reset_index()
rework_stats.columns = ['activity', 'total_repetitions', 'avg_repetitions', 'max_repetitions', 'num_cases_with_rework']
rework_stats = rework_stats.sort_values('num_cases_with_rework', ascending=False)

print(f"\nRework Analysis:")
print(f"  Total activities: {log['concept:name'].nunique()}")
print(f"  Activities with rework: {len(rework_stats)}")
print(f"\nTop 10 Activities with Most Rework:\n")

for idx, row in rework_stats.head(10).iterrows():
    activity = row['activity']
    translated = activity_translations.get(activity, activity) 
    
    print(f"  {activity:40s}" + (f" → {translated}" if translated != activity else ""))
    print(f"    Cases with rework: {row['num_cases_with_rework']:5d} | Total repetitions: {row['total_repetitions']:6.0f} | "
          f"Avg: {row['avg_repetitions']:.2f} | Max: {row['max_repetitions']:.0f}")
    print()

top_rework = rework_stats.head(10)
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_rework)), top_rework['num_cases_with_rework'], color='coral', edgecolor='black')
plt.yticks(range(len(top_rework)), top_rework['activity'])
plt.xlabel('Number of Cases with Rework')
plt.title('Top 10 Activities with Most Rework')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rework_analysis.png', dpi=300, bbox_inches='tight')
print("Rework visualization saved as 'rework_analysis.png'")
plt.close()

# Interactive version
fig = px.bar(top_rework, x='num_cases_with_rework', y='activity', orientation='h',
             title='Top 10 Activities with Most Rework', labels={'num_cases_with_rework': 'Cases with Rework'})
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
fig.write_html('rework_analysis.html')
print("Interactive rework visualization saved as 'rework_analysis.html'")

print(f"Rework Intensity:")
total_rework_per_case = rework_cases.groupby('case:concept:name')['count'].apply(lambda x: (x - 1).sum())
print(f"  Cases with ANY rework: {total_rework_per_case.count()} ({total_rework_per_case.count()/num_cases*100:.1f}%)")
print(f"  Average rework events per case (for cases with rework): {total_rework_per_case.mean():.2f}")
print(f"  Max rework events in a single case: {total_rework_per_case.max():.0f}")

# Comparing approved vs declined vs cancelled cases
# Segment process variants by outcome to uncover distinct pathways, compare throughput times and 
# rework patterns across outcomes to identify risk factors and process characteristics that drive decisions.
approved_cases = log[log['concept:name'].isin(['A_APPROVED', 'A_ACTIVATED'])]['case:concept:name'].unique()
declined_cases = terminal_activities[terminal_activities.isin(['A_DECLINED'])].index
cancelled_cases = terminal_activities[terminal_activities.isin(['A_CANCELLED', 'O_CANCELLED'])].index

approved_cases = np.array([c for c in approved_cases if c not in declined_cases and c not in cancelled_cases])

print(f"\nCase Classification:")
print(f"  Total cases: {num_cases}")
print(f"  Approved cases: {len(approved_cases)} ({len(approved_cases)/num_cases*100:.1f}%)")
print(f"  Declined cases: {len(declined_cases)} ({len(declined_cases)/num_cases*100:.1f}%)")
print(f"  Cancelled cases: {len(cancelled_cases)} ({len(cancelled_cases)/num_cases*100:.1f}%)")
print(f"  Other outcomes: {num_cases - len(approved_cases) - len(declined_cases) - len(cancelled_cases)}")

approved_log = log[log['case:concept:name'].isin(approved_cases)]
declined_log = log[log['case:concept:name'].isin(declined_cases)]
cancelled_log = log[log['case:concept:name'].isin(cancelled_cases)]

print(f"\n  Approved log: {len(approved_log)} events")
print(f"  Declined log: {len(declined_log)} events")
print(f"  Cancelled log: {len(cancelled_log)} events")

# Visualize with Inductive Miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer

tree_approved = inductive_miner.apply(approved_log)
gviz_approved = pt_visualizer.apply(tree_approved)
pt_visualizer.save(gviz_approved, "approved_cases_process_model.png")
print("Approved cases process model saved as 'approved_cases_process_model.png'")

tree_declined = inductive_miner.apply(declined_log)
gviz_declined = pt_visualizer.apply(tree_declined)
pt_visualizer.save(gviz_declined, "declined_cases_process_model.png")
print("Declined cases process model saved as 'declined_cases_process_model.png'")

tree_cancelled = inductive_miner.apply(cancelled_log)
gviz_cancelled = pt_visualizer.apply(tree_cancelled)
pt_visualizer.save(gviz_cancelled, "cancelled_cases_process_model.png")
print("Cancelled cases process model saved as 'cancelled_cases_process_model.png'")

# Compare throughput times
approved_durations = case_durations.loc[approved_cases, 'duration']
declined_durations = case_durations.loc[declined_cases, 'duration']
cancelled_durations = case_durations.loc[cancelled_cases, 'duration']

print(f"\nThroughput Time Comparison:")
print(f"\n  Approved Cases:")
print(f"    Mean: {approved_durations.mean():.2f} days")
print(f"    Median: {approved_durations.median():.2f} days")
print(f"    Std: {approved_durations.std():.2f} days")

print(f"\n  Declined Cases:")
print(f"    Mean: {declined_durations.mean():.2f} days")
print(f"    Median: {declined_durations.median():.2f} days")
print(f"    Std: {declined_durations.std():.2f} days")

print(f"\n  Cancelled Cases:")
print(f"    Mean: {cancelled_durations.mean():.2f} days")
print(f"    Median: {cancelled_durations.median():.2f} days")
print(f"    Std: {cancelled_durations.std():.2f} days")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].boxplot([approved_durations, declined_durations, cancelled_durations], 
                tick_labels=['Approved', 'Declined', 'Cancelled'])
axes[0].set_ylabel('Duration (days)')
axes[0].set_title('Throughput Time: Approved vs Declined vs Cancelled')
axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(approved_durations, bins=30, alpha=0.5, label='Approved', color='green', edgecolor='black')
axes[1].hist(declined_durations, bins=30, alpha=0.5, label='Declined', color='red', edgecolor='black')
axes[1].hist(cancelled_durations, bins=30, alpha=0.5, label='Cancelled', color='orange', edgecolor='black')
axes[1].set_xlabel('Duration (days)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Duration Distribution: Approved vs Declined vs Cancelled')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('approved_vs_declined_vs_cancelled_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison visualization saved as 'approved_vs_declined_vs_cancelled_comparison.png'")
plt.close()

# Interactive version
comparison_data = pd.DataFrame({
    'Duration': list(approved_durations) + list(declined_durations) + list(cancelled_durations),
    'Type': ['Approved']*len(approved_durations) + ['Declined']*len(declined_durations) + ['Cancelled']*len(cancelled_durations)
})
fig = px.box(comparison_data, x='Type', y='Duration', color='Type',
             title='Throughput Time: Approved vs Declined vs Cancelled',
             color_discrete_map={'Approved': 'green', 'Declined': 'red', 'Cancelled': 'orange'})
fig.write_html('approved_vs_declined_vs_cancelled_comparison.html')
print("Interactive comparison saved as 'approved_vs_declined_vs_cancelled_comparison.html'")

# Compare rework
approved_rework_cases = activity_counts_per_case[
    (activity_counts_per_case['case:concept:name'].isin(approved_cases)) & 
    (activity_counts_per_case['count'] > 1)
]
declined_rework_cases = activity_counts_per_case[
    (activity_counts_per_case['case:concept:name'].isin(declined_cases)) & 
    (activity_counts_per_case['count'] > 1)
]
cancelled_rework_cases = activity_counts_per_case[
    (activity_counts_per_case['case:concept:name'].isin(cancelled_cases)) & 
    (activity_counts_per_case['count'] > 1)
]

approved_rework_rate = len(approved_rework_cases['case:concept:name'].unique()) / len(approved_cases) * 100
declined_rework_rate = len(declined_rework_cases['case:concept:name'].unique()) / len(declined_cases) * 100
cancelled_rework_rate = len(cancelled_rework_cases['case:concept:name'].unique()) / len(cancelled_cases) * 100

print(f"\nRework Comparison:")
print(f"  Approved cases with rework: {len(approved_rework_cases['case:concept:name'].unique())} ({approved_rework_rate:.1f}%)")
print(f"  Declined cases with rework: {len(declined_rework_cases['case:concept:name'].unique())} ({declined_rework_rate:.1f}%)")
print(f"  Cancelled cases with rework: {len(cancelled_rework_cases['case:concept:name'].unique())} ({cancelled_rework_rate:.1f}%)")

# Create interactive visualization for rework comparison across outcomes to enable drill-down analysis 
# of process quality metrics and identify outcome-specific inefficiency patterns requiring intervention.
rework_comparison_data = pd.DataFrame({
    'Outcome': ['Approved', 'Declined', 'Cancelled'],
    'Rework Rate (%)': [approved_rework_rate, declined_rework_rate, cancelled_rework_rate],
    'Cases with Rework': [
        len(approved_rework_cases['case:concept:name'].unique()),
        len(declined_rework_cases['case:concept:name'].unique()),
        len(cancelled_rework_cases['case:concept:name'].unique())
    ]
})

fig = go.Figure()
fig.add_trace(go.Bar(
    x=rework_comparison_data['Outcome'],
    y=rework_comparison_data['Rework Rate (%)'],
    text=rework_comparison_data['Rework Rate (%)'].round(1),
    textposition='auto',
    marker_color=['green', 'red', 'orange'],
    hovertemplate='%{x}<br>Rework Rate: %{y:.1f}%<br>Cases: %{customdata}<extra></extra>',
    customdata=rework_comparison_data['Cases with Rework']
))
fig.update_layout(
    title='Rework Rate Comparison by Outcome',
    xaxis_title='Case Outcome',
    yaxis_title='Rework Rate (%)',
    showlegend=False
)
fig.write_html('rework_comparison_by_outcome.html')
print("Interactive rework comparison saved as 'rework_comparison_by_outcome.html'")

# Resource Analysis
print(f"\nResource Analysis:")
resource_workload = log.groupby('org:resource').size().sort_values(ascending=False)
print(f"  Total unique resources: {log['org:resource'].nunique()}")
print(f"\nTop 10 Resources by Workload:")
for resource, count in resource_workload.head(10).items():
    if pd.notna(resource):
        print(f"  Resource {resource}: {count} events")

# Create interactive resource workload visualization to identify capacity constraints and workload 
# imbalances across team members, enabling data-driven resource allocation and training prioritization.
resource_df = pd.DataFrame({
    'Resource': resource_workload.head(15).index,
    'Event Count': resource_workload.head(15).values
})

fig = px.bar(resource_df, x='Resource', y='Event Count',
             title='Top 15 Resources by Workload',
             labels={'Event Count': 'Number of Events'},
             color='Event Count',
             color_continuous_scale='Blues')
fig.update_layout(xaxis_tickangle=-45)
fig.write_html('resource_workload_analysis.html')
print("Interactive resource workload saved as 'resource_workload_analysis.html'")

# Variant Analysis
from pm4py.algo.filtering.log.variants import variants_filter
variants = variants_filter.get_variants(log)
print(f"\nVariant Analysis:")
print(f"  Total unique process variants: {len(variants)}")
print(f"\nTop 5 Most Common Process Paths:")
sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)
for i, (variant, cases) in enumerate(sorted_variants[:5], 1):
    print(f"\n  Variant {i} ({len(cases)} cases):")
    print(f"    Path: {' → '.join(variant)}")

# Create interactive variant frequency distribution to visualize process conformance and complexity, 
# highlighting the concentration of cases in standard paths versus long-tail variant proliferation.
variant_counts = [(len(cases), i+1) for i, (variant, cases) in enumerate(sorted_variants)]
variant_df = pd.DataFrame(variant_counts, columns=['Case Count', 'Variant Rank'])

fig = px.scatter(variant_df, x='Variant Rank', y='Case Count',
                 title='Process Variant Distribution',
                 labels={'Variant Rank': 'Variant (Ranked by Frequency)', 'Case Count': 'Number of Cases'},
                 log_y=True)
fig.update_traces(marker=dict(size=8, color='steelblue', line=dict(width=1, color='darkblue')))
fig.write_html('variant_distribution.html')
print("Interactive variant distribution saved as 'variant_distribution.html'")

# Waiting Time Analysis
# Measure inter-activity delays to identify process queues and resource constraints, isolate 
# activities with excessive waiting times that could indicate handoff inefficiencies or approval backlogs.
print(f"\nWaiting Time Analysis:")
log_sorted = log.sort_values(['case:concept:name', 'time:timestamp'])
log_sorted['next_timestamp'] = log_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
log_sorted['waiting_time'] = (log_sorted['next_timestamp'] - log_sorted['time:timestamp']).dt.total_seconds() / 3600

waiting_times = log_sorted[log_sorted['waiting_time'].notna()]
waiting_by_activity = waiting_times.groupby('concept:name')['waiting_time'].agg(['mean', 'median', 'max']).sort_values('mean', ascending=False)

print(f"\nTop 10 Activities with Longest Waiting Times (hours):")
for activity, row in waiting_by_activity.head(10).iterrows():
    translated = activity_translations.get(activity, activity)
    print(f"  {activity:40s}")
    if translated != activity:
        print(f"    → {translated}")
    print(f"    Mean: {row['mean']:.2f}h | Median: {row['median']:.2f}h | Max: {row['max']:.2f}h")

# Create interactive waiting time visualization to enable identification of bottleneck activities 
# and queue buildup patterns, supporting targeted interventions to reduce cycle time variability.
waiting_top = waiting_by_activity.head(15).reset_index()
waiting_top['Activity'] = waiting_top['concept:name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=waiting_top['Activity'],
    y=waiting_top['mean'],
    name='Mean',
    marker_color='indianred',
    error_y=dict(
        type='data',
        symmetric=False,
        array=waiting_top['max'] - waiting_top['mean'],
        arrayminus=waiting_top['mean'] - waiting_top['median']
    )
))
fig.update_layout(
    title='Top 15 Activities by Waiting Time',
    xaxis_title='Activity',
    yaxis_title='Mean Waiting Time (hours)',
    xaxis_tickangle=-45
)
fig.write_html('waiting_time_analysis.html')
print("Interactive waiting time analysis saved as 'waiting_time_analysis.html'")

# Activity Duration Analysis
# Measure execution time per activity to establish performance baselines, identify activities with 
# high variance that may indicate process complexity, skill gaps, or inconsistent execution patterns.
print(f"\nActivity Duration Analysis:")
activity_start = log[log['lifecycle:transition'] == 'START'][['case:concept:name', 'concept:name', 'time:timestamp']].copy()
activity_complete = log[log['lifecycle:transition'] == 'COMPLETE'][['case:concept:name', 'concept:name', 'time:timestamp']].copy()

activity_start.columns = ['case:concept:name', 'concept:name', 'start_time']
activity_complete.columns = ['case:concept:name', 'concept:name', 'complete_time']

activity_durations = pd.merge(activity_start, activity_complete, on=['case:concept:name', 'concept:name'])
activity_durations['duration'] = (activity_durations['complete_time'] - activity_durations['start_time']).dt.total_seconds() / 60

duration_stats = activity_durations.groupby('concept:name')['duration'].agg(['mean', 'median', 'max', 'std']).sort_values('mean', ascending=False)

print(f"\nTop 10 Activities by Duration (minutes):")
for activity, row in duration_stats.head(10).iterrows():
    translated = activity_translations.get(activity, activity)
    print(f"  {activity:40s}")
    if translated != activity:
        print(f"    → {translated}")
    print(f"    Mean: {row['mean']:.2f}m | Median: {row['median']:.2f}m | Max: {row['max']:.2f}m")

# Create interactive activity duration visualization with variance metrics to highlight performance 
# inconsistencies and training opportunities, enabling process standardization and efficiency gains.
duration_top = duration_stats.head(15).reset_index()
duration_top['Activity'] = duration_top['concept:name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=duration_top['Activity'],
    y=duration_top['mean'],
    name='Mean Duration',
    marker_color='steelblue',
    error_y=dict(
        type='data',
        array=duration_top['std'],
        visible=True
    ),
    hovertemplate='%{x}<br>Mean: %{y:.2f}m<br>Std: %{customdata:.2f}m<extra></extra>',
    customdata=duration_top['std']
))
fig.update_layout(
    title='Top 15 Activities by Duration (with Standard Deviation)',
    xaxis_title='Activity',
    yaxis_title='Mean Duration (minutes)',
    xaxis_tickangle=-45
)
fig.write_html('activity_duration_analysis.html')
print("Interactive activity duration analysis saved as 'activity_duration_analysis.html'")

# Time-based Patterns
# Analyze temporal correlations between submission timing (day/hour) and approval outcomes, detect 
# potential fairness issues or resource availability effects on decision consistency across time periods.
print(f"\nTime-based Patterns:")
log['submission_date'] = pd.to_datetime(log.groupby('case:concept:name')['time:timestamp'].transform('min'))
log['day_of_week'] = log['submission_date'].dt.day_name()
log['hour_of_day'] = log['submission_date'].dt.hour

case_outcomes = log.groupby('case:concept:name').agg({
    'day_of_week': 'first',
    'hour_of_day': 'first'
}).reset_index()

case_outcomes['outcome'] = case_outcomes['case:concept:name'].map(
    lambda x: 'Approved' if x in approved_cases else 
              ('Declined' if x in declined_cases else 
               ('Cancelled' if x in cancelled_cases else 'Other'))
)

print(f"\nApproval Rate by Day of Week:")
day_analysis = case_outcomes.groupby('day_of_week')['outcome'].apply(
    lambda x: (x == 'Approved').sum() / len(x) * 100
).sort_values(ascending=False)
for day, rate in day_analysis.items():
    print(f"  {day}: {rate:.1f}% approval rate")

print(f"\nApproval Rate by Hour of Day:")
hour_analysis = case_outcomes.groupby('hour_of_day')['outcome'].apply(
    lambda x: (x == 'Approved').sum() / len(x) * 100
).sort_values(ascending=False).head(5)
for hour, rate in hour_analysis.items():
    print(f"  Hour {hour:02d}: {rate:.1f}% approval rate")

# Create interactive temporal pattern visualizations to reveal time-dependent approval biases and 
# resource availability impacts, supporting fairness audits and capacity planning initiatives.
# Day of week analysis
day_outcome_counts = case_outcomes.groupby(['day_of_week', 'outcome']).size().reset_index(name='count')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_outcome_counts['day_of_week'] = pd.Categorical(day_outcome_counts['day_of_week'], categories=day_order, ordered=True)
day_outcome_counts = day_outcome_counts.sort_values('day_of_week')

fig = px.bar(day_outcome_counts, x='day_of_week', y='count', color='outcome',
             title='Case Outcomes by Day of Week',
             labels={'day_of_week': 'Day of Week', 'count': 'Number of Cases'},
             color_discrete_map={'Approved': 'green', 'Declined': 'red', 'Cancelled': 'orange', 'Other': 'gray'})
fig.write_html('outcomes_by_day_of_week.html')
print("Interactive day of week analysis saved as 'outcomes_by_day_of_week.html'")

# Hour of day analysis
hour_outcome_counts = case_outcomes.groupby(['hour_of_day', 'outcome']).size().reset_index(name='count')

fig = px.bar(hour_outcome_counts, x='hour_of_day', y='count', color='outcome',
             title='Case Outcomes by Hour of Day',
             labels={'hour_of_day': 'Hour of Day', 'count': 'Number of Cases'},
             color_discrete_map={'Approved': 'green', 'Declined': 'red', 'Cancelled': 'orange', 'Other': 'gray'})
fig.update_xaxes(dtick=1)
fig.write_html('outcomes_by_hour_of_day.html')
print("Interactive hour of day analysis saved as 'outcomes_by_hour_of_day.html'")

# Approval rate heatmap by day and hour
approval_heatmap = case_outcomes.groupby(['day_of_week', 'hour_of_day'])['outcome'].apply(
    lambda x: (x == 'Approved').sum() / len(x) * 100 if len(x) > 0 else 0
).reset_index(name='approval_rate')
approval_heatmap['day_of_week'] = pd.Categorical(approval_heatmap['day_of_week'], categories=day_order, ordered=True)
approval_heatmap = approval_heatmap.sort_values('day_of_week')

pivot_data = approval_heatmap.pivot(index='day_of_week', columns='hour_of_day', values='approval_rate')

fig = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=pivot_data.columns,
    y=pivot_data.index,
    colorscale='RdYlGn',
    text=pivot_data.values.round(1),
    texttemplate='%{text}%',
    textfont={"size": 10},
    colorbar=dict(title="Approval Rate (%)")
))
fig.update_layout(
    title='Approval Rate Heatmap: Day of Week vs Hour of Day',
    xaxis_title='Hour of Day',
    yaxis_title='Day of Week',
    xaxis=dict(dtick=1)
)
fig.write_html('approval_rate_heatmap.html')
print("Interactive approval rate heatmap saved as 'approval_rate_heatmap.html'")

# Create comprehensive index.html for GitHub Pages
# Consolidate interactive visualizations into a single dashboard for stakeholder communication; 
# enable easy navigation and sharing of analytical findings without requiring technical setup.
index_html = """<!DOCTYPE html>
<html>
<head>
    <title>Process Mining Analysis Dashboard</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 0;
            background: #f5f5f5; 
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            margin: 0;
            font-size: 2.5em;
        }
        .subtitle {
            margin-top: 10px;
            font-size: 1.1em;
            opacity: 0.9;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .chart { 
            margin: 20px 0; 
            padding: 25px; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .chart:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .chart h2 {
            color: #333;
            margin-top: 0;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .chart iframe {
            border: none;
            border-radius: 8px;
        }
        .section-header {
            background: white;
            margin: 30px 0 20px 0;
            padding: 20px 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .section-header h2 {
            margin: 0;
            color: #667eea;
            font-size: 1.8em;
        }
        .section-header p {
            margin: 10px 0 0 0;
            color: #666;
            font-size: 1.1em;
        }
        .download-section {
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .download-section h3 {
            margin-top: 0;
            color: #333;
        }
        .download-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .download-link {
            display: block;
            padding: 12px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .download-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
            background: white;
            margin-top: 40px;
            border-radius: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Process Mining Analysis Dashboard</h1>
        <div class="subtitle">Comprehensive Financial Loan Application Process Analysis</div>
    </div>
    
    <div class="container">
        <!-- Process Flow Section -->
        <div class="section-header">
            <h2>Process Flow Analysis</h2>
            <p>Visualize the complete loan application process flow with frequency metrics</p>
        </div>
        
        <div class="chart">
            <h2>Interactive Process Map (Directly-Follows Graph)</h2>
            <iframe src="process_map_dfg.html" width="100%" height="700"></iframe>
        </div>
        
        <!-- Performance Metrics Section -->
        <div class="section-header">
            <h2>Performance Metrics</h2>
            <p>Analyze throughput times, waiting times, and activity durations</p>
        </div>
        
        <div class="chart">
            <h2>Throughput Time Analysis</h2>
            <iframe src="throughput_time_analysis.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart">
            <h2>Waiting Time Analysis</h2>
            <iframe src="waiting_time_analysis.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart">
            <h2>Activity Duration Analysis</h2>
            <iframe src="activity_duration_analysis.html" width="100%" height="600"></iframe>
        </div>
        
        <!-- Quality Metrics Section -->
        <div class="section-header">
            <h2>Quality Metrics</h2>
            <p>Identify rework patterns and process inefficiencies</p>
        </div>
        
        <div class="chart">
            <h2>Rework Analysis</h2>
            <iframe src="rework_analysis.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart">
            <h2>Rework Comparison by Outcome</h2>
            <iframe src="rework_comparison_by_outcome.html" width="100%" height="600"></iframe>
        </div>
        
        <!-- Outcome Analysis Section -->
        <div class="section-header">
            <h2>Outcome Analysis</h2>
            <p>Compare approved, declined, and cancelled cases</p>
        </div>
        
        <div class="chart">
            <h2>Case Outcome Comparison</h2>
            <iframe src="approved_vs_declined_vs_cancelled_comparison.html" width="100%" height="600"></iframe>
        </div>
        
        <!-- Resource Analysis Section -->
        <div class="section-header">
            <h2>Resource Analysis</h2>
            <p>Examine workload distribution and resource utilization</p>
        </div>
        
        <div class="chart">
            <h2>Resource Workload Analysis</h2>
            <iframe src="resource_workload_analysis.html" width="100%" height="600"></iframe>
        </div>
        
        <!-- Variant Analysis Section -->
        <div class="section-header">
            <h2>Variant Analysis</h2>
            <p>Explore process complexity and path diversity</p>
        </div>
        
        <div class="chart">
            <h2>Process Variant Distribution</h2>
            <iframe src="variant_distribution.html" width="100%" height="600"></iframe>
        </div>
        
        <!-- Temporal Patterns Section -->
        <div class="section-header">
            <h2>Temporal Patterns</h2>
            <p>Discover time-based trends and approval patterns</p>
        </div>
        
        <div class="chart">
            <h2>Outcomes by Day of Week</h2>
            <iframe src="outcomes_by_day_of_week.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart">
            <h2>Outcomes by Hour of Day</h2>
            <iframe src="outcomes_by_hour_of_day.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart">
            <h2>Approval Rate Heatmap</h2>
            <iframe src="approval_rate_heatmap.html" width="100%" height="600"></iframe>
        </div>
        
        <!-- Downloads Section -->
        <div class="download-section">
            <h3>Download Individual Visualizations</h3>
            <div class="download-grid">
                <a href="process_map_dfg.html" target="_blank" class="download-link">Process Map (DFG)</a>
                <a href="throughput_time_analysis.html" target="_blank" class="download-link">Throughput Time</a>
                <a href="waiting_time_analysis.html" target="_blank" class="download-link">Waiting Time</a>
                <a href="activity_duration_analysis.html" target="_blank" class="download-link">Activity Duration</a>
                <a href="rework_analysis.html" target="_blank" class="download-link">Rework Analysis</a>
                <a href="rework_comparison_by_outcome.html" target="_blank" class="download-link">Rework by Outcome</a>
                <a href="approved_vs_declined_vs_cancelled_comparison.html" target="_blank" class="download-link">Outcome Comparison</a>
                <a href="resource_workload_analysis.html" target="_blank" class="download-link">Resource Workload</a>
                <a href="variant_distribution.html" target="_blank" class="download-link">Variant Distribution</a>
                <a href="outcomes_by_day_of_week.html" target="_blank" class="download-link">Outcomes by Day</a>
                <a href="outcomes_by_hour_of_day.html" target="_blank" class="download-link">Outcomes by Hour</a>
                <a href="approval_rate_heatmap.html" target="_blank" class="download-link">Approval Heatmap</a>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated with PM4Py | Interactive visualizations powered by Plotly</p>
            <p>© 2024 Process Mining Analysis Dashboard</p>
        </div>
    </div>
</body>
</html>
"""

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(index_html)
print("\index.html created for GitHub Pages.")