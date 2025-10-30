"""
AgentPulse AI - Main Streamlit Application
Enterprise QA & Coaching Platform with CCRE
"""

import streamlit as st
import pandas as pd
import polars as pl
import time
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from modules.auth import require_authentication, logout, get_current_user, init_session_state
from modules.ccre_engine import PIIRedactor, CCRERuleEngine
from modules.llm_coaching import LLMCoachingEngine, LLMConfig
from modules.analytics import AnalyticsEngine, PrebuiltDashboards
from modules.export_manager import ExportManager


# Page configuration
st.set_page_config(
    page_title="AgentPulse AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_app_state():
    """Initialize application session state"""
    init_session_state()
    
    if 'classified_data' not in st.session_state:
        st.session_state['classified_data'] = None
    if 'rules_df' not in st.session_state:
        st.session_state['rules_df'] = None
    if 'analytics_engine' not in st.session_state:
        st.session_state['analytics_engine'] = None
    if 'coaching_results' not in st.session_state:
        st.session_state['coaching_results'] = {}


def load_custom_css():
    """Load custom CSS for main app"""
    css = """
    <style>
        /* Main app styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        .stProgress > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 10px 20px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Initialize
    init_app_state()
    
    # Authentication
    # Try to get background from secrets, fallback to None
    background_url = st.secrets.get('BACKGROUND_IMAGE_URL', None) if hasattr(st, 'secrets') else None
    
    if not require_authentication(background_url):
        return
    
    # Load custom CSS
    load_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 AgentPulse AI")
        st.markdown(f"**User:** {get_current_user()}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        st.markdown("Use tabs to navigate through the workflow")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **AgentPulse AI** uses the Context Clustered Rule Engine (CCRE) for:
        - Hierarchical classification
        - 96% accuracy
        - Explainable results
        - LLM-powered coaching
        """)
    
    # Main header
    st.markdown('<div class="main-header"><h1>🚀 AgentPulse AI</h1><p>Enterprise QA & Coaching Platform</p></div>', unsafe_allow_html=True)
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload", 
        "🔍 Classify", 
        "📊 Analyze", 
        "🎓 Coach", 
        "📤 Export"
    ])
    
    with tab1:
        upload_tab()
    
    with tab2:
        classify_tab()
    
    with tab3:
        analyze_tab()
    
    with tab4:
        coaching_tab()
    
    with tab5:
        export_tab()


def upload_tab():
    """File upload tab"""
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Transcripts")
        
        transcript_file = st.file_uploader(
            "Upload transcript file (CSV or Excel)",
            type=['csv', 'xlsx'],
            help="File can contain up to 200K rows (200MB)"
        )
        
        if transcript_file:
            try:
                # Load file
                if transcript_file.name.endswith('.csv'):
                    df = pd.read_csv(transcript_file)
                else:
                    df = pd.read_excel(transcript_file)
                
                st.success(f"✅ Loaded {len(df):,} transcripts")
                
                # Show preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Detect transcript column
                possible_cols = ['transcript', 'text', 'conversation', 'message']
                transcript_col = None
                for col in possible_cols:
                    if col in df.columns:
                        transcript_col = col
                        break
                
                if not transcript_col:
                    transcript_col = st.selectbox("Select transcript column:", df.columns.tolist())
                else:
                    st.info(f"📝 Using column: **{transcript_col}**")
                
                # Store in session state
                st.session_state['upload_df'] = df
                st.session_state['transcript_col'] = transcript_col
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.markdown("### Upload Rules")
        
        rules_file = st.file_uploader(
            "Upload rules file (CSV)",
            type=['csv'],
            help="Upload rules_embedded.csv or leave empty to use default"
        )
        
        if rules_file:
            try:
                rules_df = pd.read_csv(rules_file)
                st.success(f"✅ Loaded {len(rules_df):,} rules")
                st.session_state['rules_df'] = rules_df
            except Exception as e:
                st.error(f"❌ Error loading rules: {str(e)}")
        else:
            # Try to load default rules
            default_rules_path = "/mnt/user-data/uploads/rules_embedded.csv"
            if Path(default_rules_path).exists():
                rules_df = pd.read_csv(default_rules_path)
                st.info(f"ℹ️ Using default rules ({len(rules_df):,} rules)")
                st.session_state['rules_df'] = rules_df


def classify_tab():
    """Classification tab"""
    st.header("🔍 Classify Transcripts")
    
    if 'upload_df' not in st.session_state or st.session_state['upload_df'] is None:
        st.warning("⚠️ Please upload transcript data first")
        return
    
    if 'rules_df' not in st.session_state or st.session_state['rules_df'] is None:
        st.warning("⚠️ Please upload rules file first")
        return
    
    df = st.session_state['upload_df']
    rules_df = st.session_state['rules_df']
    transcript_col = st.session_state.get('transcript_col', 'transcript')
    
    st.markdown(f"### Ready to classify {len(df):,} transcripts")
    
    # Classification options
    col1, col2 = st.columns(2)
    
    with col1:
        enable_pii_redaction = st.checkbox("Enable PII Redaction", value=True)
    
    with col2:
        batch_size = st.select_slider(
            "Batch size",
            options=[1000, 5000, 10000, 20000],
            value=10000,
            help="Process in batches to manage memory"
        )
    
    # Classify button
    if st.button("🚀 Start Classification", type="primary", use_container_width=True):
        classify_transcripts(df, rules_df, transcript_col, enable_pii_redaction, batch_size)


def classify_transcripts(df, rules_df, transcript_col, enable_pii, batch_size):
    """Run classification with progress tracking"""
    
    start_time = time.time()
    
    # Initialize engines
    pii_redactor = PIIRedactor() if enable_pii else None
    ccre_engine = CCRERuleEngine(rules_df)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total = len(df)
    
    # Process in batches
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = df.iloc[start_idx:end_idx]
        
        status_text.text(f"Processing transcripts {start_idx+1:,} to {end_idx:,}...")
        
        for idx, row in batch.iterrows():
            transcript_text = str(row[transcript_col])
            
            # PII Redaction
            if pii_redactor:
                redacted_text = pii_redactor.redact(transcript_text)
            else:
                redacted_text = transcript_text
            
            # Classification
            result = ccre_engine.classify_transcript(idx, redacted_text)
            
            # Add original data
            result_row = row.to_dict()
            result_row.update(result)
            result_row['redacted_transcript'] = redacted_text
            
            results.append(result_row)
        
        # Update progress
        progress = end_idx / total
        progress_bar.progress(progress)
    
    # Create DataFrame
    classified_df = pd.DataFrame(results)
    
    # Convert matched_keywords to string
    if 'matched_keywords' in classified_df.columns:
        classified_df['matched_keywords'] = classified_df['matched_keywords'].apply(
            lambda x: '|'.join(x) if isinstance(x, list) else ''
        )
    
    # Save to session state
    st.session_state['classified_data'] = classified_df
    
    # Save to parquet
    output_path = Path("/home/claude/agentpulse_ai/data/classified_transcripts.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classified_df.to_parquet(output_path, index=False)
    
    # Initialize analytics engine
    st.session_state['analytics_engine'] = AnalyticsEngine(str(output_path))
    
    elapsed_time = time.time() - start_time
    
    # Success message
    progress_bar.progress(1.0)
    status_text.empty()
    
    st.success(f"""
    ✅ **Classification Complete!**
    
    - Processed: {len(classified_df):,} transcripts
    - Time: {elapsed_time:.1f} seconds
    - Average: {(elapsed_time/len(classified_df)*1000):.1f} ms per transcript
    """)
    
    # Show sample results
    st.markdown("#### Sample Results")
    st.dataframe(
        classified_df[['category', 'subcategory', 'confidence', 'resolve_reason']].head(10),
        use_container_width=True
    )


def analyze_tab():
    """Analytics tab"""
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.get('analytics_engine') is None:
        st.warning("⚠️ Please classify data first")
        return
    
    analytics = st.session_state['analytics_engine']
    
    # Summary stats
    summary = analytics.get_summary_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transcripts", f"{summary['total_transcripts']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Categories", summary['unique_categories'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Agents", summary['unique_agents'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Category distribution
    st.markdown("### Category Distribution")
    cat_dist = analytics.get_category_distribution()
    st.bar_chart(cat_dist.set_index('category')['count'])
    
    # Detailed tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Categories")
        st.dataframe(cat_dist.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### Resolution Reasons")
        resolve_dist = analytics.get_resolve_reason_distribution()
        st.dataframe(resolve_dist, use_container_width=True)
    
    # Agent performance
    st.markdown("### Agent Performance")
    agent_perf = analytics.get_agent_performance()
    st.dataframe(agent_perf, use_container_width=True)


def coaching_tab():
    """LLM coaching tab"""
    st.header("🎓 AI Coaching")
    
    if st.session_state.get('classified_data') is None:
        st.warning("⚠️ Please classify data first")
        return
    
    classified_df = st.session_state['classified_data']
    
    # LLM Configuration
    st.markdown("### LLM Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        provider = st.radio(
            "Provider",
            ["OpenRouter (Free Models)", "Local LLM (LM Studio / Ollama)"],
            horizontal=True
        )
        
        if provider == "OpenRouter (Free Models)":
            model = st.selectbox(
                "Select Model",
                LLMCoachingEngine.OPENROUTER_FREE_MODELS,
                help="Free models from OpenRouter"
            )
            
            # Get API key
            api_key = st.secrets.get('OPENROUTER_API_KEY', '') if hasattr(st, 'secrets') else ''
            
            if not api_key:
                api_key = st.text_input("OpenRouter API Key", type="password")
            
            if not api_key:
                st.warning("⚠️ Please provide OpenRouter API key")
                return
            
            config = LLMConfig(
                provider='openrouter',
                model=model,
                api_key=api_key
            )
        
        else:
            model_name = st.text_input("Model Name", value="llama2")
            endpoint = st.text_input(
                "Endpoint URL",
                value="http://localhost:1234/v1/chat/completions",
                help="LM Studio: http://localhost:1234/v1/chat/completions"
            )
            
            config = LLMConfig(
                provider='local',
                model=model_name,
                endpoint=endpoint
            )
    
    with col2:
        st.markdown("#### Token Info")
        if provider == "OpenRouter (Free Models)":
            context_limit = LLMCoachingEngine.OPENROUTER_FREE_MODELS
            st.info(f"Context: ~8K-64K tokens")
        else:
            st.info("Context: ~4K tokens")
    
    # Agent selection
    st.markdown("### Select Agents for Coaching")
    
    # Get unique agents
    if 'agent_name' in classified_df.columns:
        agents = classified_df['agent_name'].dropna().unique().tolist()
        selected_agents = st.multiselect("Agents", agents, max_selections=5)
        
        if selected_agents and st.button("🚀 Generate Coaching", type="primary"):
            generate_coaching(classified_df, selected_agents, config)
    else:
        st.warning("⚠️ No agent_name column found in data")


def generate_coaching(df, agents, config):
    """Generate coaching insights"""
    
    llm_engine = LLMCoachingEngine(config)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, agent in enumerate(agents):
        status_text.text(f"Generating coaching for {agent}...")
        
        # Get agent data
        agent_df = df[df['agent_name'] == agent]
        
        # Prepare data
        transcripts = agent_df.to_dict('records')[:5]  # Sample 5
        
        categories = agent_df['category'].value_counts().to_dict()
        
        metrics = {
            'total_calls': len(agent_df),
            'avg_confidence': agent_df['confidence'].mean(),
            'unique_categories': agent_df['category'].nunique()
        }
        
        # Generate coaching
        result = llm_engine.generate_coaching(agent, transcripts, metrics, categories)
        
        st.session_state['coaching_results'][agent] = result
        
        progress_bar.progress((i + 1) / len(agents))
    
    status_text.empty()
    progress_bar.empty()
    
    st.success("✅ Coaching generated!")
    
    # Display results
    for agent in agents:
        result = st.session_state['coaching_results'].get(agent)
        
        if result and result.get('success'):
            with st.expander(f"📋 Coaching for {agent}", expanded=True):
                coaching = result['coaching']
                
                st.markdown(f"**Root Cause:** {coaching.get('root_cause', 'N/A')}")
                
                st.markdown("**Coaching Points:**")
                for point in coaching.get('coaching_points', []):
                    st.markdown(f"- {point}")
                
                if coaching.get('sample_script'):
                    st.markdown(f"**Sample Script:**\n{coaching['sample_script']}")
                
                st.markdown(f"**Priority:** {coaching.get('priority', 'Medium')}")


def export_tab():
    """Export tab"""
    st.header("📤 Export Results")
    
    if st.session_state.get('classified_data') is None:
        st.warning("⚠️ No classified data available")
        return
    
    classified_df = st.session_state['classified_data']
    analytics = st.session_state.get('analytics_engine')
    
    st.markdown(f"### Export {len(classified_df):,} classified transcripts")
    
    # Export options
    formats = st.multiselect(
        "Select formats",
        ['CSV', 'Excel', 'Parquet', 'HTML'],
        default=['CSV', 'HTML']
    )
    
    if st.button("📥 Export", type="primary", use_container_width=True):
        export_manager = ExportManager()
        
        # Get summary data
        if analytics:
            summary_data = analytics.get_dashboard_data()
        else:
            summary_data = {}
        
        # Export
        with st.spinner("Exporting..."):
            results = export_manager.batch_export(
                classified_df,
                summary_data,
                [f.lower() for f in formats],
                "agentpulse_export"
            )
        
        st.success("✅ Export complete!")
        
        # Download buttons
        for format_name, file_path in results.items():
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    st.download_button(
                        f"📥 Download {format_name.upper()}",
                        f,
                        file_name=Path(file_path).name,
                        mime=get_mime_type(format_name)
                    )


def get_mime_type(format_name):
    """Get MIME type for format"""
    mime_types = {
        'csv': 'text/csv',
        'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'parquet': 'application/octet-stream',
        'html': 'text/html',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    return mime_types.get(format_name, 'application/octet-stream')


if __name__ == "__main__":
    main()
