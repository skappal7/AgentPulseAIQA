"""
Analytics Module - DuckDB Integration
Provides SQL queries and analytics on classified transcript data
Handles missing columns gracefully
"""

import duckdb
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


class AnalyticsEngine:
    """DuckDB-based analytics engine for transcript analysis"""
    
    def __init__(self, parquet_path: Optional[str] = None):
        """
        Initialize analytics engine
        
        Args:
            parquet_path: Path to parquet file with classified data
        """
        self.conn = duckdb.connect(':memory:')
        self.parquet_path = parquet_path
        self.available_columns = []
        
        if parquet_path:
            self.load_data(parquet_path)
    
    def load_data(self, parquet_path: str):
        """Load parquet data into DuckDB"""
        self.parquet_path = parquet_path
        
        # Create table from parquet
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE transcripts AS 
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        
        # Get available columns
        self.available_columns = [row[0] for row in self.conn.execute("PRAGMA table_info('transcripts')").fetchall()]
    
    def _has_column(self, column_name: str) -> bool:
        """Check if column exists in the table"""
        return column_name in self.available_columns
    
    def get_category_distribution(self) -> pd.DataFrame:
        """Get distribution of categories"""
        query = """
            SELECT 
                category,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM transcripts
            GROUP BY category
            ORDER BY count DESC
        """
        return self.conn.execute(query).df()
    
    def get_subcategory_distribution(self, category: Optional[str] = None) -> pd.DataFrame:
        """Get distribution of subcategories, optionally filtered by category"""
        if category:
            query = f"""
                SELECT 
                    category,
                    subcategory,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM transcripts
                WHERE category = '{category}'
                GROUP BY category, subcategory
                ORDER BY count DESC
            """
        else:
            query = """
                SELECT 
                    category,
                    subcategory,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM transcripts
                GROUP BY category, subcategory
                ORDER BY count DESC
            """
        return self.conn.execute(query).df()
    
    def get_agent_performance(self) -> pd.DataFrame:
        """Get agent-level performance metrics - returns empty if no agent_name column"""
        if not self._has_column('agent_name'):
            return pd.DataFrame()
        
        query = """
            SELECT 
                agent_name,
                COUNT(*) as total_calls,
                ROUND(AVG(confidence), 3) as avg_confidence,
                COUNT(DISTINCT category) as unique_categories,
                COUNT(DISTINCT subcategory) as unique_subcategories,
                ROUND(AVG(num_rules_activated), 1) as avg_rules_activated
            FROM transcripts
            WHERE agent_name IS NOT NULL AND agent_name != ''
            GROUP BY agent_name
            ORDER BY total_calls DESC
        """
        try:
            return self.conn.execute(query).df()
        except:
            return pd.DataFrame()
    
    def get_agent_category_breakdown(self, agent_name: str) -> pd.DataFrame:
        """Get category breakdown for specific agent"""
        if not self._has_column('agent_name'):
            return pd.DataFrame()
        
        query = f"""
            SELECT 
                category,
                subcategory,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM transcripts
            WHERE agent_name = '{agent_name}'
            GROUP BY category, subcategory
            ORDER BY count DESC
        """
        try:
            return self.conn.execute(query).df()
        except:
            return pd.DataFrame()
    
    def get_low_confidence_transcripts(self, threshold: float = 0.6) -> pd.DataFrame:
        """Get transcripts with confidence below threshold"""
        # Build SELECT dynamically based on available columns
        select_cols = ['transcript_id', 'category', 'subcategory', 'confidence', 'resolve_reason']
        if self._has_column('agent_name'):
            select_cols.append('agent_name')
        
        select_str = ', '.join(select_cols)
        
        query = f"""
            SELECT {select_str}
            FROM transcripts
            WHERE confidence < {threshold}
            ORDER BY confidence ASC
            LIMIT 100
        """
        return self.conn.execute(query).df()
    
    def get_resolve_reason_distribution(self) -> pd.DataFrame:
        """Get distribution of resolution reasons"""
        query = """
            SELECT 
                resolve_reason,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM transcripts
            GROUP BY resolve_reason
            ORDER BY count DESC
        """
        return self.conn.execute(query).df()
    
    def get_top_n_agents_by_category(self, category: str, n: int = 10) -> pd.DataFrame:
        """Get top N agents for a specific category"""
        if not self._has_column('agent_name'):
            return pd.DataFrame()
        
        query = f"""
            SELECT 
                agent_name,
                COUNT(*) as count,
                ROUND(AVG(confidence), 3) as avg_confidence
            FROM transcripts
            WHERE category = '{category}' AND agent_name IS NOT NULL AND agent_name != ''
            GROUP BY agent_name
            ORDER BY count DESC
            LIMIT {n}
        """
        try:
            return self.conn.execute(query).df()
        except:
            return pd.DataFrame()
    
    def get_comparison_metrics(self, agent_names: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple agents"""
        if not self._has_column('agent_name'):
            return pd.DataFrame()
        
        agent_list = "', '".join(agent_names)
        query = f"""
            SELECT 
                agent_name,
                COUNT(*) as total_calls,
                ROUND(AVG(confidence), 3) as avg_confidence,
                COUNT(DISTINCT category) as unique_categories,
                MODE(category) as most_common_category,
                ROUND(AVG(num_rules_activated), 1) as avg_rules_activated
            FROM transcripts
            WHERE agent_name IN ('{agent_list}')
            GROUP BY agent_name
        """
        try:
            return self.conn.execute(query).df()
        except:
            return pd.DataFrame()
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query"""
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            raise ValueError(f"Query execution failed: {str(e)}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get overall summary statistics"""
        total = self.conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        
        avg_confidence = self.conn.execute(
            "SELECT ROUND(AVG(confidence), 3) FROM transcripts"
        ).fetchone()[0]
        
        unique_categories = self.conn.execute(
            "SELECT COUNT(DISTINCT category) FROM transcripts"
        ).fetchone()[0]
        
        unique_subcategories = self.conn.execute(
            "SELECT COUNT(DISTINCT subcategory) FROM transcripts"
        ).fetchone()[0]
        
        # Check if agent_name column exists
        if self._has_column('agent_name'):
            try:
                unique_agents = self.conn.execute(
                    "SELECT COUNT(DISTINCT agent_name) FROM transcripts WHERE agent_name IS NOT NULL AND agent_name != ''"
                ).fetchone()[0]
            except:
                unique_agents = 0
        else:
            unique_agents = 0
        
        return {
            'total_transcripts': total,
            'avg_confidence': avg_confidence,
            'unique_categories': unique_categories,
            'unique_subcategories': unique_subcategories,
            'unique_agents': unique_agents,
            'has_agent_data': self._has_column('agent_name') and unique_agents > 0
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        summary = self.get_summary_stats()
        
        dashboard = {
            'summary': summary,
            'category_dist': self.get_category_distribution().to_dict('records'),
            'subcategory_dist': self.get_subcategory_distribution().to_dict('records')[:20],
            'resolve_reasons': self.get_resolve_reason_distribution().to_dict('records')
        }
        
        # Only add agent performance if agent data exists
        if summary.get('has_agent_data', False):
            agent_perf = self.get_agent_performance()
            if not agent_perf.empty:
                dashboard['agent_performance'] = agent_perf.to_dict('records')[:20]
            else:
                dashboard['agent_performance'] = []
        else:
            dashboard['agent_performance'] = []
        
        return dashboard
    
    def export_to_csv(self, output_path: str):
        """Export transcripts table to CSV"""
        query = "SELECT * FROM transcripts"
        df = self.conn.execute(query).df()
        df.to_csv(output_path, index=False)
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PrebuiltDashboards:
    """Prebuilt analytical dashboards"""
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.engine = analytics_engine
    
    def quality_issues_dashboard(self) -> Dict[str, Any]:
        """Dashboard for identifying quality issues"""
        
        # Low confidence cases
        low_confidence = self.engine.get_low_confidence_transcripts(threshold=0.6)
        
        dashboard = {
            'low_confidence_cases': low_confidence.to_dict('records')
        }
        
        # Only add agent-specific dashboards if agent data exists
        if self.engine._has_column('agent_name'):
            try:
                # Agents with high DPA incidents
                dpa_agents = self.engine.execute_custom_query("""
                    SELECT 
                        agent_name,
                        COUNT(*) as dpa_count,
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY agent_name), 2) as dpa_percentage
                    FROM transcripts
                    WHERE subcategory = 'Data Protection Access' AND agent_name IS NOT NULL AND agent_name != ''
                    GROUP BY agent_name
                    HAVING COUNT(*) > 2
                    ORDER BY dpa_count DESC
                """)
                dashboard['dpa_high_agents'] = dpa_agents.to_dict('records')
                
                # Verification issues
                verification_issues = self.engine.execute_custom_query("""
                    SELECT 
                        agent_name,
                        COUNT(*) as verification_count
                    FROM transcripts
                    WHERE subcategory = 'Verification Issue' AND agent_name IS NOT NULL AND agent_name != ''
                    GROUP BY agent_name
                    ORDER BY verification_count DESC
                    LIMIT 10
                """)
                dashboard['verification_issues'] = verification_issues.to_dict('records')
            except:
                dashboard['dpa_high_agents'] = []
                dashboard['verification_issues'] = []
        else:
            dashboard['dpa_high_agents'] = []
            dashboard['verification_issues'] = []
        
        return dashboard
    
    def coaching_priorities_dashboard(self) -> Dict[str, Any]:
        """Dashboard for coaching priorities"""
        
        if not self.engine._has_column('agent_name'):
            return {'coaching_priorities': []}
        
        try:
            # Agents needing coaching by category
            coaching_needs = self.engine.execute_custom_query("""
                SELECT 
                    agent_name,
                    category,
                    COUNT(*) as issue_count,
                    ROUND(AVG(confidence), 3) as avg_confidence
                FROM transcripts
                WHERE agent_name IS NOT NULL AND agent_name != ''
                GROUP BY agent_name, category
                HAVING COUNT(*) >= 3
                ORDER BY issue_count DESC, avg_confidence ASC
                LIMIT 20
            """)
            
            return {
                'coaching_priorities': coaching_needs.to_dict('records')
            }
        except:
            return {'coaching_priorities': []}
    
    def category_trends_dashboard(self, limit: int = 5) -> Dict[str, Any]:
        """Dashboard for category trends"""
        
        # Top recurring issues
        top_issues = self.engine.execute_custom_query(f"""
            SELECT 
                category,
                subcategory,
                COUNT(*) as frequency,
                ROUND(AVG(confidence), 3) as avg_confidence,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM transcripts
            GROUP BY category, subcategory
            ORDER BY frequency DESC
            LIMIT {limit}
        """)
        
        return {
            'top_issues': top_issues.to_dict('records')
        }
