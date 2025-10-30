"""
Analytics Module - DuckDB Integration
Provides SQL queries and analytics on classified transcript data
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
        """Get agent-level performance metrics"""
        query = """
            SELECT 
                agent_name,
                COUNT(*) as total_calls,
                ROUND(AVG(confidence), 3) as avg_confidence,
                COUNT(DISTINCT category) as unique_categories,
                COUNT(DISTINCT subcategory) as unique_subcategories,
                ROUND(AVG(num_rules_activated), 1) as avg_rules_activated
            FROM transcripts
            WHERE agent_name IS NOT NULL
            GROUP BY agent_name
            ORDER BY total_calls DESC
        """
        return self.conn.execute(query).df()
    
    def get_agent_category_breakdown(self, agent_name: str) -> pd.DataFrame:
        """Get category breakdown for specific agent"""
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
        return self.conn.execute(query).df()
    
    def get_low_confidence_transcripts(self, threshold: float = 0.6) -> pd.DataFrame:
        """Get transcripts with confidence below threshold"""
        query = f"""
            SELECT 
                transcript_id,
                category,
                subcategory,
                confidence,
                resolve_reason,
                agent_name
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
        query = f"""
            SELECT 
                agent_name,
                COUNT(*) as count,
                ROUND(AVG(confidence), 3) as avg_confidence
            FROM transcripts
            WHERE category = '{category}' AND agent_name IS NOT NULL
            GROUP BY agent_name
            ORDER BY count DESC
            LIMIT {n}
        """
        return self.conn.execute(query).df()
    
    def get_comparison_metrics(self, agent_names: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple agents"""
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
        return self.conn.execute(query).df()
    
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
        
        unique_agents = self.conn.execute(
            "SELECT COUNT(DISTINCT agent_name) FROM transcripts WHERE agent_name IS NOT NULL"
        ).fetchone()[0]
        
        return {
            'total_transcripts': total,
            'avg_confidence': avg_confidence,
            'unique_categories': unique_categories,
            'unique_subcategories': unique_subcategories,
            'unique_agents': unique_agents
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'summary': self.get_summary_stats(),
            'category_dist': self.get_category_distribution().to_dict('records'),
            'subcategory_dist': self.get_subcategory_distribution().to_dict('records')[:20],
            'resolve_reasons': self.get_resolve_reason_distribution().to_dict('records'),
            'agent_performance': self.get_agent_performance().to_dict('records')[:20]
        }
    
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
        
        # Agents with high DPA incidents
        dpa_agents = self.engine.execute_custom_query("""
            SELECT 
                agent_name,
                COUNT(*) as dpa_count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY agent_name), 2) as dpa_percentage
            FROM transcripts
            WHERE subcategory = 'Data Protection Access' AND agent_name IS NOT NULL
            GROUP BY agent_name
            HAVING COUNT(*) > 2
            ORDER BY dpa_count DESC
        """)
        
        # Verification issues
        verification_issues = self.engine.execute_custom_query("""
            SELECT 
                agent_name,
                COUNT(*) as verification_count
            FROM transcripts
            WHERE subcategory = 'Verification Issue' AND agent_name IS NOT NULL
            GROUP BY agent_name
            ORDER BY verification_count DESC
            LIMIT 10
        """)
        
        return {
            'low_confidence_cases': low_confidence.to_dict('records'),
            'dpa_high_agents': dpa_agents.to_dict('records'),
            'verification_issues': verification_issues.to_dict('records')
        }
    
    def coaching_priorities_dashboard(self) -> Dict[str, Any]:
        """Dashboard for coaching priorities"""
        
        # Agents needing coaching by category
        coaching_needs = self.engine.execute_custom_query("""
            SELECT 
                agent_name,
                category,
                COUNT(*) as issue_count,
                ROUND(AVG(confidence), 3) as avg_confidence
            FROM transcripts
            WHERE agent_name IS NOT NULL
            GROUP BY agent_name, category
            HAVING COUNT(*) >= 3
            ORDER BY issue_count DESC, avg_confidence ASC
            LIMIT 20
        """)
        
        return {
            'coaching_priorities': coaching_needs.to_dict('records')
        }
    
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
